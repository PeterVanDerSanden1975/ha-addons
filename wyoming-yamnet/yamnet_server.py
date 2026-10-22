#!/usr/bin/env python3
"""
Wyoming-YAMNet: doet zich voor als Wyoming STT-server, maar classificeert het
binnenkomende audiofragment met Google YAMNet (521 AudioSet-klassen) en geeft
als 'transcript' terug:

    bark 0.87 (Bark)      -> blafgeluid, hoogste score van de hondklassen
    none 0.12 (Speech)    -> geen blaf; score van de hondklassen + de topklasse

Het keukendisplay (ESPHome voice_assistant.on_stt_end) parseert de eerste twee
tokens en beslist zelf of er gemeld wordt.
"""

import asyncio
import csv
import datetime as dt
import json
import logging
import os
import sys
import time
import urllib.request
import wave
from functools import partial
from pathlib import Path

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger("wyoming-yamnet")

DATA_DIR = Path(os.environ.get("YAMNET_DATA", "/data"))
MODEL_PATH = DATA_DIR / "yamnet.tflite"
CLASSMAP_PATH = DATA_DIR / "yamnet_class_map.csv"

# Zelfde YAMNet-lite-model (input float32 1x15600 @ 16 kHz, output 1x521 scores).
# De oude tfhub-URL geeft 403 sinds TF Hub naar Kaggle is verhuisd; MediaPipe
# host het model publiek. Meerdere bronnen op volgorde; lukt geen enkele, dan
# wordt gekeken of het bestand handmatig in /share/blafdetectie is gezet.
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite",
    "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite",
    "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/yamnet/classification/tflite/1.tflite",
]
CLASSMAP_URLS = [
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
    "https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv",
]
MANUAL_DIR = Path("/share/blafdetectie")   # handmatige fallback: yamnet.tflite / yamnet_class_map.csv hier neerzetten

SAMPLE_RATE = 16000
WINDOW = 15600          # 0.975 s, vaste invoerlengte van het YAMNet-lite-model
HOP = 7800              # 50 % overlap

# AudioSet-klassen die als 'blaf' tellen
BARK_CLASSES = {"Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling"}


# ----------------------------------------------------------------------------
def load_options() -> dict:
    opts = {
        "bewaar_fragmenten": True,
        "bewaar_map": "/share/blafdetectie",
        "bewaar_alleen_blaf": False,
        "threads": 2,
        "log_level": "info",
        "http_port": 10311,
    }
    p = Path("/data/options.json")
    if p.exists():
        try:
            opts.update(json.loads(p.read_text()))
        except Exception as e:  # noqa: BLE001
            _LOGGER.warning("options.json niet leesbaar: %s", e)
    return opts


def ensure_file(path: Path, urls: list[str], min_size: int = 1000) -> None:
    if path.exists() and path.stat().st_size >= min_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")

    # 1) handmatig neergezet bestand
    manual = MANUAL_DIR / path.name
    if manual.exists() and manual.stat().st_size >= min_size:
        _LOGGER.info("Gebruik handmatig bestand %s", manual)
        path.write_bytes(manual.read_bytes())
        return

    # 2) downloaden, bronnen op volgorde
    fouten = []
    for url in urls:
        try:
            _LOGGER.info("Downloaden %s -> %s", url, path)
            req = urllib.request.Request(url, headers={"User-Agent": "wyoming-yamnet/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            if tmp.stat().st_size < min_size:
                raise OSError(f"te klein ({tmp.stat().st_size} B)")
            tmp.replace(path)
            return
        except Exception as e:  # noqa: BLE001
            fouten.append(f"{url}: {e}")
            tmp.unlink(missing_ok=True)

    raise RuntimeError(
        f"{path.name} niet kunnen ophalen. Zet het bestand handmatig in {MANUAL_DIR} "
        f"en herstart de add-on. Geprobeerd:\n  " + "\n  ".join(fouten)
    )


# ----------------------------------------------------------------------------
class YamnetClassifier:
    def __init__(self, threads: int = 2) -> None:
        from tflite_runtime.interpreter import Interpreter

        ensure_file(MODEL_PATH, MODEL_URLS, min_size=1_000_000)
        ensure_file(CLASSMAP_PATH, CLASSMAP_URLS, min_size=5_000)

        self.names: list[str] = []
        with open(CLASSMAP_PATH, newline="") as f:
            for row in csv.DictReader(f):
                self.names.append(row["display_name"])
        self.bark_idx = [i for i, n in enumerate(self.names) if n in BARK_CLASSES]
        _LOGGER.info("Klassen geladen: %d, blafklassen: %s",
                     len(self.names), [self.names[i] for i in self.bark_idx])

        self.interp = Interpreter(model_path=str(MODEL_PATH), num_threads=threads)
        self.inp = self.interp.get_input_details()[0]
        # invoer naar vaste venstergrootte zetten (model heeft dynamische lengte)
        shape = [int(x) for x in self.inp["shape"]]
        if len(shape) == 1:
            shape = [WINDOW]
        else:
            shape[-1] = WINDOW
        self.interp.resize_tensor_input(self.inp["index"], shape, strict=False)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.in_shape = [int(x) for x in self.inp["shape"]]

        outs = self.interp.get_output_details()
        self.out_scores = next(o for o in outs if int(o["shape"][-1]) == len(self.names))
        _LOGGER.info("YAMNet klaar (input %s, threads %d)", self.in_shape, threads)

    # --- audio hulpfuncties ---------------------------------------------------
    @staticmethod
    def to_float16k(raw: bytes, rate: int, width: int, channels: int) -> np.ndarray:
        if width == 2:
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        if channels > 1:
            n = len(x) // channels
            x = x[: n * channels].reshape(n, channels).mean(axis=1)
        if rate != SAMPLE_RATE and len(x) > 1:
            n_out = int(round(len(x) * SAMPLE_RATE / rate))
            x = np.interp(
                np.linspace(0.0, 1.0, n_out, endpoint=False),
                np.linspace(0.0, 1.0, len(x), endpoint=False),
                x,
            ).astype(np.float32)
        return x

    # --- classificatie ---------------------------------------------------------
    def classify(self, x: np.ndarray) -> tuple[float, str, float]:
        """Geeft (blafscore, topklasse, topscore) over het hele fragment."""
        if len(x) < WINDOW:
            x = np.pad(x, (0, WINDOW - len(x)))
        best_bark = 0.0
        top_name, top_score = "?", 0.0
        for start in range(0, len(x) - WINDOW + 1, HOP):
            win = x[start : start + WINDOW]
            self.interp.set_tensor(self.inp["index"], win.reshape(self.in_shape))
            self.interp.invoke()
            scores = self.interp.get_tensor(self.out_scores["index"]).reshape(-1)
            b = float(scores[self.bark_idx].max())
            if b > best_bark:
                best_bark = b
            i = int(scores.argmax())
            if float(scores[i]) > top_score:
                top_score, top_name = float(scores[i]), self.names[i]
        return best_bark, top_name, top_score


# ----------------------------------------------------------------------------
class YamnetHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, clf: YamnetClassifier, opts: dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.clf = clf
        self.opts = opts
        self.audio = bytearray()
        self.rate, self.width, self.channels = SAMPLE_RATE, 2, 1

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            return True

        if Transcribe.is_type(event.type):
            return True

        if AudioStart.is_type(event.type):
            s = AudioStart.from_event(event)
            self.rate, self.width, self.channels = s.rate, s.width, s.channels
            self.audio = bytearray()
            return True

        if AudioChunk.is_type(event.type):
            c = AudioChunk.from_event(event)
            self.audio += c.audio
            return True

        if AudioStop.is_type(event.type):
            t0 = time.monotonic()
            text = await asyncio.get_running_loop().run_in_executor(None, self._verwerk)
            ms = (time.monotonic() - t0) * 1000
            try:
                await self.write_event(Transcript(text=text).event())
                _LOGGER.info("Resultaat: %s (%.0f ms)", text, ms)
            except (ConnectionResetError, BrokenPipeError, ConnectionError):
                _LOGGER.warning(
                    "Resultaat '%s' (%.0f ms) niet afgeleverd: HA sloot de verbinding "
                    "voordat het antwoord gelezen was (pipeline afgebroken?)", text, ms)
            return False

        return True

    def _verwerk(self) -> str:
        r = verwerk_fragment(self.clf, self.opts, bytes(self.audio),
                             self.rate, self.width, self.channels, bron="wyoming")
        return r["text"]


# ----------------------------------------------------------------------------
def verwerk_fragment(clf: YamnetClassifier, opts: dict, raw: bytes,
                     rate: int, width: int, channels: int, bron: str = "http",
                     extra: str = "") -> dict:
    """Classificeert ruwe PCM en slaat het fragment optioneel op. Gedeeld door
    het Wyoming- en het HTTP-pad."""
    t0 = time.monotonic()
    if len(raw) < width * channels * 1600:            # < 0,1 s: niks mee te beginnen
        return {"verdict": "none", "bark": 0.0, "top": "leeg", "top_score": 0.0,
                "seconds": 0.0, "ms": 0, "text": "none 0.00 (leeg 0.00)"}
    x = clf.to_float16k(raw, rate, width, channels)
    bark, top, top_score = clf.classify(x)
    verdict = "bark" if top in BARK_CLASSES or bark >= 0.3 else "none"
    ms = (time.monotonic() - t0) * 1000
    text = f"{verdict} {bark:.2f} ({top} {top_score:.2f})"
    _LOGGER.info("[%s] %s — fragment %.2f s, %.0f ms%s", bron, text, len(x) / SAMPLE_RATE, ms,
                 f" ({extra})" if extra else "")

    if opts.get("bewaar_fragmenten") and (verdict == "bark" or not opts.get("bewaar_alleen_blaf")):
        try:
            d = Path(opts.get("bewaar_map", "/share/blafdetectie"))
            d.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            p = d / f"{stamp}_{verdict}_{bark:.2f}_{top.replace(' ', '-')}.wav"
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        except Exception as e:  # noqa: BLE001
            _LOGGER.warning("fragment niet opgeslagen: %s", e)
    return {"verdict": verdict, "bark": round(float(bark), 3), "top": top,
            "top_score": round(float(top_score), 3), "seconds": round(len(x) / SAMPLE_RATE, 2),
            "ms": int(ms), "text": text}


# ----------------------------------------------------------------------------
# HTTP-endpoint voor het keukendisplay (ESPHome http_request.post, ruwe PCM):
#   POST /classify?rate=16000&bits=16&ch=1[&flanken=..&piek=..]   body = PCM
#   -> 200 {"verdict":"bark","bark":0.87,"top":"Bark","top_score":0.91,"seconds":3.5,"ms":84}
#   GET  /health -> 200 {"ok":true}
def maak_http_app(clf: YamnetClassifier, opts: dict):
    from aiohttp import web

    async def classify(request: "web.Request") -> "web.Response":
        try:
            rate = int(request.query.get("rate", SAMPLE_RATE))
            bits = int(request.query.get("bits", 16))
            ch = int(request.query.get("ch", 1))
        except ValueError:
            return web.json_response({"error": "rate/bits/ch ongeldig"}, status=400)
        if bits not in (8, 16, 32) or ch < 1 or rate < 8000:
            return web.json_response({"error": "rate/bits/ch buiten bereik"}, status=400)
        raw = await request.read()
        extra = ", ".join(f"{k}={v}" for k, v in request.query.items() if k in ("flanken", "piek"))
        r = await asyncio.get_running_loop().run_in_executor(
            None, verwerk_fragment, clf, opts, raw, rate, bits // 8, ch, "http", extra)
        r.pop("text", None)
        return web.json_response(r)

    async def health(_: "web.Request") -> "web.Response":
        return web.json_response({"ok": True, "model": "yamnet", "bark_classes": sorted(BARK_CLASSES)})

    app = web.Application(client_max_size=8 * 1024 * 1024)
    app.add_routes([web.post("/classify", classify), web.get("/health", health)])
    return app


# ----------------------------------------------------------------------------
async def main() -> None:
    opts = load_options()
    logging.basicConfig(
        level=getattr(logging, str(opts.get("log_level", "info")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    clf = YamnetClassifier(threads=int(opts.get("threads", 2)))

    info = Info(
        asr=[
            AsrProgram(
                name="yamnet",
                description="YAMNet geluidsclassificatie (blafdetectie)",
                attribution=Attribution(name="Google YAMNet", url="https://tfhub.dev/google/yamnet/1"),
                installed=True,
                version="1.0.0",
                models=[
                    AsrModel(
                        name="yamnet",
                        description="AudioSet 521 klassen; antwoord 'bark <score>' of 'none <score>'",
                        attribution=Attribution(name="Google", url="https://tfhub.dev/google/yamnet/1"),
                        installed=True,
                        languages=["nl", "en"],
                        version="1",
                    )
                ],
            )
        ]
    )

    from aiohttp import web
    http_port = int(opts.get("http_port", 10311))
    runner = web.AppRunner(maak_http_app(clf, opts))
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", http_port).start()
    _LOGGER.info("HTTP-endpoint luistert op http://0.0.0.0:%d/classify", http_port)

    uri = os.environ.get("WYOMING_URI", "tcp://0.0.0.0:10310")
    server = AsyncServer.from_uri(uri)
    _LOGGER.info("Wyoming-YAMNet luistert op %s", uri)
    await server.run(partial(YamnetHandler, info, clf, opts))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
