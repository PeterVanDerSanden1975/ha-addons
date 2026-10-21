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

MODEL_URL = (
    "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/"
    "yamnet/classification/tflite/1.tflite"
)
CLASSMAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/research/"
    "audioset/yamnet/yamnet_class_map.csv"
)

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
    }
    p = Path("/data/options.json")
    if p.exists():
        try:
            opts.update(json.loads(p.read_text()))
        except Exception as e:  # noqa: BLE001
            _LOGGER.warning("options.json niet leesbaar: %s", e)
    return opts


def ensure_file(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOGGER.info("Downloaden %s -> %s", url, path)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(path)


# ----------------------------------------------------------------------------
class YamnetClassifier:
    def __init__(self, threads: int = 2) -> None:
        from tflite_runtime.interpreter import Interpreter

        ensure_file(MODEL_PATH, MODEL_URL)
        ensure_file(CLASSMAP_PATH, CLASSMAP_URL)

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
            text = await asyncio.get_running_loop().run_in_executor(None, self._verwerk)
            await self.write_event(Transcript(text=text).event())
            _LOGGER.info("Resultaat: %s", text)
            return False

        return True

    def _verwerk(self) -> str:
        t0 = time.monotonic()
        raw = bytes(self.audio)
        if len(raw) < self.width * 1600:          # < 0,1 s: niks mee te beginnen
            return "none 0.00 (leeg)"
        x = self.clf.to_float16k(raw, self.rate, self.width, self.channels)
        bark, top, top_score = self.clf.classify(x)
        verdict = "bark" if top in BARK_CLASSES or bark >= 0.3 else "none"
        text = f"{verdict} {bark:.2f} ({top} {top_score:.2f})"
        _LOGGER.debug("fragment %.2fs, verwerkt in %.0f ms", len(x) / SAMPLE_RATE,
                      (time.monotonic() - t0) * 1000)

        if self.opts.get("bewaar_fragmenten") and (
            verdict == "bark" or not self.opts.get("bewaar_alleen_blaf")
        ):
            try:
                d = Path(self.opts.get("bewaar_map", "/share/blafdetectie"))
                d.mkdir(parents=True, exist_ok=True)
                stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                p = d / f"{stamp}_{verdict}_{bark:.2f}.wav"
                with wave.open(str(p), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SAMPLE_RATE)
                    w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
            except Exception as e:  # noqa: BLE001
                _LOGGER.warning("fragment niet opgeslagen: %s", e)
        return text


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

    uri = os.environ.get("WYOMING_URI", "tcp://0.0.0.0:10310")
    server = AsyncServer.from_uri(uri)
    _LOGGER.info("Wyoming-YAMNet luistert op %s", uri)
    await server.run(partial(YamnetHandler, info, clf, opts))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
