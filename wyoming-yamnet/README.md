# Wyoming YAMNet — blafverificatie voor het keukendisplay

HA OS add-on die Google YAMNet (521 AudioSet-klassen) op een audiofragment draait
en zegt of er een hond in zit. Twee ingangen:

- **HTTP (gebruikt door het keukendisplay, v53+)** — poort 10311:
  `POST /classify?rate=16000&bits=16&ch=1` met ruwe PCM als body →
  `{"verdict":"bark","bark":0.87,"top":"Bark","top_score":0.91,"seconds":3.5,"ms":84}`.
  Het display knipt zelf 1 s vóór + 2,5 s na de triggerflank uit zijn ringbuffer.
  Geen Assist-pipeline, geen HA-VAD, geen TTS.
- **Wyoming STT** — poort 10310, optioneel. Antwoord als 'transcript':
  `bark 0.87 (Bark 0.87)` / `none 0.04 (Speech 0.81)`. Alleen nodig als je het via
  een Assist-pipeline wilt gebruiken; voor het display niet meer.

`GET /health` geeft `{"ok":true,...}`.

## Installatie (HA OS, Pi 5)

1. Kopieer de map `wyoming-yamnet/` naar `/addons/wyoming-yamnet/` op de Pi
   (via de Samba- of SSH-add-on; `/addons` staat naast `/config`).
2. Instellingen → Add-ons → Add-on store → ⋮ rechtsboven → **Controleren op updates**.
   Onder *Lokale add-ons* verschijnt **Wyoming YAMNet** → Installeren (bouwt het
   image lokaal, ~2–3 min).
3. Opties laten staan, **Starten**. Bij de eerste start haalt de add-on het model
   (~4 MB) en de klassenlijst op naar `/data`. Log moet eindigen met
   `Wyoming-YAMNet luistert op tcp://0.0.0.0:10310`.
4. Klaar. Het display praat rechtstreeks met poort 10311 (`yamnet_url` in de
   substitutions van de display-yaml). Test vanaf een pc:
   `curl http://<pi-ip>:10311/health`.

De Wyoming-integratie en de pipeline "Blafdetectie" uit v48–v52 zijn niet meer
nodig en kunnen weg.

## Opties

| Optie | Standaard | Betekenis |
|---|---|---|
| `bewaar_fragmenten` | `true` | Elk beoordeeld fragment als WAV opslaan |
| `bewaar_map` | `/share/blafdetectie` | Map voor de WAV's (naam: `datum_tijd_bark|none_score.wav`) |
| `bewaar_alleen_blaf` | `false` | Alleen fragmenten met oordeel `bark` bewaren |
| `threads` | `2` | TFLite-threads |
| `log_level` | `info` | `debug` logt meer detail |
| `http_port` | `10311` | Poort van het HTTP-endpoint |

## Afstellen

- Luister de WAV's in `/share/blafdetectie` terug (Samba) en vergelijk met de score
  in de bestandsnaam. Zet **Blaf score minimum** op het display net onder de laagste
  score van echte blaffen.
- `Blaf kandidaten vandaag` vs `Blaffen vandaag` op het display laat zien hoeveel
  de Pi wegfiltert. Veel kandidaten zonder blaf: heuristiek op het display mag
  strenger (drempel omhoog of aantal omhoog). Blaffen die de heuristiek mist:
  drempel omlaag — de Pi vangt de vals-positieven toch af.
- Blafklassen: `Dog`, `Bark`, `Yip`, `Howl`, `Bow-wow`, `Growling`
  (bovenin `yamnet_server.py`, `BARK_CLASSES`).

## Werking

Het fragment (16 kHz mono) wordt in vensters van 0,975 s (50 % overlap) door YAMNet
gehaald; de blafscore is de hoogste score van de blafklassen over alle vensters.
Verwerking kost op een Pi 5 enkele tientallen ms per venster.
