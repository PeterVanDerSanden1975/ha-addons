# Wyoming YAMNet — blafverificatie voor het keukendisplay

Lokale HA OS add-on. Doet zich voor als Wyoming STT-server, maar draait Google
YAMNet (521 AudioSet-klassen) op het audiofragment dat het keukendisplay via de
Assist-pipeline instuurt. Antwoord komt als 'transcript' terug:

    bark 0.87 (Bark 0.87)
    none 0.04 (Speech 0.81)

Het display (`voice_assistant.on_stt_end`) leest de eerste twee tokens en beslist
zelf over de melding (number **Blaf score minimum**).

## Installatie (HA OS, Pi 5)

1. Kopieer de map `wyoming-yamnet/` naar `/addons/wyoming-yamnet/` op de Pi
   (via de Samba- of SSH-add-on; `/addons` staat naast `/config`).
2. Instellingen → Add-ons → Add-on store → ⋮ rechtsboven → **Controleren op updates**.
   Onder *Lokale add-ons* verschijnt **Wyoming YAMNet** → Installeren (bouwt het
   image lokaal, ~2–3 min).
3. Opties laten staan, **Starten**. Bij de eerste start haalt de add-on het model
   (~4 MB) en de klassenlijst op naar `/data`. Log moet eindigen met
   `Wyoming-YAMNet luistert op tcp://0.0.0.0:10310`.
4. Instellingen → Apparaten & diensten → Integratie toevoegen → **Wyoming Protocol**
   → host `localhost` (of het LAN-IP van de Pi), poort `10310`.
5. Instellingen → Spraakassistenten → **Assistent toevoegen**:
   - Naam: `Blafdetectie`
   - Taal: Nederlands
   - Conversatie-agent: Home Assistant
   - Spraak-naar-tekst: `yamnet`
   - Tekst-naar-spraak: geen
   - Wake word: geen
6. Instellingen → Apparaten → **Keukendisplay** → Configureren (tandwiel bij
   *Voice assistant*) → pipeline **Blafdetectie** kiezen.

## Opties

| Optie | Standaard | Betekenis |
|---|---|---|
| `bewaar_fragmenten` | `true` | Elk beoordeeld fragment als WAV opslaan |
| `bewaar_map` | `/share/blafdetectie` | Map voor de WAV's (naam: `datum_tijd_bark|none_score.wav`) |
| `bewaar_alleen_blaf` | `false` | Alleen fragmenten met oordeel `bark` bewaren |
| `threads` | `2` | TFLite-threads |
| `log_level` | `info` | `debug` logt ook de verwerkingstijd per fragment |

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
