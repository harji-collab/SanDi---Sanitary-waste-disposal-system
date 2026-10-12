# Team SANDI — Green AI Smart Sanitary Disposal Unit

Problem: take an AI model + workload, cut energy per inference by ≥50% with
<2% accuracy loss; measure power, latency, cost and carbon before/after.

Our workload: a sanitary disposal bin with an **IR presence sensor**, a
**DS18B20 temperature sensor** and an **HX711 load cell**. Every second the
model classifies the last 3.2 s window into
`IDLE · DISPOSAL · NEAR_FULL · FULL · THERMAL_ALERT`.

## Result (this repo, synthetic workload, 20,000 test windows)

| | Baseline FP32 | Optimised | Change |
|---|---|---|---|
| Energy / inference | ~2,860 µJ | ~1,100 µJ | **−61.5%** |
| Latency | ~191 µs | ~73 µs | −62% |
| Accuracy | 99.30% | 99.71% | +0.41 pts |
| Model size | 851 KB | 3.7 KB | −99.6% |
| MACs | 216,704 | 840 | −99.6% |

Numbers vary a few % run to run. Energy here = measured time × 15 W core
(no RAPL on the build machine). Replace with INA219 readings on hardware.

## How the energy cut is achieved
1. **Knowledge distillation** — 102-512-256-128-5 teacher → 30-24-5 student
   trained on teacher labels over 60k extra unlabelled windows.
2. **Cheaper features** — 8-step downsampled channels + 6 stats, no sorting.
3. **INT8 quantisation** — ONNX Runtime dynamic quantisation.
4. **Event-driven gating** — ~40% of windows (no IR, weight steady within
   40 g, temperature steady) skip the model and hold the bin-level state.

## Wiring (ESP32 DevKit)
| Part | Pin |
|---|---|
| IR obstacle sensor OUT (E18-D80NK / FC-51) | GPIO27 (or Sharp analog → GPIO34) |
| DS18B20 data (+4.7 kΩ pull-up to 3V3) | GPIO4 |
| HX711 DOUT / SCK | GPIO16 / GPIO17 |
| INA219 SDA / SCL (power measurement) | GPIO21 / GPIO22 |

Flash `firmware/sandi_esp32.ino` (libs: OneWire, DallasTemperature, HX711,
Adafruit_INA219). Calibrate `HX_CAL_FACTOR` with a known weight.

## Run
```bash
pip install numpy pandas scikit-learn onnx skl2onnx onnxruntime pyserial
cd src
python pipeline.py            # train, optimise, benchmark -> results/results.json
python dashboard.py           # -> results/dashboard.html
# with hardware (INA219 on the compute board's supply):
python logger.py --port /dev/ttyUSB0 --mode baseline  --seconds 300
python logger.py --port /dev/ttyUSB0 --mode optimized --seconds 300
python logger.py --summarise
python pipeline.py --ina219 ../results/ina219.json && python dashboard.py
```
Tunables: `--core-watts`, `--units`, `--inf-per-sec`, `--inr-per-kwh`,
`--grid-kgco2-per-kwh` (default 0.71, India grid; check the latest CEA figure).

## Before the final demo
- Log real bin data with `logger.py`, retrain on it — synthetic data makes
  the gating rule look better than it will in the field.
- Measure with INA219 / USB power meter so the energy claim is measured, not
  estimated. On an ESP32 or RPi the baseline's 217k MACs cost far more
  relative to the 840-MAC student than on a laptop CPU, so the gap widens.

## Live console

The standalone console is `src/live_console.html`. It includes a simulated demo and a browser-based Web Serial connection that reads the CSV stream from `firmware/sandi_esp32.ino` (115200 baud). The embedded student model and benchmark values are a snapshot; rerun the pipeline to refresh benchmark outputs, and update the console snapshot when model weights change.

Serve the repository from localhost, then open the console in Chrome or Edge:

```bash
python -m http.server 8000
# open http://localhost:8000/src/live_console.html
```

Click **Connect device**, choose the ESP32 serial port, and allow access. Use **Run demo** to view simulated readings without hardware. Install the firmware libraries listed above and connect the sensors according to the wiring table before using live readings.
