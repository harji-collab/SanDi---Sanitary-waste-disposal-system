"""Live hardware loop: read the ESP32 stream, classify, and measure power.

  python logger.py --port /dev/ttyUSB0 --mode baseline  --seconds 300
  python logger.py --port /dev/ttyUSB0 --mode optimized --seconds 300
  python logger.py --summarise   # writes ../results/ina219.json

Run the model on a Raspberry Pi / laptop with the INA219 on the *compute*
board's supply rail (or use a USB power meter and log by hand) so the
before/after power draw is real, then:
  python pipeline.py --ina219 ../results/ina219.json
Requires: pip install pyserial onnxruntime numpy
"""
import argparse, csv, json, os, time
from collections import deque
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")


def run(a):
    import serial, onnxruntime as ort
    from data import features, compact_features, CLASSES
    from pipeline import quiet_one, held_one
    model = "baseline_fp32.onnx" if a.mode == "baseline" else "optimized_int8.onnx"
    sess = ort.InferenceSession(os.path.join(RES, model), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    ser = serial.Serial(a.port, 115200, timeout=1)
    buf = deque(maxlen=32)
    out = open(os.path.join(RES, f"log_{a.mode}.csv"), "w", newline="")
    wr = csv.writer(out); wr.writerow(["t", "state", "infer_us", "bus_mv", "current_ma"])
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        line = ser.readline().decode(errors="ignore").strip().split(",")
        if len(line) != 6 or not line[0].isdigit():
            continue
        _, ir, tc, wg, mv, ma = map(float, line)
        buf.append((ir, tc, wg))
        if len(buf) < 32 or int(_) % 1000 > 100:   # classify once per second
            continue
        x = np.array(buf, np.float32)[None]
        t0 = time.perf_counter()
        if a.mode == "baseline":
            s = int(sess.run(None, {name: features(x)})[0][0])
        elif quiet_one(x[0]):
            s = held_one(x[0])
        else:
            s = int(sess.run(None, {name: compact_features(x)})[0][0])
        dt = (time.perf_counter() - t0) * 1e6
        wr.writerow([time.time(), CLASSES[s], f"{dt:.1f}", mv, ma])
        print(f"{CLASSES[s]:14s} {dt:7.1f} us  {mv*ma/1e3:6.0f} mW")


def summarise():
    """Energy/inference = mean power x mean inference time, per mode."""
    import pandas as pd
    r = {}
    for m in ("baseline", "optimized"):
        d = pd.read_csv(os.path.join(RES, f"log_{m}.csv"))
        p_w = (d.bus_mv * d.current_ma).mean() / 1e6
        r[f"{m}_j"] = p_w * d.infer_us.mean() / 1e6
        r[f"{m}_mw"] = p_w * 1e3
    json.dump(r, open(os.path.join(RES, "ina219.json"), "w"), indent=2)
    print(r)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port"); p.add_argument("--mode", choices=["baseline", "optimized"])
    p.add_argument("--seconds", type=int, default=300)
    p.add_argument("--summarise", action="store_true")
    a = p.parse_args()
    summarise() if a.summarise else run(a)
