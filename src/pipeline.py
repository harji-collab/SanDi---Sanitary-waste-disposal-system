"""SANDI energy optimiser: baseline -> optimised, measured before/after.

Steps
  1. Train a deliberately heavy FP32 baseline (102-512-256-128-5 MLP).
  2. Optimise:
       a. knowledge distillation into a tiny student (102-24-5)
       b. INT8 dynamic quantisation (ONNX Runtime)
       c. event-driven gating: skip the model when IR, weight and temperature
          are all quiet, and hold the bin-level state instead
  3. Benchmark both on the same realistic workload: accuracy, latency,
     energy/inference, cost, carbon, and project to fleet scale.

Energy source, in priority order:
  - --ina219 CSV from the ESP32/RPi power logger (true measured power)
  - Intel RAPL (/sys/class/powercap) if the machine exposes it
  - time x active-core power (--core-watts, default 15 W) — an estimate
"""
import argparse, json, os, time, glob
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from sklearn.neural_network import MLPClassifier
from skl2onnx import to_onnx
from data import make, features, compact_features, CLASSES, CAP_G

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- energy
def rapl_path():
    p = glob.glob("/sys/class/powercap/intel-rapl:0/energy_uj")
    return p[0] if p and os.access(p[0], os.R_OK) else None


def measure(fn, n, core_watts):
    """Returns (seconds per call, joules per call, method)."""
    rp = rapl_path()
    for _ in range(min(200, n)):
        fn()  # warm-up
    e0 = int(open(rp).read()) if rp else None
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n
    if rp:
        return dt, (int(open(rp).read()) - e0) / 1e6 / n, "RAPL (measured)"
    return dt, dt * core_watts, f"time x {core_watts:g} W core (estimate)"


def session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = o.inter_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def macs(mlp):
    return int(sum(w.size for w in mlp.coefs_))


# ---------------------------------------------------------------- gating
def quiet(X):
    """Window has no presence, stable weight and stable temperature."""
    ir, tp, w = X[..., 0], X[..., 1], X[..., 2]
    return (ir.max(1) == 0) & (w.max(1) - w.min(1) < 40) & \
           (np.abs(tp[:, -4:].mean(1) - tp[:, :4].mean(1)) < 1.0)


def held_state(X):
    lvl = X[..., 2].mean(1) / CAP_G
    return np.where(lvl > 0.9, 3, np.where(lvl > 0.7, 2, 0))


def quiet_one(x):
    """Same rule as quiet() for one (32,3) window, early exit on presence."""
    if x[:, 0].any():
        return False
    w = x[:, 2]
    if w.max() - w.min() >= 40:
        return False
    t = x[:, 1]
    return abs(t[-4:].mean() - t[:4].mean()) < 1.0


def held_one(x):
    lvl = x[:, 2].mean() / CAP_G
    return 3 if lvl > 0.9 else 2 if lvl > 0.7 else 0


# ---------------------------------------------------------------- main
def main(a):
    Xtr, ytr = make(30000, seed=1)
    Xte, yte = make(20000, seed=2)      # realistic field mix
    Ftr, Fte = features(Xtr), features(Xte)

    print("training FP32 baseline ...")
    teacher = MLPClassifier((512, 256, 128), max_iter=60, early_stopping=True,
                            random_state=0).fit(Ftr, ytr)

    print("distilling student ...")
    Xu, _ = make(60000, seed=3, mix=[0.2] * 5)   # unlabelled, balanced
    Cu, Ctr, Cte = compact_features(Xu), compact_features(Xtr), compact_features(Xte)
    Fs = np.concatenate([Ctr, Cu]); ys = np.concatenate([ytr, teacher.predict(features(Xu))])
    student = MLPClassifier((24,), max_iter=200, early_stopping=True,
                            random_state=0).fit(Fs, ys)

    pb, ps, pq = (os.path.join(OUT, f) for f in
                  ("baseline_fp32.onnx", "student_fp32.onnx", "optimized_int8.onnx"))
    for m, p, ex in ((teacher, pb, Ftr[:1]), (student, ps, Ctr[:1])):
        with open(p, "wb") as f:
            f.write(to_onnx(m, ex, options={id(m): {"zipmap": False}}).SerializeToString())
    quantize_dynamic(ps, pq, weight_type=QuantType.QInt8)

    sb, sq = session(pb), session(pq)
    ib, iq = sb.get_inputs()[0].name, sq.get_inputs()[0].name

    def run(sess, name, F):
        return np.concatenate([sess.run(None, {name: F[i:i + 4096]})[0]
                               for i in range(0, len(F), 4096)])

    pred_b = run(sb, ib, Fte)
    q = quiet(Xte)
    pred_o = np.where(q, held_state(Xte), run(sq, iq, Cte))
    acc_b, acc_o = float((pred_b == yte).mean()), float((pred_o == yte).mean())
    acc_student_only = float((run(sq, iq, Cte) == yte).mean())

    # ---- per-inference cost on one window, batch 1, single core
    x1 = Xte[:1]
    xb = features(x1)
    fb = lambda: sb.run(None, {ib: features(x1)})
    def fo():   # active window: gate check fails -> compact features -> INT8 student
        quiet_one(x1[0]); return sq.run(None, {iq: compact_features(x1)})
    def fg():   # quiet window: gate passes -> hold level state, model skipped
        return quiet_one(x1q[0]) and held_one(x1q[0])
    x1q = Xte[q][:1]
    n = a.reps
    tb, eb, method = measure(fb, n, a.core_watts)
    to_, eo, _ = measure(fo, n, a.core_watts)
    tg, eg, _ = measure(fg, n, a.core_watts)
    gate_rate = float(q.mean())
    eo_eff = gate_rate * eg + (1 - gate_rate) * eo        # workload-weighted
    to_eff = gate_rate * tg + (1 - gate_rate) * to_

    if a.ina219:   # measured power override from the hardware logger
        d = json.load(open(a.ina219))
        eb, eo_eff, method = d["baseline_j"], d["optimized_j"], "INA219 (measured)"

    # ---- fleet projection
    per_year = a.units * a.inf_per_sec * 3600 * 24 * 365
    def scale(j):
        kwh = j * per_year / 3.6e6
        return {"kwh_year": kwh, "inr_year": kwh * a.inr_per_kwh,
                "kgco2_year": kwh * a.grid_kgco2_per_kwh}

    size = lambda p: os.path.getsize(p) / 1024
    R = {
        "classes": CLASSES,
        "energy_method": method,
        "assumptions": {"core_watts": a.core_watts, "units": a.units,
                        "inferences_per_sec_per_unit": a.inf_per_sec,
                        "inr_per_kwh": a.inr_per_kwh,
                        "grid_kgco2_per_kwh": a.grid_kgco2_per_kwh,
                        "test_windows": int(len(yte))},
        "baseline": {"accuracy": acc_b, "latency_us": tb * 1e6, "energy_uj": eb * 1e6,
                     "macs": macs(teacher), "size_kb": size(pb),
                     "power_w": eb / tb, **scale(eb)},
        "optimized": {"accuracy": acc_o, "student_only_accuracy": acc_student_only,
                      "latency_us": to_eff * 1e6, "energy_uj": eo_eff * 1e6,
                      "model_latency_us": to_ * 1e6, "gated_latency_us": tg * 1e6,
                      "gate_rate": gate_rate, "macs": macs(student),
                      "size_kb": size(pq), "power_w": eo_eff / to_eff, **scale(eo_eff)},
        "per_class_recall": {
            c: {"baseline": float((pred_b[yte == k] == k).mean()),
                "optimized": float((pred_o[yte == k] == k).mean())}
            for k, c in enumerate(CLASSES)},
    }
    R["energy_reduction_pct"] = 100 * (1 - eo_eff / eb)
    R["accuracy_loss_pts"] = 100 * (acc_b - acc_o)
    R["pass"] = R["energy_reduction_pct"] >= 50 and R["accuracy_loss_pts"] < 2
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=2)

    print(f"\nEnergy method : {method}")
    print(f"Accuracy      : {acc_b:.4f} -> {acc_o:.4f}  (loss {R['accuracy_loss_pts']:.2f} pts)")
    print(f"Latency       : {tb*1e6:.1f} us -> {to_eff*1e6:.1f} us")
    print(f"Energy/infer  : {eb*1e6:.1f} uJ -> {eo_eff*1e6:.1f} uJ  "
          f"(-{R['energy_reduction_pct']:.1f}%)")
    print(f"Model size    : {size(pb):.0f} KB -> {size(pq):.1f} KB ; MACs {macs(teacher)} -> {macs(student)}")
    print(f"Gate rate     : {gate_rate:.1%} windows skipped the model")
    print("TARGET MET" if R["pass"] else "TARGET NOT MET")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=3000)
    p.add_argument("--core-watts", type=float, default=15.0)
    p.add_argument("--units", type=int, default=10000)
    p.add_argument("--inf-per-sec", type=float, default=1.0)
    p.add_argument("--inr-per-kwh", type=float, default=8.0)
    p.add_argument("--grid-kgco2-per-kwh", type=float, default=0.71)
    p.add_argument("--ina219", help="JSON with baseline_j / optimized_j from hardware")
    main(p.parse_args())
