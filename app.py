# app.py — Streamlit live demo for Betaworks “Agent System” pitch
# Run: streamlit run app.py
# Deploy: push to GitHub → Streamlit Community Cloud → share the public app link

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Neuro Risk Agent Demo", layout="wide")

# -----------------------------
# Demo configuration
# -----------------------------
@dataclass
class DemoConfig:
    dt_seconds: float = 0.5
    horizon_minutes: int = 30
    detect_threshold: float = 0.85
    forecast_threshold: float = 0.65
    confidence_threshold: float = 0.70
    training_labels_needed: int = 6
    refractory_minutes: int = 30

MODE_DETECTION = "Detection-only"
MODE_FORECAST = "Forecasting"
MODE_FALLBACK = "Fallback (low confidence)"

MODEL_FAMILIES = [
    "RandomForest (features)",
    "LSTM (temporal)",
    "Transformer (spatiotemporal)",
    "GraphModel (spatial)",
]

AUX_SIGNALS = [
    "HR", "SpO2", "EDA", "SkinTemp", "Motion", "SleepState"
]

# -----------------------------
# Helpers
# -----------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (np.sum(e) + 1e-9)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def now_ts(sim_t: float) -> str:
    # show minutes:seconds
    m = int(sim_t // 60)
    s = int(sim_t % 60)
    return f"{m:02d}:{s:02d}"

def scenario_params(name: str) -> Dict:
    # Tuned to demonstrate behavior; not medical advice or real clinical performance
    if name == "Stable day":
        return dict(base_risk=0.15, drift=0.0003, artifact_rate=0.05, event_prob=0.01, preictal_bump=0.15)
    if name == "Artifact-heavy dry electrodes":
        return dict(base_risk=0.20, drift=0.0005, artifact_rate=0.30, event_prob=0.01, preictal_bump=0.12)
    if name == "Pre-ictal ramp (forecast success)":
        return dict(base_risk=0.20, drift=0.0012, artifact_rate=0.08, event_prob=0.015, preictal_bump=0.35)
    if name == "False alarm recovery (self-correct)":
        return dict(base_risk=0.25, drift=0.0007, artifact_rate=0.10, event_prob=0.02, preictal_bump=0.20)
    if name == "Medication change feedback loop":
        return dict(base_risk=0.35, drift=0.0005, artifact_rate=0.08, event_prob=0.02, preictal_bump=0.25, med_effect=True)
    return dict(base_risk=0.20, drift=0.0005, artifact_rate=0.08, event_prob=0.01, preictal_bump=0.20)

def init_state():
    if "state" in st.session_state:
        return
    st.session_state.state = {
        "cfg": DemoConfig(),
        "scenario": "Pre-ictal ramp (forecast success)",
        "sim_running": False,
        "sim_time": 0.0,
        "mode": MODE_DETECTION,
        "confidence": 0.45,
        "labels_count": 0,
        "last_alert_time": -1e9,
        "caregiver_notified": False,
        "clinician_notified": False,
        "model_weights": {m: 1.0/len(MODEL_FAMILIES) for m in MODEL_FAMILIES},
        "model_perf": {m: {"tp":0,"fp":0,"fn":0} for m in MODEL_FAMILIES},
        "history": [],
        "events": [],
        "medication_state": {"dose_time_shift_min": 0, "dose_amount_delta": 0, "active": False},
    }

def simulate_signals(sim_t: float, params: Dict) -> Tuple[Dict, float, float]:
    """
    Produce synthetic EEG+aux context and a latent 'true risk' driver.
    Returns:
      aux_dict, channel_quality(0-1), latent_true_risk(0-1)
    """
    # Channel quality degrades with artifacts (dry electrode demo)
    artifact = 1.0 if random.random() < params["artifact_rate"] else 0.0
    channel_quality = clamp01(1.0 - 0.6*artifact - 0.15*random.random())

    # Latent true risk slowly drifts; optional medication effect reduces risk after a while
    base = params["base_risk"] + params["drift"] * sim_t
    base += params["preictal_bump"] * (1/(1+np.exp(-(sim_t-180)/30)))  # ramp around 3 minutes in
    if params.get("med_effect"):
        # after ~4 min, show improvement from "med timing adjustment"
        base -= 0.18 * (1/(1+np.exp(-(sim_t-240)/20)))

    true_risk = clamp01(base + 0.05*np.sin(sim_t/30) + 0.03*(random.random()-0.5))

    # Aux signals correlate weakly with risk/artifacts for demo
    aux = {
        "HR": 60 + 35*true_risk + 8*(random.random()-0.5),
        "SpO2": 98 - 2.5*true_risk - 1.5*artifact + 0.5*(random.random()-0.5),
        "EDA": 0.3 + 0.9*true_risk + 0.3*artifact + 0.1*(random.random()-0.5),
        "SkinTemp": 33.5 + 0.8*true_risk + 0.2*(random.random()-0.5),
        "Motion": 0.2 + 1.2*artifact + 0.4*(random.random()-0.5),
        "SleepState": 1 if (sim_t % 240) < 80 else 0,  # toggles for demo
    }

    return aux, channel_quality, true_risk

def model_predictions(true_risk: float, channel_quality: float, scenario: str) -> Dict[str, float]:
    """
    Each model family outputs a risk estimate.
    We deliberately make them behave differently under artifacts / contexts to show adaptive weighting.
    """
    # Sensitivity to artifacts differs by model family (demo-only)
    artifact_penalty = (1.0 - channel_quality)
    noise = lambda s: (random.random()-0.5)*s

    preds = {}
    preds["RandomForest (features)"] = clamp01(true_risk + noise(0.10) - 0.30*artifact_penalty)
    preds["LSTM (temporal)"] = clamp01(true_risk + noise(0.08) - 0.15*artifact_penalty)
    preds["Transformer (spatiotemporal)"] = clamp01(true_risk + noise(0.07) - 0.10*artifact_penalty)
    preds["GraphModel (spatial)"] = clamp01(true_risk + noise(0.09) - 0.25*artifact_penalty)

    # Scenario: gaming / low-channel could prefer temporal; simulate by shifting one model
    if "Stable day" in scenario:
        preds["RandomForest (features)"] = clamp01(preds["RandomForest (features)"] + 0.05)
    if "Artifact-heavy" in scenario:
        preds["Transformer (spatiotemporal)"] = clamp01(preds["Transformer (spatiotemporal)"] + 0.10)

    return preds

def update_weights(weights: Dict[str,float], preds: Dict[str,float], outcome: str) -> Dict[str,float]:
    """
    Simple online reweighting using multiplicative updates based on outcome.
    outcome: 'tp', 'fp', 'none'
    """
    # Penalize models that scream risk when no event (fp); reward when event happens (tp)
    new = dict(weights)
    for m, p in preds.items():
        if outcome == "tp":
            new[m] *= (1.0 + 0.15*(p - 0.5))
        elif outcome == "fp":
            new[m] *= (1.0 - 0.20*(p - 0.5))
        else:
            new[m] *= 1.0
    # normalize
    w = np.array(list(new.values()), dtype=float) + 1e-9
    w = w / w.sum()
    return {k: float(v) for k, v in zip(new.keys(), w)}

def fused_risk(preds: Dict[str,float], weights: Dict[str,float]) -> float:
    return float(sum(preds[m]*weights[m] for m in preds.keys()))

def should_notify(risk: float, confidence: float, mode: str, cfg: DemoConfig) -> Tuple[bool,bool]:
    """
    Returns: (notify_caregiver, notify_clinic)
    """
    if mode == MODE_DETECTION:
        # detection-only: only alert at very high risk (simulates detected ongoing event)
        return (risk >= cfg.detect_threshold and confidence >= 0.55), False
    if mode == MODE_FORECAST:
        caregiver = (risk >= cfg.forecast_threshold and confidence >= cfg.confidence_threshold)
        clinic = (risk >= 0.80 and confidence >= cfg.confidence_threshold)
        return caregiver, clinic
    # fallback: conservative
    caregiver = (risk >= 0.90 and confidence >= cfg.confidence_threshold)
    return caregiver, False

# -----------------------------
# UI
# -----------------------------
init_state()
S = st.session_state.state
cfg = S["cfg"]

st.title("Neuro Risk Agent System — Live Demo (Streamlit)")
st.caption("Interactive agent-system demo: continuous monitoring, memory, adaptive ensemble, mode transitions, and escalation. (Demo simulator)")

with st.sidebar:
    st.header("Demo Controls")
    S["scenario"] = st.selectbox("Scenario", [
        "Stable day",
        "Artifact-heavy dry electrodes",
        "Pre-ictal ramp (forecast success)",
        "False alarm recovery (self-correct)",
        "Medication change feedback loop",
    ], index=["Stable day","Artifact-heavy dry electrodes","Pre-ictal ramp (forecast success)",
              "False alarm recovery (self-correct)","Medication change feedback loop"].index(S["scenario"]))

    st.subheader("Agent Settings")
    cfg.forecast_threshold = st.slider("Forecast alert threshold", 0.40, 0.95, cfg.forecast_threshold, 0.01)
    cfg.confidence_threshold = st.slider("Confidence threshold", 0.40, 0.95, cfg.confidence_threshold, 0.01)
    cfg.training_labels_needed = st.slider("Labels needed to enable forecasting", 1, 20, cfg.training_labels_needed, 1)

    st.subheader("Simulation")
    col_a, col_b = st.columns(2)
    if col_a.button("Start"):
        S["sim_running"] = True
    if col_b.button("Stop"):
        S["sim_running"] = False

    if st.button("Reset"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.divider()
    st.subheader("Manual Events (Human-in-the-loop)")
    if st.button("Mark Aura / Seizure (adds label)"):
        S["labels_count"] += 1
        S["events"].append({"t": S["sim_time"], "type": "UserLabel", "note": "Manual event label (aura/seizure)"})

    st.caption("Tip: click this a few times early to unlock forecasting mode and show adaptive learning.")

# Layout panels
left, mid, right = st.columns([1.1, 1.2, 1.2])

with left:
    st.subheader("1) Wearable EEG Agent")
    st.metric("Mode", S["mode"])
    st.metric("Confidence", f"{S['confidence']:.2f}")
    st.metric("Labels collected", S["labels_count"])

    st.markdown("**Channel quality & sensor status**")
    cq_placeholder = st.empty()
    aux_placeholder = st.empty()

with mid:
    st.subheader("2) Coordination Agent (Phone/App)")
    fused_placeholder = st.empty()
    alert_placeholder = st.empty()
    weights_placeholder = st.empty()

with right:
    st.subheader("3) Clinical Interface (Clinic/Doctor)")
    clinic_placeholder = st.empty()
    log_placeholder = st.empty()

# Simulation step button + autoplay
step_col1, step_col2 = st.columns([0.2, 0.8])
manual_step = step_col1.button("Step ▶")

# -----------------------------
# Simulation loop (single step per rerun)
# -----------------------------
def do_step():
    params = scenario_params(S["scenario"])
    S["sim_time"] += cfg.dt_seconds

    aux, channel_quality, true_risk = simulate_signals(S["sim_time"], params)

    # "confidence" depends on channel quality + training maturity
    maturity = clamp01(S["labels_count"] / max(1, cfg.training_labels_needed))
    S["confidence"] = clamp01(0.35 + 0.45*maturity + 0.25*channel_quality)

    # operational mode transitions
    if S["mode"] == MODE_DETECTION and S["labels_count"] >= cfg.training_labels_needed and S["confidence"] >= cfg.confidence_threshold:
        S["mode"] = MODE_FORECAST
        S["events"].append({"t": S["sim_time"], "type": "ModeTransition", "note": "Detection-only → Forecasting enabled"})
    if S["mode"] == MODE_FORECAST and S["confidence"] < 0.55:
        S["mode"] = MODE_FALLBACK
        S["events"].append({"t": S["sim_time"], "type": "ModeTransition", "note": "Forecasting → Fallback (low confidence)"})
    if S["mode"] == MODE_FALLBACK and S["confidence"] >= cfg.confidence_threshold:
        S["mode"] = MODE_FORECAST
        S["events"].append({"t": S["sim_time"], "type": "ModeTransition", "note": "Fallback → Forecasting restored"})

    preds = model_predictions(true_risk=true_risk, channel_quality=channel_quality, scenario=S["scenario"])
    fused = fused_risk(preds, S["model_weights"])

    # Outcome simulation: in ramp scenarios, occasionally create a "true seizure" event
    # This is only to illustrate online performance-driven reweighting.
    seizure_happened = (random.random() < params["event_prob"]) and (true_risk > 0.75)
    if seizure_happened:
        S["events"].append({"t": S["sim_time"], "type": "ObservedEvent", "note": "Observed seizure (simulated ground truth)"})

    # Determine alerts
    caregiver, clinic = should_notify(fused, S["confidence"], S["mode"], cfg)

    # Refractory behavior (30 min) in demo time: compress minutes so it triggers
    # Here we treat dt_seconds as real seconds; refractory uses minutes.
    refractory_sec = cfg.refractory_minutes * 60
    in_refractory = (S["sim_time"] - S["last_alert_time"]) < refractory_sec

    caregiver = caregiver and (not in_refractory)
    clinic = clinic and (not in_refractory)

    if caregiver:
        S["caregiver_notified"] = True
        S["last_alert_time"] = S["sim_time"]
        S["events"].append({"t": S["sim_time"], "type": "Escalation", "note": "Caregiver notified (push/SMS/call)"})

    if clinic:
        S["clinician_notified"] = True
        S["last_alert_time"] = S["sim_time"]
        S["events"].append({"t": S["sim_time"], "type": "Escalation", "note": "Clinic notified (secure channel)"})

    # Weight updates based on outcomes (self-evaluation)
    # If seizure happened and we were high-risk -> reward; if we alerted but no seizure -> punish
    outcome = "none"
    if seizure_happened and fused >= cfg.forecast_threshold:
        outcome = "tp"
    elif caregiver and (not seizure_happened):
        outcome = "fp"

    if outcome in ("tp", "fp"):
        S["model_weights"] = update_weights(S["model_weights"], preds, outcome)
        S["events"].append({"t": S["sim_time"], "type": "SelfEval", "note": f"Weight update based on outcome={outcome.upper()}"})

    # Store history
    row = {
        "t_sec": S["sim_time"],
        "t": now_ts(S["sim_time"]),
        "mode": S["mode"],
        "confidence": S["confidence"],
        "channel_quality": channel_quality,
        "true_risk": true_risk,
        "fused_risk": fused,
        **{f"pred_{m}": preds[m] for m in preds},
        **{f"w_{m}": S["model_weights"][m] for m in S["model_weights"]},
        **{f"aux_{k}": aux[k] for k in aux},
    }
    S["history"].append(row)

    # Simple "clinical recommendations" for demo
    if "Medication change" in S["scenario"]:
        # When risk stays high, suggest timing shift; when risk drops, record improvement
        if fused > 0.75 and S["mode"] == MODE_FORECAST:
            S["medication_state"]["active"] = True
            S["medication_state"]["dose_time_shift_min"] = 30
            S["events"].append({"t": S["sim_time"], "type": "Clinical", "note": "Recommendation: shift dose timing by +30 min"})
        if fused < 0.55 and S["medication_state"]["active"]:
            S["events"].append({"t": S["sim_time"], "type": "Clinical", "note": "Outcome: risk decreased after timing shift (simulated)"})
            S["medication_state"]["active"] = False

    # Update UI placeholders
    cq_placeholder.progress(float(channel_quality))
    aux_df = pd.DataFrame([{k: aux[k] for k in AUX_SIGNALS}])
    aux_placeholder.dataframe(aux_df, use_container_width=True)

    fused_placeholder.metric("Fused Risk Score", f"{fused:.2f}", delta=f"{(fused-0.50):+.2f}")
    alert_txt = []
    if caregiver:
        alert_txt.append("✅ Caregiver alert triggered")
    if clinic:
        alert_txt.append("✅ Clinic alert triggered")
    if not alert_txt:
        alert_txt.append("No escalation (agent running silently)")
    alert_placeholder.info(" | ".join(alert_txt))

    weights_df = pd.DataFrame({
        "Model": list(S["model_weights"].keys()),
        "Weight": [S["model_weights"][m] for m in S["model_weights"]],
        "Latest_pred": [preds[m] for m in preds],
    }).sort_values("Weight", ascending=False)
    weights_placeholder.dataframe(weights_df, use_container_width=True, hide_index=True)

    # Clinic panel
    recs = []
    if fused >= 0.80 and S["mode"] == MODE_FORECAST and S["confidence"] >= cfg.confidence_threshold:
        recs.append("High risk: consider rescue protocol readiness.")
    if fused >= cfg.forecast_threshold and S["mode"] == MODE_FORECAST:
        recs.append(f"Forecast window: next {cfg.horizon_minutes} min elevated.")
    if S["mode"] != MODE_FORECAST:
        recs.append("System not in forecasting mode (insufficient confidence/labels).")
    if "Medication change" in S["scenario"]:
        if S["medication_state"]["active"]:
            recs.append("Suggested: adjust medication timing (demo).")
        else:
            recs.append("Monitoring medication response (demo).")

    clinic_placeholder.write("**Clinical summary (demo):**")
    clinic_placeholder.write("\n- " + "\n- ".join(recs))

    # Event log
    ev = pd.DataFrame(S["events"][-12:])  # last 12
    if not ev.empty:
        ev = ev.assign(time=lambda d: d["t"].apply(now_ts))
        ev = ev[["time", "type", "note"]]
    log_placeholder.write("**Agent activity log (recent):**")
    log_placeholder.dataframe(ev, use_container_width=True, hide_index=True)

# Decide whether to step
if manual_step:
    do_step()

# Autoplay: on each rerun, do one step and sleep briefly
if S["sim_running"]:
    do_step()
    time.sleep(cfg.dt_seconds)
    st.rerun()

# Show charts below
st.divider()
st.subheader("Telemetry (Risk, Confidence, and Weights)")

if S["history"]:
    df = pd.DataFrame(S["history"])
    df_tail = df.tail(400)

    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(df_tail.set_index("t_sec")[["fused_risk", "true_risk", "confidence"]])
    with c2:
        weight_cols = [c for c in df_tail.columns if c.startswith("w_")]
        if weight_cols:
            st.line_chart(df_tail.set_index("t_sec")[weight_cols])

st.caption("This demo uses simulated data to illustrate autonomous agent behavior: perception, memory, self-evaluation, mode transitions, and escalation.")
