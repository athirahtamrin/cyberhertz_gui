from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn

N = 512
CHECKPOINT = Path(__file__).with_name("cnn_aug_seed42_best.zip")
EXPECTED_CLASSES = [
    "BPSK", "QPSK", "16QAM", "64QAM", "FHSS", "LFM Radar",
    "Barrage Jamming", "Spoofing Jamming",
]


class CNNBaseline(nn.Module):
    """Architecture copied from cell 71 of the supplied training notebook."""

    def __init__(self, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model():
    # Only load the checkpoint you trained and supplied. PyTorch pickle files
    # must never be loaded from an untrusted user upload.
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    classes = [str(name) for name in checkpoint["class_names"]]
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected class order in checkpoint: {classes}")
    model = CNNBaseline(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, classes


def generate_example(kind, snr_db):
    rng = np.random.default_rng()
    t = np.arange(N)
    if kind == "FHSS":
        frequencies = rng.choice([0.03, 0.08, 0.14, 0.21], size=8)
        phase = np.cumsum(2 * np.pi * np.repeat(frequencies, N // 8))
        clean = np.exp(1j * phase)
    elif kind == "LFM Radar":
        clean = np.exp(1j * 2 * np.pi * (0.02 * t + 0.00035 * t**2))
    else:
        symbols = rng.choice([-1, 1], size=64)
        clean = np.repeat(symbols, 8).astype(complex)
    signal_power = np.mean(np.abs(clean) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.normal(size=N) + 1j * rng.normal(size=N)
    )
    return clean + noise


def read_csv(upload):
    df = pd.read_csv(upload)
    if not {"I", "Q"}.issubset(df.columns):
        raise ValueError("The CSV needs columns named I and Q.")
    if len(df) != N:
        raise ValueError(f"The CSV needs exactly {N} rows.")
    iq = df["I"].to_numpy(dtype=float) + 1j * df["Q"].to_numpy(dtype=float)
    if not np.all(np.isfinite(iq)):
        raise ValueError("I and Q must contain finite numbers.")
    return iq


st.set_page_config(page_title="CyberHertz RF Detector", layout="wide")
st.title("CyberHertz | RF Signal Detection")
st.caption("512 complex IQ samples • CNN baseline with noise augmentation")

source = st.radio("Signal source", ["Generate example", "Upload CSV"], horizontal=True)
iq = None
if source == "Generate example":
    kind = st.selectbox("Example signal", ["FHSS", "LFM Radar", "BPSK"])
    snr_db = st.slider("SNR (dB)", -18.0, 18.0, 5.0, 0.5)
    if st.button("Generate signal") or "example_iq" not in st.session_state:
        st.session_state.example_iq = generate_example(kind, snr_db)
        st.session_state.example_snr = snr_db
    iq = st.session_state.example_iq
    snr_display = f"{st.session_state.example_snr:.1f} dB (selected)"
else:
    upload = st.file_uploader("Upload CSV containing I and Q columns", type="csv")
    snr_display = "Unknown (not provided)"
    if upload is not None:
        try:
            iq = read_csv(upload)
        except (ValueError, TypeError) as exc:
            st.error(str(exc))

if iq is not None:
    left, right = st.columns(2)
    with left:
        st.subheader("Time-domain waveform")
        fig, ax = plt.subplots()
        ax.plot(iq.real, label="I", linewidth=1)
        ax.plot(iq.imag, label="Q", linewidth=1)
        ax.set(xlabel="Sample index", ylabel="Amplitude")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("FFT spectrum")
        spectrum = np.fft.fftshift(np.fft.fft(iq))
        frequency = np.fft.fftshift(np.fft.fftfreq(N))
        magnitude_db = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-12))
        fig, ax = plt.subplots()
        ax.plot(frequency, magnitude_db, linewidth=1)
        ax.set(xlabel="Normalized frequency", ylabel="Magnitude (dB)")
        st.pyplot(fig)
        plt.close(fig)

    if not CHECKPOINT.is_file():
        st.warning("Add cnn_aug_seed42_best.zip to the same GitHub folder as this app.py.")
    else:
        try:
            model, classes = load_model()
            # Training notebook used float32 arrays with shape [batch, 2, 512].
            x = np.stack((iq.real, iq.imag)).astype(np.float32)
            with torch.inference_mode():
                logits = model(torch.from_numpy(x).unsqueeze(0))
                probabilities = torch.softmax(logits, dim=1)[0].numpy()
            class_id = int(np.argmax(probabilities))
            predicted_class = classes[class_id]
            confidence = float(probabilities[class_id])

            a, b, c = st.columns(3)
            a.metric("Predicted class", predicted_class)
            b.metric("Confidence", f"{confidence:.1%}")
            c.metric("SNR / noise condition", snr_display)

            st.subheader("Detection status")
            if predicted_class in ("Barrage Jamming", "Spoofing Jamming"):
                st.error("Jamming detected")
            elif predicted_class in ("FHSS", "LFM Radar"):
                st.warning("Military/CEMA signal detected")
            else:
                st.success("Standard communication signal detected")
            st.caption("Confidence is the model's softmax score, not a calibrated probability. "
