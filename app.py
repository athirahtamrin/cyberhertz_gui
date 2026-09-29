import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 512
CLASSES = [
    "BPSK", "QPSK", "16QAM", "64QAM",
    "FHSS", "LFM Radar", "Barrage Jamming", "Spoofing Jamming"
]

st.set_page_config(page_title="CyberHertz RF Detector", layout="wide")
st.title("CyberHertz | RF Signal Detection")
st.caption("Upload or generate 512 complex IQ samples")

def generate_signal(kind, snr_db):
    rng = np.random.default_rng()
    t = np.arange(N)

    if kind == "FHSS":
        frequencies = rng.choice([0.03, 0.08, 0.14, 0.21], size=8)
        phase = np.cumsum(2 * np.pi * np.repeat(frequencies, N // 8))
        clean = np.exp(1j * phase)
    elif kind == "LFM Radar":
        clean = np.exp(1j * 2 * np.pi * (0.02 * t + 0.00035 * t**2))
    elif kind == "BPSK":
        symbols = rng.choice([-1, 1], size=64)
        clean = np.repeat(symbols, 8).astype(complex)
    else:
        raise ValueError("Unknown example signal")

    signal_power = np.mean(np.abs(clean) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.normal(size=N) + 1j * rng.normal(size=N)
    )
    return clean + noise

def read_csv(upload):
    df = pd.read_csv(upload)
    if not {"I", "Q"}.issubset(df.columns):
        raise ValueError("CSV must contain columns named I and Q.")
    if len(df) != N:
        raise ValueError(f"CSV must contain exactly {N} rows.")
    iq = df["I"].to_numpy(dtype=float) + 1j * df["Q"].to_numpy(dtype=float)
    if not np.all(np.isfinite(iq)):
        raise ValueError("I and Q must contain only finite numbers.")
    return iq

def predict_with_your_model(iq):
    """
    Replace this function with the SAME preprocessing and model
    inference used in your training notebook.

    Expected output: (class_name, confidence_between_0_and_1)
    """
    return None, None

source = st.radio("Signal source", ["Generate example", "Upload CSV"], horizontal=True)

if source == "Generate example":
    kind = st.selectbox("Example signal", ["FHSS", "LFM Radar", "BPSK"])
    snr_db = st.slider("SNR (dB)", -18.0, 18.0, 5.0, 0.5)
    iq = generate_signal(kind, snr_db)
    snr_display = f"{snr_db:.1f} dB (selected)"
else:
    upload = st.file_uploader("Upload a CSV with I and Q columns", type="csv")
    iq = None
    snr_display = "Unknown (not supplied)"
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

    predicted_class, confidence = predict_with_your_model(iq)

    a, b, c = st.columns(3)
    a.metric("Predicted class", predicted_class or "Model not connected")
    b.metric("Confidence", f"{confidence:.1%}" if confidence is not None else "—")
    c.metric("SNR / noise condition", snr_display)

    st.subheader("Detection status")
    if predicted_class is None:
        st.info("Signal loaded. Connect your trained model to enable detection.")
    elif predicted_class in ["Barrage Jamming", "Spoofing Jamming"]:
        st.error("Jamming detected")
    elif predicted_class in ["FHSS", "LFM Radar"]:
        st.warning("Military/CEMA signal detected")
    else:
        st.success("Standard communication signal detected")













