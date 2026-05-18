# 🐟 Fish S2 Pro — High-Performance Local WebUI

An elegant, premium local WebUI for running high-fidelity **Fish Speech S2 Pro** voice synthesis on Apple Silicon Macs. Engineered for maximum speed, VRAM efficiency, and flawless zero-shot voice cloning.

![Screen mockup](https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=1200&q=80)

---

## ✨ Features

- **⚡ Apple Silicon MPS Optimization:** Seamlessly harnesses Mac hardware using MLX. Generation is down to **3–5 seconds**!
- **🎙 Intelligent Sentence-Chunking:** Long texts are automatically split by punctuation boundaries. Synthesized sequentially in a high-speed background thread and glued back together seamlessly. No OOM spikes, memory warnings, or GPU lags.
- **🎧 Side-by-Side Play Preview:** Preview your library or auto-imported presets directly in the text-to-speech dropdown on the generate card without changing tabs.
- **🏷 Custom Tone & Speaker Design:** Describe your custom speaker directly in the WebUI (e.g. *Male, deep, age 35, professional broadcast voice, breathy*) to inject styles dynamically.
- **💾 One-Click Save Voice:** Convert any generated zero-shot voice directly into a permanent reusable library preset.
- **📥 Asynchronous Background Transcribing:** Automatically copies loose audio from Downloads presets, assigns hex-shortened titles (e.g., `Preset 08F89F`), and runs native Whisper-Tiny to transcribe references in the background without UI hangs.
- **🛑 Sidebar Server Lifecycle Control:** Shut down the FastAPI server, unload all models, and release all unified memory cleanly with a single click in the sidebar.

---

## 🛠 Setup

### Prerequisites
* macOS 14+ (M1 / M2 / M3 / M4 Apple Silicon chip)
* python 3.12+ / 3.13

### One-Click Launch
Double-click `launch.command` to start, or run from the shell:
```bash
./launch.command
```

---

## 📂 Architecture

- **`app.py`**: Fast FastAPI backend serving optimized sentence-chunked speech synthesis, Whisper transcribing, and library endpoints.
- **`index.html`**: A premium Glassmorphic, HSL-tailored single-page WebUI for seamless generation.
- **`voices/`**: Preserved library storage where individual voices and metadata are saved.
- **`outputs/`**: Temporary generation directories.
