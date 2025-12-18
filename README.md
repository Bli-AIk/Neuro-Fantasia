# Neuro-Fantasia

[![license](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-APACHE) <br>
<img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" />

> Current Status: 🚧 Early Development

**Neuro-Fantasia** — A genetic algorithm-based synthesizer timbre reverse engineering tool.

| English | Simplified Chinese          |
|---------|-----------------------------|
| English | [简体中文](./README_zh-hans.md) |

## Introduction

`Neuro-Fantasia` is an automated sound design tool that uses evolutionary algorithms to recreate target sounds.
It solves the problem of manual synthesizer patching, allowing users to simply provide a sample (or a set of samples)
and get a synthesizer preset that mimics the timbre.

With `Neuro-Fantasia`, you only need to provide a WAV file or a dataset of notes. The system will evolve a patch over
thousands of generations to match the spectral and temporal characteristics of your sound.
In the future, it may also support direct export to game engines like Bevy or VST plugins.

## Features

* **Genetic Evolution**: Uses Tournament Selection, Crossover, and Mutation to find the best parameters.
* **Advanced Subtractive Engine**:
    * **Dynamic Transient Layer**: Precisely sculpted noise envelope (Attack/Decay) to simulate realistic instrument attacks (e.g., breath, bow scratch).
    * **Exponential ADSR**: Curve-controlled envelopes for natural, non-linear decay and release, vital for brass and acoustic emulation.
    * **Enhanced Modulation**: LFOs now feature Delay and Fade-in parameters to simulate delayed vibrato and gradual timbre evolution.
    * **Non-linear Filter Chain**: Includes Pre-Filter Saturation to add warmth and harmonics before filtering, plus Post-Filter Drive.
    * **Dual LFOs**: Dedicated LFOs for Vibrato (Pitch) and Wah/PWM (Filter).
    * **FX Engine**: Built-in Stereo Chorus and Reverb to add depth and space to the sound.
    * **Sorted Wavetables**: Automatically sorts thousands of waveforms by brightness for smoother evolution.
* **Multi-Target Training**: Train on multiple pitches of the same instrument for higher accuracy.
* **Resume Capability**: Stop and resume training at any time.
* **Similarity Threshold**: Automatically stop when a desired similarity percentage is reached.

## How to Use

1. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/neuro-fantasia.git
   cd neuro-fantasia
   # Initialize wavetable submodule
   git submodule update --init --recursive
   ```

3. **Build**:
   ```bash
   cargo build --release
   ```

4. **Prepare Data**:

   **Option A: Single File**
   Have a `.wav` file ready (e.g., `target.wav`).

   **Option B: Dataset (Recommended)**
   Create a folder (e.g., `my_instrument/`) with your audio files and a `dataset.json`:
   ```json
   {
     "samples": [
       { "filename": "C4.wav", "note": 60.0 },
       { "filename": "G4.wav", "note": 67.0 }
     ]
   }
   ```

5. **Run Training**:

   **Basic Single File:**
   ```bash
   cargo run --release --bin trainer -- --target target.wav --note 60 --out-dir output
   ```

   **Dataset Training (High Performance):**
   ```bash
   cargo run --release --bin trainer -- \
     --target my_instrument/ \
     --out-dir output/ \
     --gens 10000 \
     --save-interval 50
   ```

   **Stop at 95% Similarity:**
   ```bash
   cargo run --release --bin trainer -- \
     --target my_instrument/ \
     --out-dir output/ \
     --similarity 95.0
   ```

   **Resume Training:**
   ```bash
   cargo run --release --bin trainer -- \
     --target my_instrument/ \
     --out-dir output/ \
     --resume output/latest.json
   ```

   *The trainer will automatically save `gen_XXXX_loss_YYYY.json` and `gen_XXXX_loss_YYYY.wav` files in the output directory.*

## How to Build

### Prerequisites

* Rust 1.75 or later (Edition 2024 support required)

### Build Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/neuro-fantasia.git
   cd neuro-fantasia
   ```

2. **Build the project**:
   ```bash
   cargo build --release
   ```

## Dependencies

This project uses the following key crates:

| Crate                                       | Description                  |
|---------------------------------------------|------------------------------|
| [fundsp](https://crates.io/crates/fundsp)   | Audio DSP Library            |
| [rustfft](https://crates.io/crates/rustfft) | FFT for spectral analysis    |
| [rayon](https://crates.io/crates/rayon)     | Parallelism for fitness eval |
| [clap](https://crates.io/crates/clap)       | CLI argument parsing         |
| [serde](https://crates.io/crates/serde)     | Serialization for JSON       |

## Credits

This project includes or utilizes the following audio resources:

* **Adventure Kid Waveforms (AKWF)** by Kristoffer Ekstrand.
  * Public Domain / CC0.
  * Source: [https://www.adventurekid.se/akrt/waveforms/](https://www.adventurekid.se/akrt/waveforms/)

* **Virtual Playing Orchestra 3** by Paul Battersby.
  * Used for extracting single-cycle waveforms for training.
  * Source: [http://virtualplaying.com](http://virtualplaying.com)
  * Incorporates samples from:
    * **Sonatina Symphonic Orchestra** (Creative Commons Sampling Plus 1.0)
    * **No Budget Orchestra** (CC BY-SA 4.0)
    * **VSCO 2 Community Edition** (CC0 1.0)
    * **University of Iowa Electronic Music Studios**
    * **Philharmonia Orchestra** (CC BY-SA 3.0)

## Contributing

Contributions are welcome!
Whether you want to fix a bug, add a feature, or improve documentation:

* Submit an **Issue** or **Pull Request**.
* Share ideas and discuss design or architecture.

## License

This project is licensed under either of

* Apache License, Version 2.0
* MIT license

at your option.
