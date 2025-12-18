//! # Synthesizer Module / 合成器模块
//!
//! This module defines the `PatchGenome` (parameters) and the `create_graph` function which builds the DSP graph.
//! 此模块定义了 `PatchGenome`（参数）和构建 DSP 图的 `create_graph` 函数。

use crate::resources::WAVETABLES;
use fundsp::hacker32::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// --- Patch Genome ---

/// Represents the genetic code of a synthesizer patch.
/// 代表合成器音色的遗传代码。
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct PatchGenome {
    // --- Oscillators / 振荡器 ---
    pub osc1_idx: f32,
    pub osc2_idx: f32,
    pub detune: f32,
    pub osc_mix: f32,
    pub noise_mix: f32,

    // --- Envelope / 包络 (ADSR) ---
    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,

    // --- Filter / 滤波器 ---
    pub cutoff: f32,
    pub resonance: f32,
    pub filter_type: f32,
    pub filter_env_amt: f32,

    // --- LFO / 低频振荡器 ---
    pub lfo_rate: f32,
    pub lfo_amt_pitch: f32,
    pub lfo_amt_cutoff: f32,

    // --- FX & Output / 效果与输出 ---
    pub drive: f32,
    pub master_vol: f32,
}

impl Default for PatchGenome {
    fn default() -> Self {
        Self {
            osc1_idx: 0.0,
            osc2_idx: 0.1,
            detune: 0.0,
            osc_mix: 0.5,
            noise_mix: 0.0,
            attack: 0.01,
            decay: 0.1,
            sustain: 0.8,
            release: 0.1,
            cutoff: 0.8,
            resonance: 0.2,
            filter_type: 0.0,
            filter_env_amt: 0.0,
            lfo_rate: 1.0,
            lfo_amt_pitch: 0.0,
            lfo_amt_cutoff: 0.0,
            drive: 0.0,
            master_vol: 0.8,
        }
    }
}

// --- Wavetable Oscillator ---

/// A custom oscillator that reads from a shared Wavetable.
/// 从共享波表读取的自定义振荡器。
#[derive(Clone)]
pub struct WavetableOsc {
    table: Arc<Vec<f32>>,
    phase: f32,
    sample_rate: f32,
}

impl WavetableOsc {
    pub fn new(table: Arc<Vec<f32>>) -> Self {
        Self {
            table,
            phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl AudioNode for WavetableOsc {
    const ID: u64 = 0x57_54_4F_53; // "WTOS"
    type Inputs = U1;
    type Outputs = U1;

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn set_sample_rate(&mut self, rate: f64) {
        self.sample_rate = rate as f32;
    }

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let freq = input[0];
        let delta = freq / self.sample_rate;
        self.phase += delta;
        self.phase -= self.phase.floor(); // Wrap phase to [0.0, 1.0)

        let len = self.table.len();
        let pos = self.phase * len as f32;
        let idx = pos as usize;
        let frac = pos - idx as f32;

        // Linear Interpolation
        let s0 = self.table[idx % len];
        let s1 = self.table[(idx + 1) % len];
        let val = s0 + (s1 - s0) * frac;

        Frame::from([val])
    }
}

// --- Synthesizer Engine / 合成引擎 ---

/// Creates a playable DSP graph from the genome.
/// 根据基因组创建一个可播放的 DSP 图。
pub fn create_graph(genome: &PatchGenome, midi_note: f32) -> Box<dyn AudioUnit> {
    let hz = midi_to_hz(midi_note);

    // Ensure wavetables are loaded.
    let bank = WAVETABLES.get().expect("Wavetables not initialized");

    // Select tables based on genome
    let table1_idx = (genome.osc1_idx * bank.len() as f32).floor() as usize;
    let table2_idx = (genome.osc2_idx * bank.len() as f32).floor() as usize;
    let table1 = bank.get(table1_idx);
    let table2 = bank.get(table2_idx);

    // --- LFO Section ---
    let lfo = sine_hz(genome.lfo_rate);
    let pitch_mod = lfo.clone() * genome.lfo_amt_pitch;

    // --- Oscillators ---
    // Use custom WavetableOsc
    let osc1_node = An(WavetableOsc::new(table1));
    let osc2_node = An(WavetableOsc::new(table2));

    let freq1 = dc(hz) + pitch_mod.clone();
    let osc1 = freq1 >> osc1_node;

    let detune_hz = hz * (genome.detune * 0.02);
    let freq2 = dc(hz + detune_hz) + pitch_mod;
    let osc2 = freq2 >> osc2_node;

    // --- Mixer ---
    let osc_blended = (osc1 * (1.0 - genome.osc_mix)) + (osc2 * genome.osc_mix);

    // Noise with dedicated Envelope (Attack only)
    // 噪声带有专用包络（仅起音）
    let noise_env = adsr_fixed(0.005, 0.1, 0.0, 0.0, 0.0); // Fast attack, short decay
    let noise_src = (pink() * noise_env) * genome.noise_mix;

    let src_mono = (osc_blended * (1.0 - genome.noise_mix)) + noise_src;

    // --- Main Envelope ---
    let env_node = adsr_fixed(
        genome.attack,
        genome.decay,
        genome.sustain,
        genome.release,
        1.0,
    );

    // --- Filter Modulation ---
    let env_mod = env_node.clone() * genome.filter_env_amt;
    let lfo_mod = lfo * genome.lfo_amt_cutoff;

    // Logarithmic Cutoff Mapping
    // 20Hz * (1000)^cutoff -> range 20Hz to 20kHz
    let cutoff_hz_base = 20.0 * (1000.0f32).powf(genome.cutoff);

    let raw_cutoff = dc(cutoff_hz_base) + (env_mod * 1000.0) + (lfo_mod * 500.0);
    let clamped_cutoff = raw_cutoff >> map(|x: &Frame<f32, U1>| x[0].clamp(20.0, 20000.0));

    let q = dc(genome.resonance * 10.0 + 0.1);

    // Apply Amp Envelope
    let src_with_env = src_mono * env_node.clone();

    // --- Drive ---
    let drive_amt = 1.0 + genome.drive * 5.0;
    let driven = src_with_env >> (pass() * drive_amt);
    let saturated = driven >> map(|x: &Frame<f32, U1>| x[0].tanh());

    let final_vol = genome.master_vol;

    // --- Filter Branching ---
    let make_graph = |mode: i32| -> Box<dyn AudioUnit> {
        let cutoff_node = clamped_cutoff.clone();
        let q_node = q.clone();
        let src_node = saturated.clone();
        let vol_node = mul(final_vol);

        if mode == 0 {
            Box::new(((src_node | cutoff_node | q_node) >> lowpass()) >> vol_node >> split::<U2>())
        } else if mode == 1 {
            Box::new(((src_node | cutoff_node | q_node) >> highpass()) >> vol_node >> split::<U2>())
        } else {
            Box::new(((src_node | cutoff_node | q_node) >> bandpass()) >> vol_node >> split::<U2>())
        }
    };

    if genome.filter_type < 0.33 {
        make_graph(0)
    } else if genome.filter_type < 0.66 {
        make_graph(1)
    } else {
        make_graph(2)
    }
}

fn midi_to_hz(note: f32) -> f32 {
    440.0 * 2.0f32.powf((note - 69.0) / 12.0)
}

pub fn render_sample(genome: &PatchGenome, note: f32, duration: f64) -> Vec<f32> {
    let mut graph = create_graph(genome, note);
    let sample_rate = 44100.0;
    let samples = (duration * sample_rate as f64) as usize;

    let mut buffer = Vec::with_capacity(samples);

    graph.reset();
    graph.set_sample_rate(sample_rate as f64);

    for _ in 0..samples {
        let frame = graph.get_stereo();
        buffer.push((frame.0 + frame.1) as f32 * 0.5);
    }

    buffer
}

/// Helper: ADSR with Hold time (gate).
/// a, d, s, r are standard parameters. `hold` is the time the key is pressed.
fn adsr_fixed(
    a: f32,
    d: f32,
    s: f32,
    r: f32,
    hold: f32,
) -> An<impl AudioNode<Inputs = U0, Outputs = U1> + Clone> {
    envelope(move |t| {
        let t = t as f32;
        if t < a {
            t / a
        } else if t < a + d {
            1.0 + (s - 1.0) * (t - a) / d
        } else if t < a + d + hold {
            s
        } else if t < a + d + hold + r {
            let rel_t = t - (a + d + hold);
            s * (1.0 - rel_t / r)
        } else {
            0.0
        }
    })
}
