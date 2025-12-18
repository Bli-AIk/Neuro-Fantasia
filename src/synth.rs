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
#[serde(default)]
pub struct PatchGenome {
    // --- Oscillators / 振荡器 ---
    pub osc1_idx: f32,
    pub osc2_idx: f32,
    pub detune: f32,
    pub osc_mix: f32,
    pub noise_mix: f32,
    pub noise_attack: f32,
    pub noise_decay: f32,

    // --- Amp Envelope / 音量包络 (ADSR) ---
    pub amp_attack: f32,
    pub amp_decay: f32,
    pub amp_sustain: f32,
    pub amp_release: f32,

    // --- Filter Envelope / 滤波器包络 (ADSR) ---
    pub filter_attack: f32,
    pub filter_decay: f32,
    pub filter_sustain: f32,
    pub filter_release: f32,
    pub filter_env_amt: f32,

    // --- Filter / 滤波器 ---
    pub cutoff: f32,
    pub resonance: f32,
    pub filter_type: f32,

    // --- LFO 1 (Modulation/Filter/PWM) ---
    pub lfo1_rate: f32,
    pub lfo1_amt_cutoff: f32,
    pub lfo1_delay: f32,
    pub lfo1_fade: f32,

    // --- LFO 2 (Vibrato/Pitch) ---
    pub lfo2_rate: f32,
    pub lfo2_amt_pitch: f32,
    pub lfo2_delay: f32,
    pub lfo2_fade: f32,

    // --- FX & Output / 效果与输出 ---
    pub drive: f32,      // Post-filter drive (existing)
    pub saturation: f32, // Pre-filter saturation (new)
    pub env_curve: f32,  // Envelope curvature (0.0 linear -> large exponential)
    pub chorus_mix: f32,
    pub reverb_mix: f32,
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
            noise_attack: 0.005,
            noise_decay: 0.1,

            amp_attack: 0.01,
            amp_decay: 0.1,
            amp_sustain: 0.8,
            amp_release: 0.1,

            filter_attack: 0.01,
            filter_decay: 0.1,
            filter_sustain: 0.5,
            filter_release: 0.1,
            filter_env_amt: 0.0,

            cutoff: 0.8,
            resonance: 0.2,
            filter_type: 0.0,

            lfo1_rate: 1.0,
            lfo1_amt_cutoff: 0.0,
            lfo1_delay: 0.0,
            lfo1_fade: 0.1,

            lfo2_rate: 5.0,
            lfo2_amt_pitch: 0.0,
            lfo2_delay: 0.0,
            lfo2_fade: 0.1,

            drive: 0.0,
            saturation: 0.0,
            env_curve: 2.0, // Default to quadratic (natural)
            chorus_mix: 0.0,
            reverb_mix: 0.0,
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
pub fn create_graph(genome: &PatchGenome, midi_note: f32, duration: f64) -> Box<dyn AudioUnit> {
    let hz = midi_to_hz(midi_note);

    // Ensure wavetables are loaded.
    let bank = WAVETABLES.get().expect("Wavetables not initialized");

    // Select tables based on genome
    let table1_idx = (genome.osc1_idx * bank.len() as f32).floor() as usize;
    let table2_idx = (genome.osc2_idx * bank.len() as f32).floor() as usize;
    let table1 = bank.get(table1_idx);
    let table2 = bank.get(table2_idx);

    // --- LFO Section ---
    // LFO Fade envelopes
    let lfo1_env = lfo_fade(genome.lfo1_delay, genome.lfo1_fade);
    let lfo2_env = lfo_fade(genome.lfo2_delay, genome.lfo2_fade);

    // LFO1: Filter / Timbre Modulation
    let lfo1 = sine_hz(genome.lfo1_rate) * lfo1_env;
    let cutoff_mod = lfo1 * genome.lfo1_amt_cutoff;

    // LFO2: Pitch / Vibrato
    let lfo2 = sine_hz(genome.lfo2_rate) * lfo2_env;
    let pitch_mod = lfo2 * genome.lfo2_amt_pitch;

    // --- Oscillators ---
    let osc1_node = An(WavetableOsc::new(table1));
    let osc2_node = An(WavetableOsc::new(table2));

    let freq1 = dc(hz) + pitch_mod.clone();
    let osc1 = freq1 >> osc1_node;

    let detune_hz = hz * (genome.detune * 0.02);
    let freq2 = dc(hz + detune_hz) + pitch_mod;
    let osc2 = freq2 >> osc2_node;

    // --- Mixer ---
    let osc_blended = (osc1 * (1.0 - genome.osc_mix)) + (osc2 * genome.osc_mix);

    // Noise (Attack transient) - Noise uses linear ADSR usually, but curved is fine too.
    let noise_env = adsr_curved(
        genome.noise_attack,
        genome.noise_decay,
        0.0,
        0.0,
        0.0,
        1.0, // Linear noise envelope usually works well for simple transients
    );
    let noise_src = (pink() * noise_env) * genome.noise_mix;

    // Pre-Filter Mix
    let raw_src = (osc_blended * (1.0 - genome.noise_mix)) + noise_src;

    // --- Pre-Filter Saturation ---
    // Simulating analog gain staging or VCA before filter.
    // Soft clip if saturation > 0
    let sat_amount = genome.saturation * 5.0; // Scale 0-1 to reasonable drive

    // We apply saturation unconditionally to avoid type mismatch in conditional branches.
    // When sat_amount is 0, the effect is negligible (linear).
    // U1 -> U1
    let drive_node = map(move |f: &Frame<f32, U1>| {
        let x = f[0] * (1.0 + sat_amount);
        // Soft clipping: tanh is standard for this
        if sat_amount > 0.001 {
            x.tanh()
        } else {
            x // Bypass if practically zero
        }
    });

    let saturated_src = raw_src >> drive_node;

    // --- Envelopes ---
    // Hold time is set to duration to simulate key press length
    let hold_time = duration as f32;

    let amp_env = adsr_curved(
        genome.amp_attack,
        genome.amp_decay,
        genome.amp_sustain,
        genome.amp_release,
        hold_time,
        genome.env_curve,
    );

    let filter_env = adsr_curved(
        genome.filter_attack,
        genome.filter_decay,
        genome.filter_sustain,
        genome.filter_release,
        hold_time,
        genome.env_curve,
    );

    // --- Filter Modulation ---
    let env_mod = filter_env * genome.filter_env_amt;

    // Logarithmic Cutoff
    let cutoff_hz_base = 20.0 * (1000.0f32).powf(genome.cutoff);

    // Combine modulations: Base + Env + LFO
    let raw_cutoff = dc(cutoff_hz_base) + (env_mod * 1000.0) + (cutoff_mod * 0.5);
    let clamped_cutoff = raw_cutoff >> map(|x: &Frame<f32, U1>| x[0].clamp(20.0, 20000.0));

    let q = dc(genome.resonance * 10.0 + 0.1);

    // --- Post-Filter Drive & Amp ---
    let g_drive = genome.drive;
    let g_reverb_mix = genome.reverb_mix;
    let g_chorus_mix = genome.chorus_mix;
    let g_master_vol = genome.master_vol;

    // --- Filter Branching & Final Chain ---
    let make_full_graph = move |mode: i32| -> Box<dyn AudioUnit> {
        let c = clamped_cutoff.clone();
        let q_val = q.clone();
        let src = saturated_src.clone();

        let drive_amt = 1.0 + g_drive * 5.0;

        // Post-filter drive
        let input_forced = map(|f: &Frame<f32, U1>| f.clone());
        let drive = (input_forced * drive_amt) >> map(|f: &Frame<f32, U1>| f[0].tanh());

        // Amp
        let amp_forced = amp_env.clone() >> map(|f: &Frame<f32, U1>| f.clone());
        let mono_out = (drive * amp_forced) * g_master_vol;

        // FX Chain Construction
        // Chorus (Mono -> Stereo)
        let chorus_l = chorus(0, 0.015, 0.2, 0.5);
        let chorus_r = chorus(1, 0.015, 0.2, 0.55);
        let chorus_stereo = chorus_l | chorus_r;

        let chorus_path = split() >> chorus_stereo;
        let dry_path = split();

        let mixed_chorus = (dry_path * (1.0 - g_chorus_mix)) & (chorus_path * g_chorus_mix);

        // Reverb (Stereo -> Stereo)
        let reverb_op = reverb_stereo(10.0, 2.0, 0.5);
        let dry_reverb = multipass::<U2>();

        let final_fx =
            mixed_chorus >> ((dry_reverb * (1.0 - g_reverb_mix)) & (reverb_op * g_reverb_mix));

        if mode == 0 {
            Box::new((src | c | q_val) >> lowpass() >> mono_out >> final_fx)
        } else if mode == 1 {
            Box::new((src | c | q_val) >> highpass() >> mono_out >> final_fx)
        } else {
            Box::new((src | c | q_val) >> bandpass() >> mono_out >> final_fx)
        }
    };

    if genome.filter_type < 0.33 {
        make_full_graph(0)
    } else if genome.filter_type < 0.66 {
        make_full_graph(1)
    } else {
        make_full_graph(2)
    }
}

fn midi_to_hz(note: f32) -> f32 {
    440.0 * 2.0f32.powf((note - 69.0) / 12.0)
}

pub fn render_sample(genome: &PatchGenome, note: f32, duration: f64) -> Vec<f32> {
    // We add a bit of tail time for release and reverb
    let tail = genome.amp_release as f64 + 1.0;
    let total_duration = duration + tail;

    let mut graph = create_graph(genome, note, duration);
    let sample_rate = 44100.0;
    let samples = (total_duration * sample_rate as f64) as usize;

    let mut buffer = Vec::with_capacity(samples);

    graph.reset();
    graph.set_sample_rate(sample_rate as f64);

    for _ in 0..samples {
        let frame = graph.get_stereo();
        buffer.push((frame.0 + frame.1) as f32 * 0.5);
    }

    buffer
}

/// Helper: ADSR with Curve control.
/// curve = 1.0 (linear).
/// curve > 1.0: Attack becomes convex (punchy), Decay/Release become concave (natural analog).
/// curve < 1.0: Opposite (slow attack, linear-ish decay).
fn adsr_curved(
    a: f32,
    d: f32,
    s: f32,
    r: f32,
    hold: f32,
    curve: f32,
) -> An<impl AudioNode<Inputs = U0, Outputs = U1> + Clone> {
    envelope(move |t| {
        let t = t as f32;
        if t < a {
            // Attack: 0 -> 1
            // Use inverse curve for "convex" (charging capacitor) feel if curve > 1
            let phase = t / a;
            if curve > 1.0 {
                phase.powf(1.0 / curve)
            } else {
                phase.powf(curve)
            }
        } else if t < a + d {
            // Decay: 1 -> s
            let phase = (t - a) / d; // 0 -> 1
            // Concave decay
            let val = 1.0 - phase;
            let curved_val = val.powf(curve);
            s + (1.0 - s) * curved_val
        } else if t < a + d + hold {
            // Sustain
            s
        } else if t < a + d + hold + r {
            // Release: s -> 0
            let phase = (t - (a + d + hold)) / r; // 0 -> 1
            let val = 1.0 - phase;
            s * val.powf(curve)
        } else {
            0.0
        }
    })
}

/// Helper: Fade-in Envelope for LFOs
fn lfo_fade(delay: f32, fade: f32) -> An<impl AudioNode<Inputs = U0, Outputs = U1> + Clone> {
    envelope(move |t| {
        let t = t as f32;
        if t < delay {
            0.0
        } else if t < delay + fade {
            (t - delay) / fade
        } else {
            1.0
        }
    })
}
