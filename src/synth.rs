//! # Synthesizer Module / 合成器模块
//!
//! This module defines the `PatchGenome` (parameters) and the `create_graph` function which builds the DSP graph.
//! 此模块定义了 `PatchGenome`（参数）和构建 DSP 图的 `create_graph` 函数。

use crate::resources::WAVETABLES;
use fundsp::hacker32::*;
use serde::{Deserialize, Serialize};

// --- Patch Genome ---

/// Represents the genetic code of a synthesizer patch.
/// 代表合成器音色的遗传代码。
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(default)]
pub struct PatchGenome {
    // --- Oscillators / 振荡器 ---
    pub osc1_idx: f32, // Base wavetable position (0.0-1.0)
    pub osc2_idx: f32,
    pub detune: f32,
    pub osc_mix: f32,
    pub noise_mix: f32,
    pub noise_attack: f32,
    pub noise_decay: f32,

    // --- FM & Morph / 调频与变形 (New) ---
    pub fm_amount: f32,         // Osc1 modulates Osc2 freq (Linear FM)
    pub osc_morph_env_amt: f32, // Filter Env modulates Wavetable Position
    pub osc_morph_lfo_amt: f32, // LFOs modulate Wavetable Position

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
    pub drive: f32,      // Post-filter drive
    pub saturation: f32, // Pre-filter saturation
    pub env_curve: f32,  // Envelope curvature
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

            fm_amount: 0.0,
            osc_morph_env_amt: 0.0,
            osc_morph_lfo_amt: 0.0,

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
            env_curve: 2.0,
            chorus_mix: 0.0,
            reverb_mix: 0.0,
            master_vol: 0.8,
        }
    }
}

// --- Morphing Wavetable Oscillator ---

/// An oscillator that can morph through the global wavetable bank.
/// Inputs: [Frequency (Hz), Morph Position (0.0-1.0)]
/// 一个可以在全局波表库中变形的振荡器。
/// 输入：[频率 (Hz), 变形位置 (0.0-1.0)]
#[derive(Clone)]
pub struct MorphingWavetableOsc {
    phase: f32,
    sample_rate: f32,
}

impl MorphingWavetableOsc {
    pub fn new() -> Self {
        Self {
            phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl AudioNode for MorphingWavetableOsc {
    const ID: u64 = 0x4D_57_54_4F; // "MWTO"
    type Inputs = U2; // Freq, Morph
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
        // Ensure morph is in 0.0-1.0 range (wrap or clamp?)
        // Clamp is safer for "scanning", Wrap is better for "continuous".
        // Let's use Clamp to avoid jumping from last to first (which might be very different).
        let morph = input[1].clamp(0.0, 0.9999);

        let delta = freq / self.sample_rate;
        self.phase += delta;
        self.phase -= self.phase.floor();

        // Access Global Bank
        if let Some(bank) = WAVETABLES.get() {
            let bank_len = bank.len();
            if bank_len == 0 {
                return Frame::from([0.0]);
            }

            // Calculate position in bank
            let table_pos = morph * bank_len as f32;
            let table_idx = table_pos as usize; // Integer part
            let table_frac = table_pos - table_idx as f32; // Fractional part for morphing

            // Safety check
            let idx_a = table_idx % bank_len;
            let idx_b = (table_idx + 1) % bank_len;

            let table_a = &bank.tables[idx_a];
            let table_b = &bank.tables[idx_b];

            // Sample within the tables
            let table_len = table_a.len();
            let phase_pos = self.phase * table_len as f32;
            let sample_idx = phase_pos as usize;
            let sample_frac = phase_pos - sample_idx as f32;

            // Interpolate Table A
            let a0 = table_a[sample_idx % table_len];
            let a1 = table_a[(sample_idx + 1) % table_len];
            let val_a = a0 + (a1 - a0) * sample_frac;

            // Interpolate Table B
            let b0 = table_b[sample_idx % table_len];
            let b1 = table_b[(sample_idx + 1) % table_len];
            let val_b = b0 + (b1 - b0) * sample_frac;

            // Morph between Table A and B
            let final_val = val_a + (val_b - val_a) * table_frac;

            Frame::from([final_val])
        } else {
            Frame::from([0.0])
        }
    }
}

// --- Synthesizer Engine / 合成引擎 ---

/// Creates a playable DSP graph from the genome.
pub fn create_graph(genome: &PatchGenome, midi_note: f32, duration: f64) -> Box<dyn AudioUnit> {
    let hz = midi_to_hz(midi_note);

    // Ensure wavetables are loaded (init if mostly for tests, but main loads it)
    if WAVETABLES.get().is_none() {
        // Fallback for tests if needed, though main handles it.
    }

    // --- LFO Section ---
    let lfo1_env = lfo_fade(genome.lfo1_delay, genome.lfo1_fade);
    let lfo2_env = lfo_fade(genome.lfo2_delay, genome.lfo2_fade);

    // LFO1: Filter / Morph
    let lfo1 = sine_hz(genome.lfo1_rate) * lfo1_env;
    let cutoff_mod = lfo1.clone() * genome.lfo1_amt_cutoff;
    let lfo1_morph_mod = lfo1 * genome.osc_morph_lfo_amt;

    // LFO2: Pitch / Morph
    let lfo2 = sine_hz(genome.lfo2_rate) * lfo2_env;
    let pitch_mod = lfo2.clone() * genome.lfo2_amt_pitch;
    let lfo2_morph_mod = lfo2 * genome.osc_morph_lfo_amt;

    // --- Envelopes ---
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

    // Envelope for Morphing (using Filter Env for now, common in synths)
    let morph_env_mod = filter_env.clone() * genome.osc_morph_env_amt;

    // --- Oscillators & FM ---

    // Morph Modulation Logic:
    // Base Index + LFO + Env
    // We create control signals for morph inputs.
    // LFO1 modulates Osc1 morph, LFO2 modulates Osc2 morph (arbitrary choice for variety)

    // Osc 1 Morph Control
    let osc1_base = dc(genome.osc1_idx);
    let osc1_morph_ctrl = osc1_base + lfo1_morph_mod + morph_env_mod.clone();

    // Osc 2 Morph Control
    let osc2_base = dc(genome.osc2_idx);
    let osc2_morph_ctrl = osc2_base + lfo2_morph_mod + morph_env_mod;

    // Nodes
    let osc1_node = An(MorphingWavetableOsc::new());
    let osc2_node = An(MorphingWavetableOsc::new());

    // Osc 1 Setup
    let freq1 = dc(hz) + pitch_mod.clone();
    let osc1_out = (freq1 | osc1_morph_ctrl) >> osc1_node;

    // FM Logic: Osc1 Output -> Modulates Osc2 Freq
    // FM Amount is ratio relative to base freq? Or raw Hz?
    // Yamaha FM uses ratios. Linear FM adds (Modulator * Amt) to Carrier Freq.
    // Let's use Linear FM: Freq2 = Base + Detune + Vibrato + (Osc1 * FM_Amt * Multiplier)
    // To make FM effective, the amount often needs to scale with frequency or be large.
    let fm_signal = osc1_out.clone() * (genome.fm_amount * 1000.0); // Scale up for audible effect

    // Osc 2 Setup
    let detune_hz = hz * (genome.detune * 0.02);
    let freq2_base = dc(hz + detune_hz) + pitch_mod;
    let freq2_final = freq2_base + fm_signal; // Add FM

    let osc2_out = (freq2_final | osc2_morph_ctrl) >> osc2_node;

    // --- Mixer ---
    let osc_blended = (osc1_out * (1.0 - genome.osc_mix)) + (osc2_out * genome.osc_mix);

    // Noise (Attack transient)
    let noise_env = adsr_curved(genome.noise_attack, genome.noise_decay, 0.0, 0.0, 0.0, 1.0);
    let noise_src = (pink() * noise_env) * genome.noise_mix;

    // Pre-Filter Mix
    let raw_src = (osc_blended * (1.0 - genome.noise_mix)) + noise_src;

    // --- Pre-Filter Saturation ---
    let sat_amount = genome.saturation * 5.0;
    let drive_node = map(move |f: &Frame<f32, U1>| {
        let x = f[0] * (1.0 + sat_amount);
        if sat_amount > 0.001 { x.tanh() } else { x }
    });

    let saturated_src = raw_src >> drive_node;

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
            let phase = t / a;
            if curve > 1.0 {
                phase.powf(1.0 / curve)
            } else {
                phase.powf(curve)
            }
        } else if t < a + d {
            // Decay: 1 -> s
            let phase = (t - a) / d; // 0 -> 1
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
