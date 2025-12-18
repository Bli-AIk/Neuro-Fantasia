use crate::resources::WAVETABLES;
use fundsp::hacker32::*;
use serde::{Deserialize, Serialize};

// --- Patch Genome ---

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct PatchGenome {
    pub osc1_idx: f32,
    pub osc2_idx: f32,
    pub detune: f32,
    pub osc_mix: f32,
    pub noise_mix: f32,
    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,
    pub cutoff: f32,
    pub resonance: f32,
    pub filter_type: f32,
    pub filter_env_amt: f32,
    pub lfo_rate: f32,
    pub lfo_amt_pitch: f32,
    pub lfo_amt_cutoff: f32,
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

// --- Synthesizer Engine ---

pub fn create_graph(genome: &PatchGenome, midi_note: f32) -> Box<dyn AudioUnit> {
    let hz = midi_to_hz(midi_note);

    // Wavetable logic placeholder (using saw/square for now)
    let bank = WAVETABLES.get().expect("Wavetables not initialized");
    let _ = bank;

    let lfo = sine_hz(genome.lfo_rate);
    let pitch_mod = lfo.clone() * genome.lfo_amt_pitch;

    // Using simple shapes for now
    let freq1 = dc(hz) + pitch_mod.clone();
    let osc1 = freq1 >> saw();

    let detune_hz = hz * (genome.detune * 0.02);
    let freq2 = dc(hz + detune_hz) + pitch_mod;
    let osc2 = freq2 >> square();

    // Mix (Mono)
    let osc_blended = (osc1 * (1.0 - genome.osc_mix)) + (osc2 * genome.osc_mix);
    let src_mono = (osc_blended * (1.0 - genome.noise_mix)) + (pink() * genome.noise_mix);

    // Envelope
    let env_node = adsr_fixed(
        genome.attack,
        genome.decay,
        genome.sustain,
        genome.release,
        1.0,
    );

    // Filter
    let env_mod = env_node.clone() * genome.filter_env_amt;
    let lfo_mod = lfo * genome.lfo_amt_cutoff;

    let cutoff_hz = 20.0 + (20000.0 - 20.0) * (genome.cutoff * genome.cutoff);

    let raw_cutoff = dc(cutoff_hz) + (env_mod * 1000.0) + (lfo_mod * 500.0);
    let clamped_cutoff = raw_cutoff >> map(|x: &Frame<f32, U1>| x[0].clamp(20.0, 20000.0));

    let q = dc(genome.resonance * 10.0 + 0.1);

    let src_with_env = src_mono * env_node.clone();

    // Drive & Tanh
    let drive_amt = 1.0 + genome.drive * 5.0;
    let driven = src_with_env >> (pass() * drive_amt);
    let saturated = driven >> map(|x: &Frame<f32, U1>| x[0].tanh());

    let final_vol = genome.master_vol;

    // Filter Branches
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

fn adsr_fixed(
    a: f32,
    d: f32,
    s: f32,
    r: f32,
    hold: f32,
) -> An<impl AudioNode<Inputs = U0, Outputs = U1>> {
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
