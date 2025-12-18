//! # Synthesizer Module / 合成器模块
//!
//! This module defines the `PatchGenome` (parameters) and the `create_graph` function which builds the DSP graph.
//! 此模块定义了 `PatchGenome`（参数）和构建 DSP 图的 `create_graph` 函数。
//!
//! ## Architecture / 架构
//! The engine uses a Subtractive Synthesis architecture:
//! 引擎使用减法合成架构：
//! - **Source**: 2 Oscillators (Wavetable) + Noise Generator. / **源**：2 个振荡器（波表）+ 噪声发生器。
//! - **Shaping**: Tanh Distortion (Drive). / **整形**：Tanh 失真（驱动）。
//! - **Filter**: Multi-mode SVF (LP/HP/BP) modulated by Envelope and LFO. / **滤波器**：由包络和 LFO 调制的多种模式 SVF (LP/HP/BP)。
//! - **Modulation**: ADSR Envelope, LFO (Pitch, Cutoff). / **调制**：ADSR 包络，LFO（音高，截止频率）。
//!
//! ## Embedding / 嵌入
//! This module depends only on `fundsp` and standard types, making it easy to embed in game engines like Bevy.
//! 此模块仅依赖 `fundsp` 和标准类型，因此很容易嵌入到像 Bevy 这样的游戏引擎中。

use crate::resources::WAVETABLES;
use fundsp::hacker32::*;
use serde::{Deserialize, Serialize};

// --- Patch Genome ---

/// Represents the genetic code of a synthesizer patch.
/// 代表合成器音色的遗传代码。
///
/// Contains normalized values (mostly 0.0 to 1.0) which are mapped to DSP parameters inside `create_graph`.
/// 包含归一化值（大多为 0.0 到 1.0），这些值在 `create_graph` 内部映射到 DSP 参数。
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct PatchGenome {
    // --- Oscillators / 振荡器 ---
    /// Index of Wavetable 1 (mapped to bank size).
    /// 波表 1 的索引（映射到库大小）。
    pub osc1_idx: f32,
    /// Index of Wavetable 2.
    /// 波表 2 的索引。
    pub osc2_idx: f32,
    /// Detune amount between oscillators.
    /// 振荡器之间的失真量。
    pub detune: f32,
    /// Mix between Osc 1 (0.0) and Osc 2 (1.0).
    /// Osc 1 (0.0) 和 Osc 2 (1.0) 之间的混合。
    pub osc_mix: f32,
    /// Mix between Oscillators (0.0) and Noise (1.0).
    /// 振荡器 (0.0) 和噪声 (1.0) 之间的混合。
    pub noise_mix: f32,

    // --- Envelope / 包络 (ADSR) ---
    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,

    // --- Filter / 滤波器 ---
    /// Filter Cutoff Frequency (0.0..1.0 -> 20Hz..20kHz).
    /// 滤波器截止频率 (0.0..1.0 -> 20Hz..20kHz)。
    pub cutoff: f32,
    /// Resonance (Q factor).
    /// 共振 (Q 因子)。
    pub resonance: f32,
    /// Filter Type Selection (<0.33 LP, <0.66 HP, else BP).
    /// 滤波器类型选择 (<0.33 低通, <0.66 高通, 其他 带通)。
    pub filter_type: f32,
    /// Amount of Envelope modulation on Cutoff.
    /// 包络对截止频率的调制量。
    pub filter_env_amt: f32,

    // --- LFO / 低频振荡器 ---
    /// LFO Rate (Hz).
    /// LFO 速率 (Hz)。
    pub lfo_rate: f32,
    /// LFO Amount on Pitch (Vibrato).
    /// 音高上的 LFO 量 (颤音)。
    pub lfo_amt_pitch: f32,
    /// LFO Amount on Cutoff (Wah).
    /// 截止频率上的 LFO 量 (哇音)。
    pub lfo_amt_cutoff: f32,

    // --- FX & Output / 效果与输出 ---
    /// Distortion Drive.
    /// 失真驱动。
    pub drive: f32,
    /// Master Volume.
    /// 主音量。
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

// --- Synthesizer Engine / 合成引擎 ---

/// Creates a playable DSP graph from the genome.
/// 根据基因组创建一个可播放的 DSP 图。
///
/// Returns a `Box<dyn AudioUnit>` which processes audio buffers.
/// 返回一个处理音频缓冲区的 `Box<dyn AudioUnit>`。
pub fn create_graph(genome: &PatchGenome, midi_note: f32) -> Box<dyn AudioUnit> {
    let hz = midi_to_hz(midi_note);

    // Ensure wavetables are loaded.
    // 确保波表已加载。
    let bank = WAVETABLES.get().expect("Wavetables not initialized");
    let _ = bank; // Suppress unused warning (see below) / 抑制未使用的警告

    // --- LFO Section / LFO 部分 ---
    let lfo = sine_hz(genome.lfo_rate);
    let pitch_mod = lfo.clone() * genome.lfo_amt_pitch;

    // --- Oscillators / 振荡器 ---
    // Note: Ideally we use `wave(&arc)` here. Using simple shapes for robustness in this demo version.
    // 注意：理想情况下这里我们使用 `wave(&arc)`。在这个演示版本中，为了稳健性使用简单的形状。
    // TODO: Connect `WAVETABLES` to `fundsp` wave player when `wave` API is stable.

    let freq1 = dc(hz) + pitch_mod.clone();
    let osc1 = freq1 >> saw();

    let detune_hz = hz * (genome.detune * 0.02);
    let freq2 = dc(hz + detune_hz) + pitch_mod;
    let osc2 = freq2 >> square();

    // --- Mixer / 混音器 ---
    // Blend Oscillators (Mono) / 混合振荡器 (单声道)
    let osc_blended = (osc1 * (1.0 - genome.osc_mix)) + (osc2 * genome.osc_mix);
    // Add Noise / 添加噪声
    let src_mono = (osc_blended * (1.0 - genome.noise_mix)) + (pink() * genome.noise_mix);

    // --- Envelope / 包络 ---
    // ADSR Envelope for Amplitude / 幅度的 ADSR 包络
    let env_node = adsr_fixed(
        genome.attack,
        genome.decay,
        genome.sustain,
        genome.release,
        1.0,
    );

    // --- Filter Modulation / 滤波器调制 ---
    let env_mod = env_node.clone() * genome.filter_env_amt;
    let lfo_mod = lfo * genome.lfo_amt_cutoff;

    // Map normalized cutoff to Hz (Exponential curve)
    // 将归一化的截止频率映射到 Hz（指数曲线）
    let cutoff_hz = 20.0 + (20000.0 - 20.0) * (genome.cutoff * genome.cutoff);

    // Calculate final cutoff with modulation and clamping
    // 计算带有调制和钳位的最终截止频率
    let raw_cutoff = dc(cutoff_hz) + (env_mod * 1000.0) + (lfo_mod * 500.0);
    let clamped_cutoff = raw_cutoff >> map(|x: &Frame<f32, U1>| x[0].clamp(20.0, 20000.0));

    let q = dc(genome.resonance * 10.0 + 0.1);

    // Apply Amp Envelope to Source
    // 将振幅包络应用于源
    let src_with_env = src_mono * env_node.clone();

    // --- Drive / 驱动 (Distortion) ---
    let drive_amt = 1.0 + genome.drive * 5.0;
    let driven = src_with_env >> (pass() * drive_amt);
    // Soft clipping using Tanh / 使用 Tanh 的软削波
    let saturated = driven >> map(|x: &Frame<f32, U1>| x[0].tanh());

    let final_vol = genome.master_vol;

    // --- Filter Branching / 滤波器分支 ---
    // Construct the final graph based on selected filter type.
    // 根据选择的滤波器类型构建最终图。
    let make_graph = |mode: i32| -> Box<dyn AudioUnit> {
        let cutoff_node = clamped_cutoff.clone();
        let q_node = q.clone();
        let src_node = saturated.clone();
        let vol_node = mul(final_vol);

        // (Signal | Freq | Q) >> Filter >> Vol >> Split(Stereo)
        // (信号 | 频率 | Q) >> 滤波器 >> 音量 >> 分裂(立体声)
        if mode == 0 {
            Box::new(((src_node | cutoff_node | q_node) >> lowpass()) >> vol_node >> split::<U2>())
        } else if mode == 1 {
            Box::new(((src_node | cutoff_node | q_node) >> highpass()) >> vol_node >> split::<U2>())
        } else {
            Box::new(((src_node | cutoff_node | q_node) >> bandpass()) >> vol_node >> split::<U2>())
        }
    };

    if genome.filter_type < 0.33 {
        make_graph(0) // Lowpass / 低通
    } else if genome.filter_type < 0.66 {
        make_graph(1) // Highpass / 高通
    } else {
        make_graph(2) // Bandpass / 带通
    }
}

/// Helper to convert MIDI Note Number to Frequency (Hz).
/// 将 MIDI 音符编号转换为频率 (Hz) 的辅助函数。
fn midi_to_hz(note: f32) -> f32 {
    440.0 * 2.0f32.powf((note - 69.0) / 12.0)
}

/// Renders a fixed-duration sample of the patch.
/// 渲染音色的固定持续时间样本。
///
/// Useful for analysis and genetic evaluation.
/// 对分析和遗传评估很有用。
pub fn render_sample(genome: &PatchGenome, note: f32, duration: f64) -> Vec<f32> {
    let mut graph = create_graph(genome, note);
    let sample_rate = 44100.0;
    let samples = (duration * sample_rate as f64) as usize;

    let mut buffer = Vec::with_capacity(samples);

    graph.reset();
    graph.set_sample_rate(sample_rate as f64);

    // Process audio block by block or sample by sample
    // 逐块或逐样本处理音频
    for _ in 0..samples {
        let frame = graph.get_stereo();
        buffer.push((frame.0 + frame.1) as f32 * 0.5); // Mono mixdown / 单声道混合
    }

    buffer
}

/// Custom ADSR helper for fixed duration rendering.
/// 用于固定持续时间渲染的自定义 ADSR 辅助函数。
///
/// Simulates a key press held for `hold` seconds.
/// 模拟按键按住 `hold` 秒。
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
