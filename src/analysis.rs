//! # Analysis Module / 分析模块
//!
//! This module provides tools to extract psychoacoustic features from audio signals and calculate the similarity (loss) between them.

use rustfft::{FftPlanner, num_complex::Complex};

/// Container for extracted audio features.
/// 提取的音频特征的容器。
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFeatures {
    /// Spectrum of the attack phase (first 100ms).
    /// 起音阶段的频谱（前 100ms）。
    pub attack_spectrum: Vec<f32>,
    /// Spectrum of the sustain phase (middle).
    /// 延音阶段的频谱（中间）。
    pub sustain_spectrum: Vec<f32>,
    /// RMS Envelope over time (normalized to fixed points).
    /// 随时间变化的 RMS 包络（归一化为固定点）。
    pub rms_envelope: Vec<f32>,
    /// Global RMS.
    pub rms: f32,
}

/// Extracts features from a raw audio buffer.
/// 从原始音频缓冲区提取特征。
pub fn extract_features(audio: &[f32], sample_rate: u32) -> AudioFeatures {
    let fft_size = 2048;

    // 1. Attack Spectrum (0 ~ 100ms)
    // 1. 起音频谱 (0 ~ 100ms)
    let attack_len = (sample_rate as f32 * 0.1) as usize; // 100ms
    let attack_audio = if audio.len() > attack_len {
        &audio[0..attack_len]
    } else {
        audio
    };
    // Take a window from the attack (center of attack or just start?)
    // For transient, taking the start 2048 samples is good.
    let attack_window = if attack_audio.len() >= fft_size {
        &attack_audio[0..fft_size]
    } else {
        attack_audio // Will be padded
    };
    let attack_spectrum = compute_mel_spectrum(attack_window, sample_rate, fft_size);

    // 2. Sustain Spectrum (Middle)
    // 2. 延音频谱 (中间)
    let center = audio.len() / 2;
    let start = if center > fft_size / 2 {
        center - fft_size / 2
    } else {
        0
    };
    let end = (start + fft_size).min(audio.len());
    let sustain_window = &audio[start..end];
    let sustain_spectrum = compute_mel_spectrum(sustain_window, sample_rate, fft_size);

    // 3. RMS Envelope
    // 3. RMS 包络
    let rms_envelope = compute_rms_envelope(audio, 50); // 50 points normalized
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len().max(1) as f32).sqrt();

    AudioFeatures {
        attack_spectrum,
        sustain_spectrum,
        rms_envelope,
        rms,
    }
}

/// Calculates the difference (loss) between target and candidate features.
/// 计算目标特征和候选特征之间的差异（损失）。
pub fn calculate_loss(target: &AudioFeatures, candidate: &AudioFeatures) -> f32 {
    let mse = |a: &[f32], b: &[f32]| -> f32 {
        let len = a.len().min(b.len());
        if len == 0 {
            return 1.0;
        }
        a.iter()
            .zip(b.iter())
            .take(len)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            / len as f32
    };

    let attack_loss = mse(&target.attack_spectrum, &candidate.attack_spectrum);
    let sustain_loss = mse(&target.sustain_spectrum, &candidate.sustain_spectrum);
    let envelope_loss = mse(&target.rms_envelope, &candidate.rms_envelope);
    let rms_loss = (target.rms - candidate.rms).powi(2);

    // Weights:
    // Spectral match is key. Attack is important for instrument ID. Envelope is important for dynamics.
    // 权重：频谱匹配是关键。起音对乐器识别很重要。包络对动态很重要。
    attack_loss * 1.5 + sustain_loss * 1.0 + envelope_loss * 2.0 + rms_loss * 0.5
}

/// Computes RMS envelope normalized to specific number of points.
fn compute_rms_envelope(audio: &[f32], points: usize) -> Vec<f32> {
    if audio.is_empty() {
        return vec![0.0; points];
    }

    let chunk_size = (audio.len() as f32 / points as f32).ceil() as usize;
    if chunk_size == 0 {
        return vec![0.0; points];
    }

    let mut envelope = Vec::with_capacity(points);
    for i in 0..points {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(audio.len());
        if start >= audio.len() {
            envelope.push(0.0);
            continue;
        }

        let chunk = &audio[start..end];
        let sum_sq = chunk.iter().map(|s| s * s).sum::<f32>();
        let rms = (sum_sq / chunk.len() as f32).sqrt();
        envelope.push(rms);
    }
    envelope
}

fn compute_mel_spectrum(audio: &[f32], sample_rate: u32, fft_size: usize) -> Vec<f32> {
    let mut input: Vec<Complex<f32>> = audio
        .iter()
        .enumerate()
        .map(|(i, &sample)| {
            let window =
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos());
            Complex::new(sample * window, 0.0)
        })
        .collect();

    if input.len() < fft_size {
        input.resize(fft_size, Complex::new(0.0, 0.0));
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut input);

    let magnitudes: Vec<f32> = input.iter().take(fft_size / 2).map(|c| c.norm()).collect();

    compute_mel_energies(&magnitudes, sample_rate, fft_size, 40)
}

fn compute_mel_energies(
    magnitudes: &[f32],
    sample_rate: u32,
    fft_size: usize,
    num_filters: usize,
) -> Vec<f32> {
    let min_freq = 20.0;
    let max_freq = sample_rate as f32 / 2.0;

    let min_mel = hz_to_mel(min_freq);
    let max_mel = hz_to_mel(max_freq);

    let mel_points: Vec<f32> = (0..num_filters + 2)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (num_filters + 1) as f32)
        .collect();

    let bin_points: Vec<usize> = mel_points
        .iter()
        .map(|&m| {
            let hz = mel_to_hz(m);
            (hz * fft_size as f32 / sample_rate as f32).floor() as usize
        })
        .collect();

    let mut energies = vec![0.0; num_filters];

    for i in 0..num_filters {
        let start = bin_points[i];
        let center = bin_points[i + 1];
        let end = bin_points[i + 2];

        for f in start..end {
            if f >= magnitudes.len() {
                break;
            }

            let weight = if f < center {
                (f - start) as f32 / (center - start) as f32
            } else {
                (end - f) as f32 / (end - center) as f32
            };

            energies[i] += magnitudes[f] * weight;
        }
        energies[i] = (energies[i] + 1e-6).ln();
    }
    energies
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}
