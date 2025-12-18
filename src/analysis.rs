//! # Analysis Module / 分析模块
//!
//! This module provides tools to extract psychoacoustic features from audio signals and calculate the similarity (loss) between them.
//! 此模块提供从音频信号提取心理声学特征并计算它们之间相似度（损失）的工具。
//!
//! Key components:
//! 关键组件：
//! - **FFT**: Fast Fourier Transform for spectral analysis. / **FFT**: 用于频谱分析的快速傅里叶变换。
//! - **Mel Filterbank**: Maps linear frequency to human perceptual scale (Mel scale). / **Mel 滤波器组**: 将线性频率映射到人类感知尺度（Mel 标度）。
//! - **RMS**: Root Mean Square for envelope/loudness matching. / **RMS**: 用于包络/响度匹配的均方根。
//! - **Loss Function**: Weighted combination of Spectral Error and Amplitude Error. / **损失函数**: 频谱误差和振幅误差的加权组合。

use rustfft::{FftPlanner, num_complex::Complex};

/// Container for extracted audio features.
/// 提取的音频特征的容器。
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFeatures {
    /// Energies in 40 Mel frequency bands.
    /// 40 个 Mel 频带中的能量。
    pub mel_bands: Vec<f32>,
    /// Root Mean Square (Loudness).
    /// 均方根（响度）。
    pub rms: f32,
}

/// Extracts features from a raw audio buffer.
/// 从原始音频缓冲区提取特征。
///
/// Performs FFT, applies Mel windowing, and calculates RMS.
/// 执行 FFT，应用 Mel 加窗，并计算 RMS。
pub fn extract_features(audio: &[f32], sample_rate: u32) -> AudioFeatures {
    let fft_size = 2048;

    // Select a stable segment from the middle of the audio.
    // 从音频中间选择一个稳定的片段。
    let center = audio.len() / 2;
    let start = if center > fft_size / 2 {
        center - fft_size / 2
    } else {
        0
    };
    let end = (start + fft_size).min(audio.len());

    // Prepare input with Hann Window to reduce spectral leakage.
    // 使用 Hann 窗准备输入以减少频谱泄漏。
    let mut input: Vec<Complex<f32>> = audio[start..end]
        .iter()
        .enumerate()
        .map(|(i, &sample)| {
            let window =
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos());
            Complex::new(sample * window, 0.0)
        })
        .collect();

    // Zero padding if needed.
    // 如果需要，进行补零。
    if input.len() < fft_size {
        input.resize(fft_size, Complex::new(0.0, 0.0));
    }

    // Perform FFT.
    // 执行 FFT。
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut input);

    // Compute Magnitude Spectrum.
    // 计算幅度谱。
    let magnitudes: Vec<f32> = input
        .iter()
        .take(fft_size / 2) // Nyquist limit / 奈奎斯特极限
        .map(|c| c.norm())
        .collect();

    // Compute Mel Energies.
    // 计算 Mel 能量。
    let mel_bands = compute_mel_energies(&magnitudes, sample_rate, fft_size, 40);

    // Compute RMS.
    // 计算 RMS。
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();

    AudioFeatures { mel_bands, rms }
}

/// Calculates the difference (loss) between target and candidate features.
/// 计算目标特征和候选特征之间的差异（损失）。
///
/// Lower value means better match.
/// 值越低表示匹配越好。
pub fn calculate_loss(target: &AudioFeatures, candidate: &AudioFeatures) -> f32 {
    // Mean Squared Error of Mel Bands.
    // Mel 频带的均方误差。
    let mut mel_error = 0.0;
    for (t, c) in target.mel_bands.iter().zip(&candidate.mel_bands) {
        mel_error += (t - c).powi(2);
    }
    mel_error /= target.mel_bands.len() as f32;

    // Squared Error of RMS.
    // RMS 的平方误差。
    let rms_error = (target.rms - candidate.rms).powi(2);

    // Weighted Sum (Spectral match is usually more important for timbre).
    // 加权和（频谱匹配通常对音色更重要）。
    mel_error + rms_error * 0.5
}

/// Computes energy in Mel filterbands.
/// 计算 Mel 滤波器组中的能量。
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

    // Generate Mel points.
    // 生成 Mel 点。
    let mel_points: Vec<f32> = (0..num_filters + 2)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (num_filters + 1) as f32)
        .collect();

    // Convert to FFT bin indices.
    // 转换为 FFT bin 索引。
    let bin_points: Vec<usize> = mel_points
        .iter()
        .map(|&m| {
            let hz = mel_to_hz(m);
            (hz * fft_size as f32 / sample_rate as f32).floor() as usize
        })
        .collect();

    let mut energies = vec![0.0; num_filters];

    // Triangular Filterbank integration.
    // 三角滤波器组积分。
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

        // Logarithmic compression (decibel-like).
        // 对数压缩（类分贝）。
        energies[i] = (energies[i] + 1e-6).ln();
    }

    energies
}

/// Convert Hz to Mel scale.
/// 将 Hz 转换为 Mel 标度。
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert Mel scale to Hz.
/// 将 Mel 标度转换为 Hz。
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}
