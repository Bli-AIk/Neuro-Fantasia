//! # Analysis Module / 分析模块
//!
//! This module provides tools to extract psychoacoustic features from audio signals and calculate the similarity (loss) between them.
//! Now updated to use STFT-based Spectrograms for better temporal evolution tracking.

use ndarray::{Array2, ArrayView2, Axis, s};
use rustfft::num_traits::Zero;
use rustfft::{FftPlanner, num_complex::Complex};

/// Container for extracted audio features.
/// 提取的音频特征的容器。
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFeatures {
    /// Spectrogram: Time (Frames) x Frequency (Bins)
    /// Log-Magnitude Spectrogram
    pub spectrogram: Array2<f32>,
}

/// Extracts features from a raw audio buffer using STFT.
/// 使用 STFT 从原始音频缓冲区提取特征。
pub fn extract_features(audio: &[f32], _sample_rate: u32) -> AudioFeatures {
    let fft_size = 2048;
    let hop_size = 512;

    let spectrogram = compute_stft_spectrogram(audio, fft_size, hop_size);

    AudioFeatures { spectrogram }
}

/// Calculates the difference (loss) between target and candidate features.
/// Computes L1 distance between spectrograms.
pub fn calculate_loss(target: &AudioFeatures, candidate: &AudioFeatures) -> f32 {
    let t_spec = &target.spectrogram;
    let c_spec = &candidate.spectrogram;

    // Determine the common shape to compare
    let min_rows = t_spec.nrows().min(c_spec.nrows());
    let cols = t_spec.ncols(); // Should be same (fft_size / 2 + 1)

    if min_rows == 0 {
        return 100.0; // Punishment for empty
    }

    let t_view = t_spec.slice(s![0..min_rows, ..]);
    let c_view = c_spec.slice(s![0..min_rows, ..]);

    // L1 Distance (Mean Absolute Error)
    // Sum of absolute differences / total elements
    let diff_sum: f32 = (&t_view - &c_view).mapv(|x| x.abs()).sum();

    diff_sum / (min_rows * cols) as f32
}

/// Computes the Log-Magnitude Spectrogram using STFT.
fn compute_stft_spectrogram(audio: &[f32], fft_size: usize, hop_size: usize) -> Array2<f32> {
    if audio.len() < fft_size {
        // Handle very short audio by padding
        let mut padded = audio.to_vec();
        padded.resize(fft_size, 0.0);
        return compute_stft_spectrogram(&padded, fft_size, hop_size);
    }

    let num_frames = (audio.len() - fft_size) / hop_size + 1;
    let num_bins = fft_size / 2; // We drop DC and Nyquist usually or keep them. Let's keep fft_size/2 for simplicity.

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Precompute Hann window
    let window: Vec<f32> = (0..fft_size)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32).cos())
        })
        .collect();

    let mut spectrogram_data = Vec::with_capacity(num_frames * num_bins);

    let mut buffer: Vec<Complex<f32>> = vec![Complex::zero(); fft_size];

    for i in 0..num_frames {
        let start = i * hop_size;
        let end = start + fft_size;
        let chunk = &audio[start..end];

        // Apply Window & Copy to buffer
        for (j, &sample) in chunk.iter().enumerate() {
            buffer[j] = Complex::new(sample * window[j], 0.0);
        }

        // FFT
        fft.process(&mut buffer);

        // Compute Log-Magnitude for positive frequencies
        // Skip DC (index 0) if we want, but let's keep it generally.
        // We take first fft_size / 2 bins.
        for j in 0..num_bins {
            let norm = buffer[j].norm();
            // Log scale for better perceptual range: log(1 + x) or log(1e-6 + x)
            // Using log10(norm + epsilon) * 20.0 for dB-like scale
            let val = (norm + 1e-6).log10();
            spectrogram_data.push(val);
        }
    }

    // Create Array2
    // Shape: (num_frames, num_bins)
    Array2::from_shape_vec((num_frames, num_bins), spectrogram_data)
        .unwrap_or_else(|_| Array2::zeros((0, 0)))
}
