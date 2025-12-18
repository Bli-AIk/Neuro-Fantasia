use rustfft::{FftPlanner, num_complex::Complex};

#[derive(Debug, Clone, PartialEq)]
pub struct AudioFeatures {
    pub mel_bands: Vec<f32>, // 40 bands
    pub rms: f32,
}

pub fn extract_features(audio: &[f32], sample_rate: u32) -> AudioFeatures {
    let fft_size = 2048;

    let center = audio.len() / 2;
    let start = if center > fft_size / 2 {
        center - fft_size / 2
    } else {
        0
    };
    let end = (start + fft_size).min(audio.len());

    let mut input: Vec<Complex<f32>> = audio[start..end]
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

    let magnitudes: Vec<f32> = input
        .iter()
        .take(fft_size / 2) // Nyquist
        .map(|c| c.norm())
        .collect();

    let mel_bands = compute_mel_energies(&magnitudes, sample_rate, fft_size, 40);

    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();

    AudioFeatures { mel_bands, rms }
}

pub fn calculate_loss(target: &AudioFeatures, candidate: &AudioFeatures) -> f32 {
    let mut mel_error = 0.0;
    for (t, c) in target.mel_bands.iter().zip(&candidate.mel_bands) {
        mel_error += (t - c).powi(2);
    }
    mel_error /= target.mel_bands.len() as f32;

    let rms_error = (target.rms - candidate.rms).powi(2);

    mel_error + rms_error * 0.5
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
