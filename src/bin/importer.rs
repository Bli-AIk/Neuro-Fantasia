use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const TABLE_SIZE: usize = 2048;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "Virtual-Playing-Orchestra3/libs")]
    input_dir: String,

    #[arg(short, long, default_value = "assets/VPO3")]
    output_dir: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input_root = Path::new(&args.input_dir);
    let output_root = Path::new(&args.output_dir);

    println!("Scanning {:?} for wav files...", input_root);

    let entries: Vec<PathBuf> = WalkDir::new(input_root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "wav"))
        .map(|e| e.path().to_path_buf())
        .collect();

    println!("Found {} wav files. Processing...", entries.len());

    entries.par_iter().for_each(|path| {
        if let Err(e) = process_file(path, input_root, output_root) {
            // Be quiet about failures, many files might be percussion or unprocessable
            // eprintln!("Failed to process {:?}: {}", path, e);
        }
    });

    println!("Done!");
    Ok(())
}

fn process_file(path: &Path, input_root: &Path, output_root: &Path) -> Result<()> {
    // 1. Calculate relative path to maintain structure
    let rel_path = path.strip_prefix(input_root).unwrap_or(path);
    let out_path = output_root.join(rel_path);

    // 2. Load Audio
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let max_val = 2u32.pow(spec.bits_per_sample as u32 - 1) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 / max_val)).collect::<Result<Vec<_>, _>>()?
        }
    };

    if samples.len() < 44100 / 20 { // Ignore very short files (< 50ms)
        return Ok(());
    }

    // Convert to Mono if stereo
    let channels = spec.channels as usize;
    let frames = samples.len() / channels;
    let mut mono_samples = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut sum = 0.0;
        for c in 0..channels {
            sum += samples[i * channels + c];
        }
        mono_samples.push(sum / channels as f32);
    }

    // 3. Find Stable Region (Center)
    // We take a window from the middle 30% of the file
    let center_start = (frames as f32 * 0.35) as usize;
    let center_end = (frames as f32 * 0.65) as usize;
    if center_end <= center_start + 1024 {
        return Ok(());
    }
    let analysis_window = &mono_samples[center_start..center_end];

    // 4. Detect Pitch / Period
    if let Some(period_len) = detect_period_autocorr(analysis_window, spec.sample_rate) {
        // 5. Extract One Cycle
        // Find the zero crossing near the start of the window to align phase roughly
        let mut best_start = 0;
        for i in 0..period_len.min(analysis_window.len() - 1) {
            if analysis_window[i] <= 0.0 && analysis_window[i+1] > 0.0 {
                best_start = i;
                break;
            }
        }
        
        // Exact extraction with linear interpolation for fractional period is too complex for this quick script.
        // We will just take the floor(period_len) samples and resample.
        // Better: Try to find a window of size `period_len` that minimizes start/end discontinuity.
        
        let cycle_len = period_len;
        if best_start + cycle_len >= analysis_window.len() {
             return Ok(());
        }
        
        let source_cycle = &analysis_window[best_start..best_start + cycle_len];

        // 6. Resample to TABLE_SIZE (2048)
        let mut table = Vec::with_capacity(TABLE_SIZE);
        for i in 0..TABLE_SIZE {
            let phase = i as f32 / TABLE_SIZE as f32;
            let pos = phase * (cycle_len as f32);
            let idx = pos as usize;
            let frac = pos - idx as f32;
            
            let s0 = source_cycle[idx % cycle_len];
            let s1 = source_cycle[(idx + 1) % cycle_len]; // Wrap logic
            let val = s0 + (s1 - s0) * frac;
            table.push(val);
        }
        
        // 7. Normalize
        let max_val = table.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        if max_val > 0.0 {
            for s in &mut table {
                *s /= max_val;
            }
        }

        // 8. Save
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&out_path, spec)?;
        for s in table {
            writer.write_sample((s * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
        
        // println!("Processed: {:?}", rel_path);
    }

    Ok(())
}

fn detect_period_autocorr(audio: &[f32], sample_rate: u32) -> Option<usize> {
    // Search range: 50Hz to 2000Hz (Typical instrument fundamental range)
    let min_period = (sample_rate as f32 / 2000.0) as usize;
    let max_period = (sample_rate as f32 / 50.0) as usize;

    if audio.len() < max_period * 2 {
        return None;
    }

    let mut best_period = 0;
    let mut max_corr = 0.0;

    // Simple Autocorrelation
    // Optimization: Don't check every single lag, or use FFT for speed. 
    // Given we are offline processing, brute force is acceptable for a few thousand samples.
    // We only check lags in the valid range.

    for lag in min_period..max_period {
        let mut sum = 0.0;
        let mut count = 0;
        // Compare audio[i] with audio[i + lag]
        // Use a subset of points to speed up
        for i in (0..audio.len() - lag).step_by(4) { 
            sum += audio[i] * audio[i + lag];
            count += 1;
        }
        let corr = sum / count as f32;

        if corr > max_corr {
            max_corr = corr;
            best_period = lag;
        }
    }

    // Threshold check: signal must have some periodicity
    // RMS of the signal
    let rms = (audio.iter().take(1000).map(|x| x*x).sum::<f32>() / 1000.0).sqrt();
    if max_corr < rms * rms * 0.5 { // Arbitrary threshold
         return None;
    }

    if best_period > 0 {
        Some(best_period)
    } else {
        None
    }
}
