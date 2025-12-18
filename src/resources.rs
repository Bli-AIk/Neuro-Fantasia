use anyhow::{Context, Result};
use once_cell::sync::OnceCell;
use std::path::Path;
use std::sync::Arc;
use walkdir::WalkDir;

pub const TABLE_SIZE: usize = 2048;

/// Global singleton for the Wavetable Bank
pub static WAVETABLES: OnceCell<WavetableBank> = OnceCell::new();

/// Structure to hold loaded wavetables
#[derive(Debug, Clone)]
pub struct WavetableBank {
    pub tables: Vec<Arc<Vec<f32>>>,
    pub names: Vec<String>,
}

impl WavetableBank {
    pub fn new() -> Self {
        Self {
            tables: Vec::new(),
            names: Vec::new(),
        }
    }

    /// Recursively loads .wav files from a directory, resamples/interpolates to TABLE_SIZE,
    /// normalizes them, and stores them.
    pub fn load_from_directory(path: &str) -> Result<Self> {
        let mut bank = Self::new();
        let walker = WalkDir::new(path).into_iter();

        for entry in walker.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "wav") {
                match load_and_process_wav(path) {
                    Ok(table) => {
                        bank.tables.push(Arc::new(table));
                        bank.names
                            .push(path.file_stem().unwrap().to_string_lossy().to_string());
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load {:?}: {}", path, e);
                    }
                }
            }
        }

        if bank.tables.is_empty() {
            eprintln!(
                "Warning: No wavetables found in '{}'. Generating fallback Sawtooth.",
                path
            );
            let sawtooth: Vec<f32> = (0..TABLE_SIZE)
                .map(|i| 2.0 * (i as f32 / TABLE_SIZE as f32) - 1.0)
                .collect();
            bank.tables.push(Arc::new(sawtooth));
            bank.names.push("fallback_saw".to_string());
        } else {
            println!("Loaded {} wavetables.", bank.tables.len());
        }

        Ok(bank)
    }

    pub fn get(&self, index: usize) -> Arc<Vec<f32>> {
        if self.tables.is_empty() {
            panic!("WavetableBank is empty!");
        }
        self.tables[index % self.tables.len()].clone()
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }
}

fn load_and_process_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).context("Failed to open wav file")?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let max_val = 2u32.pow(spec.bits_per_sample as u32 - 1) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    if samples.is_empty() {
        anyhow::bail!("Empty wav file");
    }

    let mut table = Vec::with_capacity(TABLE_SIZE);
    let source_len = samples.len();

    for i in 0..TABLE_SIZE {
        let phase = i as f32 / TABLE_SIZE as f32;
        let pos = phase * source_len as f32;
        let idx = pos as usize;
        let frac = pos - idx as f32;

        let s0 = samples[idx % source_len];
        let s1 = samples[(idx + 1) % source_len];
        let val = s0 + (s1 - s0) * frac;
        table.push(val);
    }

    let max_peak = table.iter().fold(0.0f32, |max, &v| max.max(v.abs()));
    if max_peak > 0.0 {
        for s in &mut table {
            *s /= max_peak;
        }
    }

    Ok(table)
}
