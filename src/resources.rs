//! # Resources Module / 资源模块
//!
//! This module handles the loading and management of audio assets, specifically wavetables for the synthesizer.
//! 此模块处理音频资源的加载和管理，特别是用于合成器的波表。
//!
//! Key responsibilities:
//! 主要职责：
//! 1. Recursively scan directories for `.wav` files. / 递归扫描目录中的 `.wav` 文件。
//! 2. Resample single-cycle waveforms to a standardized size (2048 samples). / 将单周期波形重采样为标准大小（2048 样本）。
//! 3. Normalize audio data. / 归一化音频数据。
//! 4. Provide a thread-safe global access point via `once_cell`. / 通过 `once_cell` 提供线程安全的全局访问点。

use anyhow::{Context, Result};
use once_cell::sync::OnceCell;
use std::path::Path;
use std::sync::Arc;
use walkdir::WalkDir;

/// Standard wavetable size (power of 2 is preferred for easy phase wrapping).
/// 标准波表大小（首选 2 的幂，以便于相位包裹）。
pub const TABLE_SIZE: usize = 2048;

/// Global singleton for the Wavetable Bank.
/// 波表库的全局单例。
///
/// Accessed by the synthesis engine to retrieve waveform data without reloading files.
/// 合成引擎访问此单例以检索波形数据，而无需重新加载文件。
pub static WAVETABLES: OnceCell<WavetableBank> = OnceCell::new();

/// Structure to hold loaded wavetables.
/// 用于保存已加载波表的结构体。
#[derive(Debug, Clone)]
pub struct WavetableBank {
    /// List of wavetable data (each is a normalized `Vec<f32>`).
    /// 波表数据列表（每个都是归一化的 `Vec<f32>`）。
    pub tables: Vec<Arc<Vec<f32>>>,
    /// Corresponding filenames for debugging/UI.
    /// 对应的文件名，用于调试/UI。
    pub names: Vec<String>,
}

impl WavetableBank {
    pub fn new() -> Self {
        Self {
            tables: Vec::new(),
            names: Vec::new(),
        }
    }

    /// Recursively loads .wav files from a directory.
    /// 递归地从目录加载 .wav 文件。
    ///
    /// The loading process includes:
    /// 加载过程包括：
    /// - Finding all `.wav` files. / 查找所有 `.wav` 文件。
    /// - Loading samples using `hound`. / 使用 `hound` 加载样本。
    /// - Interpolating to `TABLE_SIZE`. / 插值到 `TABLE_SIZE`。
    /// - Normalizing amplitude. / 归一化振幅。
    ///
    /// If the directory is empty or missing, a fallback Sawtooth wave is generated.
    /// 如果目录为空或丢失，将生成一个后备锯齿波。
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
                        eprintln!("Warning: Failed to load {:?}: {}", path, e); // 警告：加载失败
                    }
                }
            }
        }

        if bank.tables.is_empty() {
            eprintln!(
                "Warning: No wavetables found in '{}'. Generating fallback Sawtooth.",
                path
            );
            // 警告：未找到波表。生成后备锯齿波。
            let sawtooth: Vec<f32> = (0..TABLE_SIZE)
                .map(|i| 2.0 * (i as f32 / TABLE_SIZE as f32) - 1.0)
                .collect();
            bank.tables.push(Arc::new(sawtooth));
            bank.names.push("fallback_saw".to_string());
        } else {
            // Sort wavetables by brightness to make the search space smoother
            // 按亮度对波表进行排序，使搜索空间更平滑
            let mut zipped: Vec<(f32, Arc<Vec<f32>>, String)> = bank
                .tables
                .iter()
                .zip(bank.names.iter())
                .map(|(t, n)| {
                    let brightness = calculate_brightness(t);
                    (brightness, t.clone(), n.clone())
                })
                .collect();

            zipped.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            bank.tables = zipped.iter().map(|x| x.1.clone()).collect();
            bank.names = zipped.iter().map(|x| x.2.clone()).collect();

            println!("Loaded and sorted {} wavetables.", bank.tables.len()); // 已加载并排序 X 个波表
        }

        Ok(bank)
    }

    /// Retrieve a wavetable by index (wraps around if out of bounds).
    /// 按索引检索波表（如果超出范围则循环）。
    pub fn get(&self, index: usize) -> Arc<Vec<f32>> {
        if self.tables.is_empty() {
            panic!("WavetableBank is empty!"); // 波表库为空！
        }
        self.tables[index % self.tables.len()].clone()
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }
}

fn calculate_brightness(table: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 1..table.len() {
        sum += (table[i] - table[i - 1]).abs();
    }
    sum
}

/// Helper function to load, resample, and normalize a single WAV file.
/// 辅助函数：加载、重采样和归一化单个 WAV 文件。
fn load_and_process_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).context("Failed to open wav file")?; // 打开wav文件失败
    let spec = reader.spec();

    // Read samples as f32. Handle integer to float conversion if necessary.
    // 将样本读取为 f32。如有必要，处理整数到浮点数的转换。
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
        anyhow::bail!("Empty wav file"); // 空 wav 文件
    }

    // Linear Interpolation to fit TABLE_SIZE.
    // 线性插值以适应 TABLE_SIZE。
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

    // Normalization (Peak).
    // 归一化（峰值）。
    let max_peak = table.iter().fold(0.0f32, |max, &v| max.max(v.abs()));
    if max_peak > 0.0 {
        for s in &mut table {
            *s /= max_peak;
        }
    }

    Ok(table)
}
