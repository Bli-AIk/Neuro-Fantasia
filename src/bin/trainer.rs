//! # Trainer CLI / 训练器 CLI
//!
//! Command-line interface to run the Neuro-Fantasia genetic algorithm.
//! 运行 Neuro-Fantasia 遗传算法的命令行接口。

use anyhow::{Context, Result};
use clap::Parser;
use neuro_fantasia::analysis::{AudioFeatures, extract_features};
use neuro_fantasia::genetic::GeneticAlgorithm;
use neuro_fantasia::resources::{WAVETABLES, WavetableBank};
use neuro_fantasia::synth::PatchGenome;
use serde::Deserialize;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Arguments for the CLI.
/// CLI 参数。
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the target folder containing 'dataset.json' and audio files, OR path to a single wav file.
    /// 包含 'dataset.json' 和音频文件的目标文件夹路径，或者单个 wav 文件路径。
    ///
    /// If a folder is provided, it looks for `dataset.json` inside.
    /// 如果提供文件夹，它将在其中查找 `dataset.json`。
    ///
    /// If a single file is provided, use `--note` to specify pitch.
    /// 如果提供单个文件，请使用 `--note` 指定音高。
    #[arg(short, long)]
    target: PathBuf,

    /// Target MIDI note (only used if --target is a single file).
    /// 目标 MIDI 音符（仅当 --target 为单个文件时使用）。
    #[arg(long, default_value_t = 60.0)]
    note: f32,

    /// Path to folder containing AKWF or other single-cycle waveforms.
    /// 包含 AKWF 或其他单周期波形的文件夹路径。
    #[arg(short, long, default_value = "assets/AKWF")]
    assets: String,

    /// Population size (Higher = better search, slower).
    /// 种群大小（越高 = 搜索越好，但越慢）。
    #[arg(short, long, default_value_t = 100)]
    pop: usize,

    /// Output JSON file for the best patch.
    /// 最佳音色的输出 JSON 文件。
    #[arg(short, long, default_value = "best_patch.json")]
    out: PathBuf,

    /// Number of generations to run.
    /// 要运行的代数。
    #[arg(long, default_value_t = 1000)]
    gens: usize,

    /// Resume training from a previous patch JSON file.
    /// 从以前的音色 JSON 文件恢复训练。
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Save interval in generations.
    /// 保存间隔（代数）。
    #[arg(long, default_value_t = 10)]
    save_interval: usize,
}

#[derive(Deserialize, Debug)]
struct Dataset {
    samples: Vec<DatasetEntry>,
}

#[derive(Deserialize, Debug)]
struct DatasetEntry {
    filename: String,
    note: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("--- Neuro-Fantasia: Sound Matcher ---");

    // 1. Load Resources / 1. 加载资源
    println!("Loading wavetables from '{}'...", args.assets);
    match WavetableBank::load_from_directory(&args.assets) {
        Ok(bank) => {
            WAVETABLES
                .set(bank)
                .map_err(|_| anyhow::anyhow!("Failed to set global wavetables"))?;
        }
        Err(e) => {
            eprintln!("Error loading wavetables: {}", e);
            return Err(e);
        }
    }

    // 2. Load Target(s) / 2. 加载目标
    let targets = load_targets(&args)?;
    if targets.is_empty() {
        anyhow::bail!("No targets found. Please check your target path or dataset.json.");
    }
    println!("Loaded {} target sample(s).", targets.len());

    // 3. Load Resume Patch (Optional) / 3. 加载恢复音色（可选）
    let seed_genome = if let Some(path) = &args.resume {
        if path.exists() {
            println!("Resuming from {:?}", path);
            let file = File::open(path)?;
            Some(serde_json::from_reader(file)?)
        } else {
            eprintln!(
                "Warning: Resume file {:?} not found. Starting from scratch.",
                path
            );
            None
        }
    } else {
        None
    };

    // 4. Initialize GA / 4. 初始化遗传算法
    let mut ga = GeneticAlgorithm::new(args.pop, targets, seed_genome);

    println!("Starting evolution for {} generations...", args.gens);

    // 5. Evolution Loop / 5. 进化循环
    for i in 0..args.gens {
        ga.evolve();

        if i % args.save_interval == 0 {
            let best = ga.best_individual();
            println!("Gen {}: Loss = {:.5}", i, best.loss);

            // Periodic save / 定期保存
            save_patch(&best.genome, &args.out)?;
        }
    }

    let best = ga.best_individual();
    println!("Final Result: Loss = {:.5}", best.loss);
    save_patch(&best.genome, &args.out)?;
    println!("Saved best patch to {:?}", args.out);

    Ok(())
}

/// Loads targets based on whether input is a file or directory.
/// 根据输入是文件还是目录加载目标。
fn load_targets(args: &Args) -> Result<Vec<(f32, AudioFeatures)>> {
    let mut results = Vec::new();
    let path = &args.target;

    if path.is_dir() {
        // Look for dataset.json
        let config_path = path.join("dataset.json");
        if !config_path.exists() {
            anyhow::bail!(
                "Directory provided but 'dataset.json' not found in {:?}",
                path
            );
        }

        let file = File::open(&config_path)?;
        let dataset: Dataset = serde_json::from_reader(file)?;

        for entry in dataset.samples {
            let audio_path = path.join(&entry.filename);
            println!(
                "Loading dataset entry: {:?} (Note: {})",
                audio_path, entry.note
            );
            let audio = load_audio_file(&audio_path)?;
            let features = extract_features(&audio, 44100);
            results.push((entry.note, features));
        }
    } else {
        // Single file mode
        println!("Loading single target: {:?} (Note: {})", path, args.note);
        let audio = load_audio_file(path)?;
        let features = extract_features(&audio, 44100);
        results.push((args.note, features));
    }

    Ok(results)
}

/// Helper to load a wav file into a mono float vector.
/// 将 wav 文件加载到单声道浮点向量的辅助函数。
fn load_audio_file(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("Failed to open wav: {:?}", path))?;
    let spec = reader.spec();

    let raw_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let max_val = 2u32.pow(spec.bits_per_sample as u32 - 1) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    // Mixdown to Mono if stereo
    let channels = spec.channels as usize;
    let frames = raw_samples.len() / channels;
    let mut mono = Vec::with_capacity(frames);

    for i in 0..frames {
        let mut sum = 0.0;
        for c in 0..channels {
            sum += raw_samples[i * channels + c];
        }
        mono.push(sum / channels as f32);
    }

    if spec.sample_rate != 44100 {
        eprintln!(
            "Warning: Target sample rate is {}, expected 44100.",
            spec.sample_rate
        );
    }

    Ok(mono)
}

fn save_patch(genome: &PatchGenome, path: &PathBuf) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, genome)?;
    Ok(())
}
