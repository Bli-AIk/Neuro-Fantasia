//! # Trainer CLI / 训练器 CLI
//!
//! Command-line interface to run the Neuro-Fantasia genetic algorithm.
//! 运行 Neuro-Fantasia 遗传算法的命令行接口。
//!
//! ## Workflow / 工作流程
//! 1. **Load Resources**: Loads wavetables from the assets folder. / **加载资源**: 从资产文件夹加载波表。
//! 2. **Load Target**: Reads the target `.wav` file and extracts its features. / **加载目标**: 读取目标 `.wav` 文件并提取其特征。
//! 3. **Initialize Population**: Creates a random starting population. / **初始化种群**: 创建一个随机的初始种群。
//! 4. **Evolve**: Runs the evolution loop, periodically saving the best genome to JSON. / **进化**: 运行进化循环，定期将最佳基因组保存为 JSON。

use anyhow::{Context, Result};
use clap::Parser;
use neuro_fantasia::analysis::extract_features;
use neuro_fantasia::genetic::GeneticAlgorithm;
use neuro_fantasia::resources::{WAVETABLES, WavetableBank};
use std::fs::File;
use std::path::PathBuf;

/// Arguments for the CLI.
/// CLI 参数。
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the target .wav file to reverse engineer.
    /// 要逆向工程的目标 .wav 文件路径。
    #[arg(short, long)]
    target: PathBuf,

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

    /// Target MIDI note (default 60 = C4). Should match the target audio pitch.
    /// 目标 MIDI 音符（默认为 60 = C4）。应与目标音频音高匹配。
    #[arg(long, default_value_t = 60.0)]
    note: f32,

    /// Number of generations to run.
    /// 要运行的代数。
    #[arg(long, default_value_t = 1000)]
    gens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("--- Neuro-Fantasia: Sound Matcher ---");

    // 1. Load Resources / 1. 加载资源
    println!("Loading wavetables from '{}'...", args.assets);
    match WavetableBank::load_from_directory(&args.assets) {
        Ok(bank) => {
            // Initialize the global singleton.
            // 初始化全局单例。
            WAVETABLES
                .set(bank)
                .map_err(|_| anyhow::anyhow!("Failed to set global wavetables"))?;
        }
        Err(e) => {
            eprintln!("Error loading wavetables: {}", e);
            return Err(e);
        }
    }

    // 2. Load Target Audio / 2. 加载目标音频
    println!("Loading target audio: {:?}", args.target);
    let target_audio = load_target_audio(&args.target)?;
    println!("Target loaded. {} samples.", target_audio.len());

    // Extract features once.
    // 提取一次特征。
    let target_features = extract_features(&target_audio, 44100);
    println!("Target features extracted.");

    // 3. Initialize GA / 3. 初始化遗传算法
    let mut ga = GeneticAlgorithm::new(args.pop, target_features, args.note);

    println!("Starting evolution for {} generations...", args.gens);

    // 4. Evolution Loop / 4. 进化循环
    for i in 0..args.gens {
        ga.evolve();

        if i % 10 == 0 {
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

/// Helper to load a wav file into a mono float vector.
/// 将 wav 文件加载到单声道浮点向量的辅助函数。
fn load_target_audio(path: &PathBuf) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).context("Failed to open target wav")?;
    let spec = reader.spec();

    // Read and convert to f32.
    // 读取并转换为 f32。
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

    // Mixdown to Mono if stereo.
    // 如果是立体声，则混合为单声道。
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
            "Warning: Target sample rate is {}, expected 44100. Analysis might be skewed.",
            spec.sample_rate
        );
    }

    Ok(mono)
}

/// Helper to save the genome as JSON.
/// 将基因组保存为 JSON 的辅助函数。
fn save_patch(genome: &neuro_fantasia::synth::PatchGenome, path: &PathBuf) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, genome)?;
    Ok(())
}
