//! # Trainer CLI / 训练器 CLI
//!
//! Command-line interface to run the Neuro-Fantasia genetic algorithm.

use anyhow::{Context, Result};
use clap::Parser;
use neuro_fantasia::analysis::{AudioFeatures, extract_features};
use neuro_fantasia::genetic::GeneticAlgorithm;
use neuro_fantasia::resources::{WAVETABLES, WavetableBank};
use neuro_fantasia::synth::{PatchGenome, render_sample};
use serde::Deserialize;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

/// Arguments for the CLI.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    target: PathBuf,

    #[arg(long, default_value_t = 60.0)]
    note: f32,

    #[arg(short, long, default_value = "assets/AKWF")]
    assets: String,

    #[arg(short, long, default_value_t = 100)]
    pop: usize,

    /// Output directory for results.
    #[arg(short, long, default_value = "output")]
    out_dir: PathBuf,

    #[arg(long, default_value_t = 1000)]
    gens: usize,

    #[arg(long)]
    similarity: Option<f32>,

    #[arg(long)]
    resume: Option<PathBuf>,

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

    println!("-- Neuro-Fantasia: Sound Matcher --");

    // Create output directory
    fs::create_dir_all(&args.out_dir)?;

    // 1. Load Resources
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

    // 2. Load Targets
    let targets = load_targets(&args)?;
    if targets.is_empty() {
        anyhow::bail!("No targets found. Please check your target path or dataset.json.");
    }
    println!("Loaded {} target sample(s).", targets.len());

    // 3. Load Resume Patch
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

    let stop_loss = if let Some(sim_percent) = args.similarity {
        let sim = sim_percent.clamp(0.1, 100.0);
        let thresh = (100.0 / sim) - 1.0;
        println!(
            "Target Similarity: {:.1}% (Stop when Loss <= {:.5})",
            sim, thresh
        );
        Some(thresh)
    } else {
        None
    };

    // 4. Initialize GA
    let mut ga = GeneticAlgorithm::new(args.pop, targets, seed_genome);

    println!("Starting evolution for {} generations...", args.gens);

    // 5. Evolution Loop
    for i in 0..args.gens {
        ga.evolve();

        let best = ga.best_individual();
        let current_sim = 100.0 / (1.0 + best.loss);

        if i % args.save_interval == 0 {
            println!(
                "Gen {}: Loss = {:.5} (Sim: {:.2}%)",
                i, best.loss, current_sim
            );
            save_checkpoint(&args, i, best.loss, &best.genome)?;
        }

        if let Some(thresh) = stop_loss {
            if best.loss <= thresh {
                println!("\nTarget similarity reached!");
                println!(
                    "Gen {}: Loss = {:.5} (Sim: {:.2}%)",
                    i, best.loss, current_sim
                );
                save_checkpoint(&args, i, best.loss, &best.genome)?;
                break;
            }
        }
    }

    let best = ga.best_individual();
    println!(
        "Final Result: Loss = {:.5} (Sim: {:.2}%)",
        best.loss,
        100.0 / (1.0 + best.loss)
    );
    save_checkpoint(&args, args.gens, best.loss, &best.genome)?;

    // Save as "latest.json" for convenience
    let latest_path = args.out_dir.join("latest.json");
    save_patch(&best.genome, &latest_path)?;
    println!("Saved latest patch to {:?}", latest_path);

    Ok(())
}

fn save_checkpoint(args: &Args, generation: usize, loss: f32, genome: &PatchGenome) -> Result<()> {
    let filename = format!("gen_{:04}_loss_{:.4}", generation, loss);

    // Save JSON
    let json_path = args.out_dir.join(format!("{}.json", filename));
    save_patch(genome, &json_path)?;

    // Save WAV
    let wav_path = args.out_dir.join(format!("{}.wav", filename));
    save_wav(genome, args.note, &wav_path)?;

    Ok(())
}

fn save_wav(genome: &PatchGenome, note: f32, path: &PathBuf) -> Result<()> {
    let duration = 2.0; // Standard duration
    let samples = render_sample(genome, note, duration);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;
    let amplitude = i16::MAX as f32;

    for sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * amplitude) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn load_targets(args: &Args) -> Result<Vec<(f32, AudioFeatures)>> {
    let mut results = Vec::new();
    let path = &args.target;

    if path.is_dir() {
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
        println!("Loading single target: {:?} (Note: {})", path, args.note);
        let audio = load_audio_file(path)?;
        let features = extract_features(&audio, 44100);
        results.push((args.note, features));
    }

    Ok(results)
}

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
