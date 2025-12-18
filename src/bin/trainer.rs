use anyhow::{Context, Result};
use clap::Parser;
use neuro_fantasia::analysis::extract_features;
use neuro_fantasia::genetic::GeneticAlgorithm;
use neuro_fantasia::resources::{WAVETABLES, WavetableBank};
use std::fs::File;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the target .wav file to reverse engineer
    #[arg(short, long)]
    target: PathBuf,

    /// Path to folder containing AKWF or other single-cycle waveforms
    #[arg(short, long, default_value = "assets/AKWF")]
    assets: String,

    /// Population size
    #[arg(short, long, default_value_t = 100)]
    pop: usize,

    /// Output JSON file for the best patch
    #[arg(short, long, default_value = "best_patch.json")]
    out: PathBuf,

    /// Target MIDI note (default 60 = C4)
    #[arg(long, default_value_t = 60.0)]
    note: f32,

    /// Number of generations
    #[arg(long, default_value_t = 1000)]
    gens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("--- Neuro-Fantasia: Sound Matcher ---");

    // 1. Load Resources
    println!("Loading wavetables from '{}'...", args.assets);
    match WavetableBank::load_from_directory(&args.assets) {
        Ok(bank) => {
            // Initialize the global singleton
            WAVETABLES
                .set(bank)
                .map_err(|_| anyhow::anyhow!("Failed to set global wavetables"))?;
        }
        Err(e) => {
            eprintln!("Error loading wavetables: {}", e);
            return Err(e);
        }
    }

    // 2. Load Target Audio
    println!("Loading target audio: {:?}", args.target);
    let target_audio = load_target_audio(&args.target)?;
    println!("Target loaded. {} samples.", target_audio.len());

    let target_features = extract_features(&target_audio, 44100);
    println!("Target features extracted.");

    // 3. Initialize GA
    let mut ga = GeneticAlgorithm::new(args.pop, target_features, args.note);

    println!("Starting evolution for {} generations...", args.gens);

    // 4. Evolution Loop
    for i in 0..args.gens {
        ga.evolve();

        if i % 10 == 0 {
            let best = ga.best_individual();
            println!("Gen {}: Loss = {:.5}", i, best.loss);

            // Periodic save
            save_patch(&best.genome, &args.out)?;
        }
    }

    let best = ga.best_individual();
    println!("Final Result: Loss = {:.5}", best.loss);
    save_patch(&best.genome, &args.out)?;
    println!("Saved best patch to {:?}", args.out);

    Ok(())
}

fn load_target_audio(path: &PathBuf) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).context("Failed to open target wav")?;
    let spec = reader.spec();

    // Read and convert to f32 mono
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

    // Mixdown if stereo
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

fn save_patch(genome: &neuro_fantasia::synth::PatchGenome, path: &PathBuf) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, genome)?;
    Ok(())
}
