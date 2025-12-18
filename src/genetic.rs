use crate::analysis::{AudioFeatures, calculate_loss, extract_features};
use crate::synth::{PatchGenome, render_sample};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct Individual {
    pub genome: PatchGenome,
    pub loss: f32,
}

impl Individual {
    pub fn new(genome: PatchGenome) -> Self {
        Self {
            genome,
            loss: f32::MAX,
        }
    }
}

pub struct GeneticAlgorithm {
    population: Vec<Individual>,
    target_features: AudioFeatures,
    note_freq: f32, // Midi note
    pub generation: usize,
}

impl GeneticAlgorithm {
    pub fn new(pop_size: usize, target_features: AudioFeatures, note_freq: f32) -> Self {
        let mut rng = thread_rng();
        let population: Vec<Individual> = (0..pop_size)
            .map(|_| Individual::new(random_genome(&mut rng)))
            .collect();

        Self {
            population,
            target_features,
            note_freq,
            generation: 0,
        }
    }

    pub fn evolve(&mut self) {
        // 1. Evaluate Fitness (Parallel)
        let note = self.note_freq;
        let target = &self.target_features;

        self.population.par_iter_mut().for_each(|ind| {
            // Render candidate
            let audio = render_sample(&ind.genome, note, 2.0); // 2 seconds render
            let feats = extract_features(&audio, 44100);
            ind.loss = calculate_loss(target, &feats);
        });

        // 2. Sort
        self.population
            .sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(Ordering::Equal));

        // Elitism: Keep best 10%
        let keep_count = self.population.len() / 10;
        let mut new_pop = self.population[0..keep_count].to_vec();

        // 3. Breed
        let mut rng = thread_rng();
        let dist_idx = Uniform::new(0, self.population.len() / 2); // Select from top 50%

        while new_pop.len() < self.population.len() {
            let p1 = &self.population[dist_idx.sample(&mut rng)];
            let p2 = &self.population[dist_idx.sample(&mut rng)];

            let mut child_genome = crossover(&p1.genome, &p2.genome, &mut rng);
            mutate(&mut child_genome, &mut rng);

            new_pop.push(Individual::new(child_genome));
        }

        self.population = new_pop;
        self.generation += 1;
    }

    pub fn best_individual(&self) -> &Individual {
        &self.population[0]
    }
}

fn random_genome(rng: &mut ThreadRng) -> PatchGenome {
    PatchGenome {
        osc1_idx: rng.r#gen(),
        osc2_idx: rng.r#gen(),
        detune: rng.r#gen(),
        osc_mix: rng.r#gen(),
        noise_mix: rng.r#gen(),
        attack: rng.gen_range(0.001..1.0),
        decay: rng.gen_range(0.001..1.0),
        sustain: rng.r#gen(),
        release: rng.gen_range(0.001..2.0),
        cutoff: rng.r#gen(),
        resonance: rng.r#gen(),
        filter_type: rng.r#gen(),
        filter_env_amt: rng.gen_range(-1.0..1.0),
        lfo_rate: rng.gen_range(0.1..15.0),
        lfo_amt_pitch: rng.gen_range(0.0..10.0), // Hz deviation approx
        lfo_amt_cutoff: rng.gen_range(0.0..1000.0),
        drive: rng.r#gen(),
        master_vol: 0.8,
    }
}

fn crossover(g1: &PatchGenome, g2: &PatchGenome, rng: &mut ThreadRng) -> PatchGenome {
    // Uniform Crossover
    let mut child = *g1;
    if rng.gen_bool(0.5) {
        child.osc1_idx = g2.osc1_idx;
    }
    if rng.gen_bool(0.5) {
        child.osc2_idx = g2.osc2_idx;
    }
    if rng.gen_bool(0.5) {
        child.detune = g2.detune;
    }
    if rng.gen_bool(0.5) {
        child.osc_mix = g2.osc_mix;
    }
    if rng.gen_bool(0.5) {
        child.noise_mix = g2.noise_mix;
    }

    // ADSR block swap
    if rng.gen_bool(0.5) {
        child.attack = g2.attack;
        child.decay = g2.decay;
        child.sustain = g2.sustain;
        child.release = g2.release;
    }

    // Filter block swap
    if rng.gen_bool(0.5) {
        child.cutoff = g2.cutoff;
        child.resonance = g2.resonance;
        child.filter_type = g2.filter_type;
        child.filter_env_amt = g2.filter_env_amt;
    }

    // LFO swap
    if rng.gen_bool(0.5) {
        child.lfo_rate = g2.lfo_rate;
        child.lfo_amt_pitch = g2.lfo_amt_pitch;
        child.lfo_amt_cutoff = g2.lfo_amt_cutoff;
    }

    if rng.gen_bool(0.5) {
        child.drive = g2.drive;
    }

    child
}

fn mutate(g: &mut PatchGenome, rng: &mut ThreadRng) {
    let mut_prob = 0.1;
    let normal = Normal::new(0.0, 0.1).unwrap();

    let mut apply = |val: &mut f32, min: f32, max: f32| {
        if rng.gen_bool(mut_prob) {
            *val += normal.sample(rng);
            *val = val.clamp(min, max);
        }
    };

    apply(&mut g.osc1_idx, 0.0, 1.0);
    apply(&mut g.osc2_idx, 0.0, 1.0);
    apply(&mut g.detune, 0.0, 1.0);
    apply(&mut g.osc_mix, 0.0, 1.0);
    apply(&mut g.noise_mix, 0.0, 1.0);

    apply(&mut g.attack, 0.001, 2.0);
    apply(&mut g.decay, 0.001, 2.0);
    apply(&mut g.sustain, 0.0, 1.0);
    apply(&mut g.release, 0.001, 5.0);

    apply(&mut g.cutoff, 0.0, 1.0);
    apply(&mut g.resonance, 0.0, 1.0);
    apply(&mut g.filter_type, 0.0, 1.0);
    apply(&mut g.filter_env_amt, -1.0, 1.0);

    apply(&mut g.lfo_rate, 0.1, 20.0);
    apply(&mut g.lfo_amt_pitch, 0.0, 50.0);
    apply(&mut g.lfo_amt_cutoff, 0.0, 2000.0);

    apply(&mut g.drive, 0.0, 1.0);
}
