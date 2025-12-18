//! # Genetic Algorithm Module / 遗传算法模块
//!
//! This module implements the evolutionary logic to search for the best synthesizer patch that matches the target sound.
//! 此模块实现了进化逻辑，以搜索与目标声音最匹配的合成器音色。

use crate::analysis::{AudioFeatures, calculate_loss, extract_features};
use crate::synth::{PatchGenome, render_sample};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rayon::prelude::*;
use std::cmp::Ordering;

/// Represents a single candidate in the population.
/// 代表种群中的一个候选者。
#[derive(Clone, Debug)]
pub struct Individual {
    pub genome: PatchGenome,
    /// Fitness value (Loss). Lower is better.
    /// 适应度值（损失）。越低越好。
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

/// The main Genetic Algorithm controller.
/// 主遗传算法控制器。
pub struct GeneticAlgorithm {
    population: Vec<Individual>,
    /// List of targets: (Midi Note, Extracted Features)
    /// 目标列表：(MIDI 音符, 提取的特征)
    targets: Vec<(f32, AudioFeatures)>,
    pub generation: usize,
}

impl GeneticAlgorithm {
    /// Initializes a new population.
    /// 初始化一个新种群。
    ///
    /// - `seed`: Optional genome to start from (Resume training). / 可选的起始基因组（恢复训练）。
    pub fn new(
        pop_size: usize,
        targets: Vec<(f32, AudioFeatures)>,
        seed: Option<PatchGenome>,
    ) -> Self {
        let mut rng = thread_rng();
        let mut population = Vec::with_capacity(pop_size);

        if let Some(seed_genome) = seed {
            // Seeding strategy:
            // 1. Keep the seed itself (Elitism at start). / 保留种子本身（初始精英）。
            population.push(Individual::new(seed_genome));

            // 2. Generate 30% mutated variants of the seed. / 生成 30% 种子的变异体。
            let mutants_count = pop_size / 3;
            for _ in 0..mutants_count {
                let mut variant = seed_genome;
                mutate(&mut variant, &mut rng);
                population.push(Individual::new(variant));
            }
        }

        // Fill the rest with random genomes.
        // 用随机基因组填充剩余部分。
        while population.len() < pop_size {
            population.push(Individual::new(random_genome(&mut rng)));
        }

        Self {
            population,
            targets,
            generation: 0,
        }
    }

    /// Advances the population by one generation.
    /// 将种群推进一代。
    pub fn evolve(&mut self) {
        // 1. Evaluate Fitness (Parallel)
        // 1. 评估适应度（并行）
        let targets = &self.targets;

        self.population.par_iter_mut().for_each(|ind| {
            let mut total_loss = 0.0;

            // Sum loss across all target notes
            // 累加所有目标音符的损失
            for (note, target_feats) in targets {
                // Render candidate for this specific note
                // 为此特定音符渲染候选者
                let audio = render_sample(&ind.genome, *note, 2.0);
                let feats = extract_features(&audio, 44100);
                total_loss += calculate_loss(target_feats, &feats);
            }

            // Average loss
            // 平均损失
            ind.loss = total_loss / targets.len() as f32;
        });

        // 2. Sort (Ascending Loss)
        // 2. 排序（损失升序）
        self.population
            .sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(Ordering::Equal));

        // Elitism: Keep best 10%
        // 精英策略：保留最好的 10%
        let keep_count = self.population.len() / 10;
        let mut new_pop = self.population[0..keep_count].to_vec();

        // 3. Breed to fill population
        // 3. 繁殖以填充种群
        let mut rng = thread_rng();
        let dist_idx = Uniform::new(0, self.population.len() / 2); // Select parents from top 50% / 从前 50% 选择父代

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

    /// Returns the best individual of the current generation.
    /// 返回当前一代的最佳个体。
    pub fn best_individual(&self) -> &Individual {
        &self.population[0]
    }
}

/// Generates a completely random genome.
/// 生成一个完全随机的基因组。
fn random_genome(rng: &mut ThreadRng) -> PatchGenome {
    PatchGenome {
        osc1_idx: rng.r#gen(),
        osc2_idx: rng.r#gen(),
        detune: rng.r#gen(),
        osc_mix: rng.r#gen(),
        noise_mix: rng.r#gen(),
        noise_attack: rng.gen_range(0.001..0.5),
        noise_decay: rng.gen_range(0.001..0.5),

        // Amp Env
        amp_attack: rng.gen_range(0.001..2.0),
        amp_decay: rng.gen_range(0.001..2.0),
        amp_sustain: rng.r#gen(),
        amp_release: rng.gen_range(0.001..2.0),

        // Filter Env
        filter_attack: rng.gen_range(0.001..2.0),
        filter_decay: rng.gen_range(0.001..2.0),
        filter_sustain: rng.r#gen(),
        filter_release: rng.gen_range(0.001..2.0),
        filter_env_amt: rng.gen_range(-1.0..1.0),

        cutoff: rng.r#gen(),
        resonance: rng.r#gen(),
        filter_type: rng.r#gen(),

        // LFOs
        lfo1_rate: rng.gen_range(0.1..20.0),
        lfo1_amt_cutoff: rng.gen_range(0.0..1000.0),
        lfo1_delay: rng.gen_range(0.0..1.0),
        lfo1_fade: rng.gen_range(0.0..1.0),

        lfo2_rate: rng.gen_range(0.1..20.0),
        lfo2_amt_pitch: rng.gen_range(0.0..50.0),
        lfo2_delay: rng.gen_range(0.0..1.0),
        lfo2_fade: rng.gen_range(0.0..1.0),

        drive: rng.r#gen(),
        saturation: rng.r#gen(),            // Pre-filter drive
        env_curve: rng.gen_range(0.5..4.0), // Exponential envelope

        chorus_mix: rng.gen_range(0.0..0.5),
        reverb_mix: rng.gen_range(0.0..0.5),
        master_vol: 0.8,
    }
}

/// Performs Uniform Crossover.
/// 执行均匀交叉。
fn crossover(g1: &PatchGenome, g2: &PatchGenome, rng: &mut ThreadRng) -> PatchGenome {
    let mut child = *g1;

    // Independent mixing for critical parameters
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
    if rng.gen_bool(0.5) {
        child.noise_attack = g2.noise_attack;
        child.noise_decay = g2.noise_decay;
    }

    // Block swap for Amp ADSR
    if rng.gen_bool(0.5) {
        child.amp_attack = g2.amp_attack;
        child.amp_decay = g2.amp_decay;
        child.amp_sustain = g2.amp_sustain;
        child.amp_release = g2.amp_release;
    }

    // Block swap for Filter ADSR
    if rng.gen_bool(0.5) {
        child.filter_attack = g2.filter_attack;
        child.filter_decay = g2.filter_decay;
        child.filter_sustain = g2.filter_sustain;
        child.filter_release = g2.filter_release;
        child.filter_env_amt = g2.filter_env_amt;
    }

    // Block swap for Filter
    if rng.gen_bool(0.5) {
        child.cutoff = g2.cutoff;
        child.resonance = g2.resonance;
        child.filter_type = g2.filter_type;
    }

    // Block swap for LFO1
    if rng.gen_bool(0.5) {
        child.lfo1_rate = g2.lfo1_rate;
        child.lfo1_amt_cutoff = g2.lfo1_amt_cutoff;
        child.lfo1_delay = g2.lfo1_delay;
        child.lfo1_fade = g2.lfo1_fade;
    }

    // Block swap for LFO2
    if rng.gen_bool(0.5) {
        child.lfo2_rate = g2.lfo2_rate;
        child.lfo2_amt_pitch = g2.lfo2_amt_pitch;
        child.lfo2_delay = g2.lfo2_delay;
        child.lfo2_fade = g2.lfo2_fade;
    }

    // FX & Structure
    if rng.gen_bool(0.5) {
        child.drive = g2.drive;
    }
    if rng.gen_bool(0.5) {
        child.saturation = g2.saturation;
    }
    if rng.gen_bool(0.5) {
        child.env_curve = g2.env_curve;
    }
    if rng.gen_bool(0.5) {
        child.chorus_mix = g2.chorus_mix;
    }
    if rng.gen_bool(0.5) {
        child.reverb_mix = g2.reverb_mix;
    }

    child
}

/// Applies Gaussian Mutation to genes.
/// 将高斯变异应用于基因。
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
    apply(&mut g.noise_attack, 0.001, 0.5);
    apply(&mut g.noise_decay, 0.001, 0.5);

    // Amp
    apply(&mut g.amp_attack, 0.001, 2.0);
    apply(&mut g.amp_decay, 0.001, 2.0);
    apply(&mut g.amp_sustain, 0.0, 1.0);
    apply(&mut g.amp_release, 0.001, 5.0);

    // Filter Env
    apply(&mut g.filter_attack, 0.001, 2.0);
    apply(&mut g.filter_decay, 0.001, 2.0);
    apply(&mut g.filter_sustain, 0.0, 1.0);
    apply(&mut g.filter_release, 0.001, 5.0);
    apply(&mut g.filter_env_amt, -1.0, 1.0);

    // Filter
    apply(&mut g.cutoff, 0.0, 1.0);
    apply(&mut g.resonance, 0.0, 1.0);
    apply(&mut g.filter_type, 0.0, 1.0);

    // LFOs
    apply(&mut g.lfo1_rate, 0.1, 20.0);
    apply(&mut g.lfo1_amt_cutoff, 0.0, 2000.0);
    apply(&mut g.lfo1_delay, 0.0, 2.0);
    apply(&mut g.lfo1_fade, 0.0, 2.0);

    apply(&mut g.lfo2_rate, 0.1, 20.0);
    apply(&mut g.lfo2_amt_pitch, 0.0, 50.0);
    apply(&mut g.lfo2_delay, 0.0, 2.0);
    apply(&mut g.lfo2_fade, 0.0, 2.0);

    // FX & Structure
    apply(&mut g.drive, 0.0, 1.0);
    apply(&mut g.saturation, 0.0, 1.0);
    apply(&mut g.env_curve, 0.5, 5.0);

    apply(&mut g.chorus_mix, 0.0, 1.0);
    apply(&mut g.reverb_mix, 0.0, 1.0);
}
