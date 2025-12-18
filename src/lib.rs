//! # Neuro-Fantasia Library
//!
//! Core library for the Neuro-Fantasia project, a timbre reverse engineering tool using genetic algorithms.
//! Neuro-Fantasia 项目的核心库，这是一个使用遗传算法进行音色逆向工程的工具。
//!
//! ## Modules / 模块
//!
//! - [`resources`]: Manages audio assets like wavetables.
//!   管理音频资源，如波表。
//! - [`synth`]: The synthesizer engine and genome definition. Decoupled and embeddable.
//!   合成器引擎和基因组定义。解耦且可嵌入。
//! - [`analysis`]: Audio feature extraction and loss calculation.
//!   音频特征提取和损失计算。
//! - [`genetic`]: Genetic algorithm implementation (Population, Evolution).
//!   遗传算法实现（种群，进化）。

pub mod analysis;
pub mod genetic;
pub mod resources;
pub mod synth;
