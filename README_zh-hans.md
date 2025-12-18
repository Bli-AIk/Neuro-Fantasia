# Neuro-Fantasia

[![license](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-APACHE) <br>
<img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" />

> 当前状态: 🚧 早期开发中

**Neuro-Fantasia** — 基于遗传算法的合成器音色逆向工程工具。

| English | Simplified Chinese |
|-----------------|
| [English](./README.md) | 简体中文 |

## 简介

`Neuro-Fantasia` 是一个自动化的声音设计工具，利用进化算法来重建目标声音。
它解决了手动调节合成器参数的难题，用户只需提供一个或一组音频样本，即可获得模仿该音色的合成器预设。

使用 `Neuro-Fantasia`，你只需要提供一个 WAV 文件或包含音符的数据集。系统将在数千代的进化过程中，自动调整参数以匹配声音的频谱和时域特征。
未来，它可能支持直接导出到 Bevy 游戏引擎或 VST 插件中。

## 功能

* **遗传进化**: 使用锦标赛选择、交叉和变异来寻找最佳参数。
* **高级减法合成引擎**:
    * **动态瞬态层 (Transient)**: 可精确塑形的噪音包络 (Attack/Decay)，用于模拟真实的乐器起音（如吹气声、擦弦声）。
    * **双包络系统**: 独立的 ADSR 包络分别控制音量 (Amp) 和音色 (Filter)。
    * **双 LFO 系统**: 专用的 LFO 分别用于颤音 (Pitch) 和 哇音/PWM (Filter)。
    * **FX 效果引擎**: 内置立体声合唱 (Chorus) 和混响 (Reverb)，为声音增加厚度与空间感。
    * **智能波表排序**: 自动按亮度对数千个波形进行排序，使进化过程更加平滑。
* **多目标训练**: 支持对同一乐器的多个音高进行训练，以提高准确性。
* **断点续练**: 随时停止并恢复训练。
* **相似度阈值**: 当达到预期的相似度百分比时自动停止。

## 如何使用

1. **安装 Rust** (如果尚未安装):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **克隆仓库**:
   ```bash
   git clone https://github.com/your_username/neuro-fantasia.git
   cd neuro-fantasia
   # 初始化波表子模块
   git submodule update --init --recursive
   ```

3. **构建**:
   ```bash
   cargo build --release
   ```

4. **准备数据**:

   **选项 A: 单个文件**
   准备一个 `.wav` 文件 (例如 `target.wav`)。

   **选项 B: 数据集 (推荐)**
   创建一个文件夹 (例如 `my_instrument/`)，放入音频文件和 `dataset.json`：
   ```json
   {
     "samples": [
       { "filename": "C4.wav", "note": 60.0 },
       { "filename": "G4.wav", "note": 67.0 }
     ]
   }
   ```

5. **运行训练**:

   **基础单文件训练:**
   ```bash
   cargo run --release --bin trainer -- --target target.wav --note 60 --out-dir output
   ```

   **数据集训练 (高性能模式):**
   ```bash
   cargo run --release --bin trainer -- \
     --target my_instrument/ \
     --out-dir output/ \
     --gens 10000 \
     --save-interval 50
   ```

   **达到 95% 相似度时停止:**
   ```bash
   cargo run --release --bin trainer -- \
     --target my_instrument/ \
     --out-dir output/ \
     --similarity 95.0
   ```

   **恢复训练 (读档):**
   ```bash
   cargo run --release --bin trainer -- \
     --target my_instrument/ \
     --out-dir output/ \
     --resume output/latest.json
   ```

   *训练器会自动在输出目录中保存 `gen_XXXX_loss_YYYY.json` 和 `gen_XXXX_loss_YYYY.wav` 文件。*

## 如何构建

### 前置条件

* Rust 1.75 或更高版本 (需要 Edition 2024 支持)

### 构建步骤

1. **克隆仓库**:
   ```bash
   git clone https://github.com/your_username/neuro-fantasia.git
   cd neuro-fantasia
   ```

2. **构建项目**:
   ```bash
   cargo build --release
   ```

## 依赖

本项目主要使用以下 Crates:

| Crate                                       | Description |
|---------------------------------------------|-------------|
| [fundsp](https://crates.io/crates/fundsp)   | 音频 DSP 库    |
| [rustfft](https://crates.io/crates/rustfft) | 用于频谱分析的 FFT |
| [rayon](https://crates.io/crates/rayon)     | 并行计算适应度     |
| [clap](https://crates.io/crates/clap)       | CLI 参数解析    |
| [serde](https://crates.io/crates/serde)     | JSON 序列化    |

## 贡献

欢迎贡献！
无论是修复 Bug、添加新功能还是改进文档：

* 提交 **Issue** 或 **Pull Request**。
* 分享想法并讨论设计或架构。

## 许可证

本项目基于以下任一许可证授权：

* Apache License, Version 2.0
* MIT license

由你选择。
