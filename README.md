# Comic Voice-Over System: Tone-Preserved Zero-Shot Vietnamese TTS for Comic Characters

A research-oriented repository for Vietnamese comic narration, speaker-conditioned speech synthesis, and tone-preserved voice cloning built around the VietNeu-140h dataset.

## Abstract

This project studies a complete pipeline for converting comic content into natural Vietnamese speech. The system combines computer vision, text understanding, speaker attribution, and speech synthesis to generate narration for comic characters with stable speaker identity and improved tone accuracy.

The core research objective is to preserve the tonal structure of Vietnamese while transferring a speaker voice from a short reference utterance. The repository therefore focuses on a modular design that supports both signal processing and learning-based synthesis experiments.

## Research Motivation

Vietnamese is a tonal language, where lexical meaning can be strongly affected by tone variation. In text-to-speech systems, this creates a meaningful challenge when transferring speaker identity from a short reference sample. The project is designed to address this problem in the context of comic voice-over generation, where each character should maintain a stable voice identity across different panels and scenes.

The main research directions are:

- tone-aware speech synthesis for Vietnamese,
- zero-shot speaker adaptation from short reference audio,
- comic layout understanding and speech-bubble localization,
- robust evaluation for speech quality and tone correctness.

## System Overview

The pipeline is organized as a multi-stage system:

1. Comic understanding module
   - panel and object detection,
   - speech-bubble localization,
   - OCR-based text extraction,
   - character attribution and speaker assignment.

2. Speech synthesis module
   - text normalization and phoneme preparation,
   - tone-aware conditioning,
   - acoustic modeling and waveform generation.

3. Voice cloning module
   - reference speaker embedding extraction,
   - identity conditioning for zero-shot synthesis,
   - preservation of Vietnamese tone patterns.

4. Evaluation module
   - objective evaluation for voice similarity and tone accuracy,
   - qualitative analysis of generated outputs,
   - ablation studies across model variants.

## Dataset: VietNeu-140h

VietNeu-140h is used as the primary corpus for training and evaluation in this project. The repository is designed to support a clean and reproducible research workflow without storing large audio assets directly in the Git history.

Recommended dataset layout:

```text
data/
├── train/
├── val/
└── test/

metadata/
├── transcripts.csv
├── speaker_info.csv
└── manifest.json
```

The public repository emphasizes:

- stable code organization,
- metadata accessibility,
- configuration transparency,
- lightweight distribution of research artifacts.

Large raw audio, model checkpoints, and generated outputs should be kept locally or in an external dataset storage service.

## Methodology

### 1. Comic Pipeline

The visual pipeline extracts structured information from comic pages, including:

- panel boundaries,
- bubble regions,
- OCR text,
- speaker-related hints and localization features.

### 2. Tone-Preserved TTS

The speech system is built to preserve Vietnamese tonal contrast while generating natural-sounding audio. The project explores conditioning strategies that improve speaker identity control without degrading tone stability.

### 3. Zero-Shot Voice Cloning

A short audio reference is used as the identity signal for voice transfer. This enables speaker cloning without requiring a large amount of per-speaker data for every new target character.

### 4. Evaluation Strategy

The design includes both objective and qualitative evaluation:

- tone consistency,
- speaker similarity,
- intelligibility,
- output stability across transcripts and character styles.

## Repository Structure

```text
Capstone_project/
├── README.md
├── docs/
├── configs/
├── data_pipeline/
├── evaluation/
├── scripts/
├── tone_encoder/
├── voice_cloning/
├── metadata/
└── data/
```

## Installation

This project depends on Python and deep-learning libraries. A basic environment can be created using:

```bash
pip install -r requirements.txt
```

For GPU-enabled runs, additional manual installation may be required depending on the environment and the model family used.

## Quick Start

1. Clone the repository.
2. Prepare the VietNeu-140h data locally in the `data/` directory.
3. Add metadata files under `metadata/`.
4. Review configuration files in `configs/`.
5. Run the required scripts from `scripts/` for preprocessing, training, or inference.

## Experimental Workflow

A typical workflow is:

1. Preprocess the dataset and metadata.
2. Build the comic-to-text pipeline input.
3. Train or evaluate the TTS and voice cloning components.
4. Compare output quality across variants.
5. Archive reports, metrics, and generated sample results.

## Large-File Policy

To keep the public repository practical and maintainable, the project intentionally excludes heavy runtime assets from Git history, including:

- audio files such as `.wav`, `.mp3`, `.flac`
- model checkpoints such as `.pt`, `.pth`, `.safetensors`
- archives such as `.zip`, `.tar`, `.gz`
- generated outputs such as `logs/`, `checkpoints/`, `samples*/`

This repository is designed to preserve source code, configuration, documentation, and metadata in a public and reviewable format.

## Limitations and Future Work

This project is a research codebase and not a production deployment pipeline by default. Current limitations may include:

- dataset availability constraints,
- high computational requirements for training,
- sensitivity to annotation quality,
- need for rigorous evaluation across diverse comic styles.

Future directions may include:

- stronger tone-aware conditioning,
- better speaker disentanglement,
- broader multilingual adaptation,
- deployment-ready inference tools.

## License

This repository is intended for academic, research, and educational use.

## Citation

If you use this project as a reference or build on its methods, please cite the repository and associated work appropriately.
