# Comic Voice-Over System

A Vietnamese comic voice-over and zero-shot voice cloning research project built around the VietNeu-140h dataset.

## Overview

This repository implements a research-oriented pipeline for Vietnamese comic narration and speech synthesis. The goal is to convert comic text and character context into expressive speech while preserving Vietnamese tone accuracy and speaker identity.

The system is designed to support:

- speech bubble and character region detection,
- text extraction from comic panels,
- speaker attribution for characters,
- tone-aware Vietnamese TTS generation,
- zero-shot voice cloning from short reference audio.

## Project Focus

The core idea of the project is to combine:

- a computer vision pipeline for comic understanding,
- a speech synthesis pipeline for natural Vietnamese voice generation,
- a tone-preserved voice cloning module for speaker identity transfer.

This makes the repo suitable for research, experimentation, and future demo deployment.

## Dataset: VietNeu-140h

VietNeu-140h is the main corpus used for training and experimentation in this project.

The repository is organized to support a public, lightweight GitHub structure while keeping the large raw audio and model artifacts out of the Git history.

Recommended dataset organization:

- `data/train/` for training audio
- `data/val/` for validation audio
- `data/test/` for held-out test audio
- `metadata/` for transcripts, manifest files, and speaker metadata

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

## Quick Start

1. Clone the repository.
2. Prepare the VietNeu-140h data locally under the `data/` folder.
3. Add metadata files under `metadata/`.
4. Review and adjust configuration files in `configs/`.
5. Run scripts from `scripts/` for training or inference.

## Git and Large Files Policy

This repository intentionally ignores large or generated artifacts such as:

- audio files like `.wav`, `.mp3`, `.flac`
- model checkpoints like `.pt`, `.pth`, `.safetensors`
- archives like `.zip`, `.tar`, `.gz`
- temporary outputs such as `logs/`, `checkpoints/`, `samples*/`

This keeps the public GitHub repository clean and suitable for code, configs, and research documentation.

## Research Notes

- The project is structured for reproducibility.
- The large training corpus should be kept locally or stored separately in a dataset storage service.
- GitHub is used for the source code, configuration, and metadata, not for large raw audio releases.

## License

This project is intended for research and educational use.
