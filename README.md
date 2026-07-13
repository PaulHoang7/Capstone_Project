# Comic Voice-Over System

Vietnamese TTS with tone-preserved zero-shot voice cloning for comic characters.

This repository is organized for the VietNeu-140h dataset and the Voice Clone / TTS research pipeline used in the project.

## Overview

The project focuses on building an end-to-end Vietnamese comic voice-over system that can:

- detect comic speech bubbles and character regions,
- extract source text from panels,
- assign voice to characters,
- generate expressive Vietnamese speech with tone preservation,
- clone a speaker voice from short reference audio in a zero-shot setting.

## Dataset: VietNeu-140h

The repository is prepared to work with a Vietnamese speech dataset collection named VietNeu-140h.

Recommended dataset conventions:

- audio files stored in `data/` or split into `train/`, `val/`, `test/`
- transcript metadata stored in `metadata/`
- speaker identity metadata stored separately when available
- large audio assets and checkpoints should stay out of Git history

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

## How to Use

1. Place the dataset under `data/`.
2. Prepare transcript metadata under `metadata/`.
3. Review the config files in `configs/`.
4. Run the training or inference scripts in `scripts/`.

## Git / Large Files

Large audio files, checkpoints, and generated artifacts are intentionally ignored through the project `.gitignore`.

This keeps the repo lightweight while preserving the source code and dataset metadata structure.

## Notes

- This repo is structured to support research and reproducibility.
- The actual large training corpus should be stored locally or in a dataset hosting service.
- GitHub is used to track code, configs, docs, and metadata rather than raw audio archives.

## License

This repository is intended for research and educational use.
