# ZIDR

Official implementation of **ZIDR** (**Z**ero-shot **I**mage **D**enoising from Intrinsic **R**edundancy).

---

## Introduction

<p>
<img src='./figs/pipeline.png' align="right" width=500>
</p>

ZIDR is a zero-shot denoising framework for fluorescence microscopy. Instead of relying on external clean-noisy training pairs, it learns directly from a single noisy acquisition through test-time optimization. The core idea is to exploit intrinsic redundancy within the image and to stabilize restoration with a multiscale coherence constraint in the wavelet domain.
ZIDR is designed for photon-limited imaging, where reducing excitation power or exposure time often introduces severe noise and weakens downstream analysis. The method can be applied to both conventional fluorescence images and reconstruction-domain SIM data. In our study, ZIDR suppresses noise and reconstruction artefacts while preserving structural continuity and local contrast, and it also improves analysis tasks such as segmentation in low-dose imaging conditions.

## Repository Structure

```text
ZIDR/
├── dataset/
│   ├── BioSR/
│   ├── FMD/
│   ├── Leica/
│   ├── N2F/
│   ├── Nikon/
│   └── Sunny/
├── MatlabCode/
│   ├── crate_sim_data.m
│   ├── simulation_curve.m
│   ├── simulator_sim.m
│   └── simtools/
├── PythonCode/
│   ├── main_standard_unified.py
│   ├── main_standard_unified_seq.py
│   ├── main_sim_unified.py
│   ├── util_unified.py
│   └── util_unified_seq.py
├── README.md
└── requirements.txt
```

## What Is Included

### `dataset/`
This folder stores the related datasets used in this project. In the current organization, it contains:

- `BioSR`
- `FMD`
- `Leica`
- `N2F`
- `Nikon`
- `Sunny`

### `MatlabCode/`
This folder contains MATLAB scripts mainly for simulation and data preparation.

- `simulation_curve.m`: generates synthetic curve-like structures.
- `simulator_sim.m`: simulates SIM-style observations with configurable noise.
- `crate_sim_data.m`: prepares related multi-view data from raw SIM files.
- `simtools/`: helper functions used by the MATLAB simulation pipeline.

### `PythonCode/`
This folder contains the main Python implementation of ZIDR.

- `main_standard_unified.py`: standard zero-shot denoising for single images or TIFF stacks.
- `main_standard_unified_seq.py`: sequential denoising.
- `main_sim_unified.py`: SIM denoising.
- `util_unified.py`: core model, I/O, and denoising utilities.
- `util_unified_seq.py`: utilities for the sequential version.

## Installation

We recommend creating a clean Python environment and installing PyTorch first according to your CUDA version.

Example:

```bash
conda create -n zidr python=3.10 -y
conda activate zidr

# Install PyTorch first according to your platform / CUDA version
# Example only:
# pip install torch==2.5.1

pip install -r requirements.txt
```

## Requirements

The Python code needs a set of dependencies for the released scripts:

```text
numpy
torch
opencv-python-headless
imageio
tifffile
pytorch-wavelets
pytorch-msssim
natsort
```

If you prefer, `opencv-python-headless` can be replaced with `opencv-python`.

## Usage

The main Python scripts are under `PythonCode/`. Before running them, please edit the input / output paths and hyperparameters in the corresponding script.

### Standard denoising

```bash
cd PythonCode
python main_standard_unified.py
```

### Sequential denoising

```bash
cd PythonCode
python main_standard_unified_seq.py
```

### SIM denoising

```bash
cd PythonCode
python main_sim_unified.py
```

## Notes

- The current entry scripts contain hard-coded `CUDA_VISIBLE_DEVICES` settings. Please modify or remove them before release.
- The standard and sequential pipelines support `png`, `tif`, and `tiff` inputs.
- Since ZIDR uses test-time optimization, GPU execution is strongly recommended for practical runtime.











