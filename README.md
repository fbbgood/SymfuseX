# SymfuseX: Bridging Predictive Modeling and Generative Design for Target-Specific Drug Discovery

> Multi-task deep learning framework for drug–target interaction / affinity prediction and target-preference molecule generation.
<img width="552" height="772" alt="image" src="https://github.com/user-attachments/assets/e48e5325-c5db-461a-8a96-9f7c8a2c2598" />


## 🔍 Overview

**SymfuseX** is a unified framework that couples **drug–target prediction** with **target-conditioned molecule generation**.

It supports:

- ✅ **Drug–Target Interaction (DTI) classification**
- ✅ **Drug–Target Affinity (DTA) regression**
- ✅ **Target-preference molecule generation**

The core of SymfuseX is a **Symbiotic Fusion Mechanism**:

- A **symbiosis module** performs **dimension-wise FiLM-style modulation** between drug and protein representations, allowing both sides to co-adapt to each other’s binding preferences.
- A **fusion module** aggregates **raw** and **modulated** features along **multiple interaction paths** with a gating network, emphasizing paths that contribute most to binding probability and affinity.
- The resulting **target-conditioned drug descriptor** `Mod_d` is reused as a **latent guidance vector** to drive fragment-based molecular generation around the target’s local chemical space.

## ⚙️ Installation & Requirements

### Tested environment

OS: Ubuntu 24.04.3 LTS

GPU: NVIDIA A100-SXM4-80GB × 4 (DGX class server)

Driver: 570.195.03

Python: 3.9.24

PyTorch: 2.7.1+cu118 (CUDA 11.8 runtime)

DGL: 1.1.3+cu118

DGL-LifeSci: 0.3.2

RDKit: 2024.09.5

NumPy: 1.26.3, Pandas: 2.3.2, Matplotlib: 3.9.4

importlib-metadata: 8.7.0

## Installation

We recommend creating a fresh conda environment before installing the dependencies.
### 1) Create and activate a new conda environment (Python 3.9)
```bash
conda create -n symfusex python=3.9.24 -y
conda activate symfusex
python -m pip install -U pip
```
### 2) Install RDKit (recommended via conda-forge)
```bash
conda install -y -c conda-forge rdkit=2024.09.5
```
### 3) Install the remaining Python dependencies
```bash
pip install -r requirements.txt
```
### 4) clone the source code of SymfuseX
$ git clone https://github.com/fbbgood/SymfuseX.git

$ cd SymfuseX

## 🚀 Run SymfuseX to Reproduce Results
Regardless of the task type, training can be launched with a single command.

1️⃣ Train DTI classification

**Set MAX_EPOCH to 100 and BATCH_SIZE to 64 in SymfuseX.yaml**, then run:
```bash
python main.py --cfg "configs/SymfuseX.yaml" --task "DTI" --split "${split}" --dataset "${dataset}"
```
${split} can be random or fold; ${dataset} can be bindingdb, human, or biosnap.
For example, to run the human dataset under a random split:
```bash
python main.py --cfg "configs/SymfuseX.yaml" --task "DTI" --split "random" --dataset "human"
```

2️⃣ Train DTA regression

**Set MAX_EPOCH to 1000 and BATCH_SIZE to 256 in SymfuseX.yaml**, then run:
```bash
python main.py --cfg "configs/SymfuseX.yaml" --task "DTA" --split "${split}" --dataset "${dataset}"
```
${split} can be random or cold; ${dataset} can be davis or kiba.
For example, to run the kiba dataset under a random split:
```bash
python main.py --cfg "configs/SymfuseX.yaml" --task "DTA" --split "random" --dataset "davis"
```

3️⃣ Molecular generation

After completing the DTI and DTA tasks, a **.pth checkpoint will be produced for each**. Update the paths to these two checkpoints at the beginning of generate.py to load the trained models. Then, provide the input in /datasets/Generate-sample.csv (typically a known active drug–target pair) and run the following command to start the target-specific molecule generation pipeline:
```bash
python generate.py
```
We also provide pretrained DTI and DTA models, together with 100 example inputs. You can set the **SAMPLE_IDX parameter in generate.py (0–99)** to perform generation for different targets.To facilitate peer review and reduce evaluation time, we release a lightweight version of the generation pipeline. The full version will be made available in a subsequent update, packaged with a more user-friendly web-based UI.

## ✨ Main Features
