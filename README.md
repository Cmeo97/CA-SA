# CA-SA: Object-Centric Temporal Consistency via Conditional Autoregressive Inductive Biases

[**Object-Centric Temporal Consistency via Conditional Autoregressive Inductive Biases**](https://arxiv.org/pdf/2410.15728) <br/>
Cristian Meo, Akihiro Nakano, Mircea Lică, Aniket Didolkar, Masahiro Suzuki, Anirudh Goyal, Mengmi Zhang, Justin Dauwels, Yutaka Matsuo, Yoshua Bengio
<br/>


This repository provides a PyTorch implementation for the paper: [Object-Centric Temporal Consistency via Conditional Autoregressive Inductive Biases](https://arxiv.org/abs/2410.15728).
The paper introduces Conditional Autoregressive Slot Attention (CA-SA), a framework designed to enhance the temporal consistency of extracted object-centric representations in video-centric vision tasks. Unsupervised object-centric learning from videos often struggles with maintaining consistent object-to-slot mapping across frames. CA-SA addresses this by leveraging an autoregressive prior network to condition representations on previous timesteps and a novel consistency loss function, termed Objects Permutation Consistency (OPC) Loss, to impose consistency across frames. The proposed method is model-agnostic and demonstrates improved performance on downstream tasks such as video prediction and visual question-answering.

The code contains:

* Training the Conditional Autoregressive Slot Attention (CA-SA) module, including the autoregressive prior and OPC loss.
* Integrating CA-SA with existing object-centric models (e.g., SAVI+SlotFormer, SlotDiffusion+SlotFormer).
* Video prediction task evaluation on CLEVRER and Physion datasets.
* Visual Question Answering (VQA) task evaluation on CLEVRER and Physion datasets.

## Update

* 2024.10.21: Initial code release corresponding to [arXiv:2410.15728v1](https://arxiv.org/abs/2410.15728).
    * Support for CA-SA module training.
    * Support for integration with SlotFormer and SlotDiffusion as base models.
    * Support evaluation on video prediction tasks (CLEVRER, Physion).
    * Support evaluation on VQA tasks (CLEVRER, Physion).

## Installation

Please refer to `docs/install.md` for step-by-step guidance on how to install the necessary packages.

## Experiments

**This codebase is tailored to [Slurm](https://slurm.schedmd.com/documentation.html) GPU clusters with preemption mechanism.**
For the configs, we mainly use NVIDIA V100 and A100 GPUs (see paper Appendix D for details on training times and GPU usage).
Please modify the code accordingly if you are using other hardware settings:

* Please go through `scripts/train.py` (or equivalent training scripts) and change the fields marked by `TODO:`
* Please read the config file for the model you want to train.
    We use DDP with multiple GPUs to accelerate training.
    You can use fewer GPUs to achieve a different memory-speed trade-off.

### Dataset Preparation

Please refer to `docs/data.md` for steps to download and pre-process each dataset (CLEVRER, Physion).

### Reproduce Results

Please see `docs/benchmark.md` for detailed instructions on how to reproduce the results presented in our paper.

