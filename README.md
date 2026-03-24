# End-to-End Training for Unified Tokenization and Latent Denoising
### Let's UNITE! Single-stage training, Unified model for Tokenization & Generation <br/><br/>[Paper](https://arxiv.org/abs/2603.22283) | [Project Page](https://xingjianbai.com/unite-tokenization-generation/)

This is the code repository of the paper:
> [End-to-End Training for Unified Tokenization and Latent Denoising](https://arxiv.org/abs/2603.22283) (arXiv 2026)  
> [Shivam Duggal*](https://shivamduggal4.github.io/), [Xingjian Bai*](https://xingjianbai.com/), [Zongze Wu](https://betterze.github.io/website/), [Richard Zhang](https://richzhang.github.io/), [Eli Shechtman](https://scholar.google.com/citations?user=B_FTboQAAAAJ&hl=zh-CN), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), [Phillip Isola](https://web.mit.edu/phillipi/), [William T. Freeman](https://billf.mit.edu/)  
> MIT, Adobe  


## Table of Content
[Abstract](#Abstract)  
[Approach Overview](#Overview)  
[Setup](#Setup)  
[Datasets](#Datasets)  
[Training](#Training)  
[Pretrained Checkpoints](#PretrainedCheckpoints)  
[Evaluation](#Evaluation)  
[Citation](#Citation)  

<a name="Abstract"></a>
## Abstract

<div style="text-align: justify;">
Latent diffusion models (LDMs) enable high-fidelity synthesis by operating in learned latent spaces. However, training state-of-the-art LDMs requires complex staging: a tokenizer must be trained first, before the diffusion model can be
trained in the frozen latent space. We propose UNITE – an autoencoder architecture for unified tokenization and latent diffusion. UNITE consists of a Generative Encoder that serves as both image tokenizer and latent generator via weight sharing. Our key insight is that tokenization and generation can be viewed as the same latent inference problem under different conditioning regimes: tokenization infers latents from fully observed images, whereas generation infers them from noise together with text or class conditioning. Motivated by this, we introduce a single-stage training procedure that jointly optimizes both tasks via two forward passes through the same Generative Encoder. The shared parameters enable gradients
to jointly shape the latent space, encouraging a “common latent language”. Across image and molecule modalities, UNITE achieves near state of the art performance without adversarial losses or pretrained encoders (e.g., DINO), reaching FID
2.12 and 1.73 for Base and Large models on ImageNet 256 × 256. We further analyze the Generative Encoder through the lenses of representation alignment and compression. These results show that single stage joint training of tokenization & generation from scratch is feasible.
</div>

<a name="Approach Overview"></a>
## Approach Overview
![](./assets/overview.png)

<a name="Setup"></a>
## Setup

```bash
# Clone and install dependencies
mamba create -n unite python=3.10 -y
mamba activate unite
pip install uv

# Install PyTorch 2.10.0 with CUDA 12.8 or higher # or your own cuda version
uv pip install -r requirements.txt
```

Required packages: `torch>=2.1`, `torchvision`, `einops`, `torchdiffeq`, `lpips`, `pyyaml`, `clean-fid`.  
Main requirement is to have a torch version which supports Muon optimizer.

**For FID evaluation on Imagenet 256 x 256:**
```
wget https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth  
wget https://raw.githubusercontent.com/LTH14/JiT/main/fid_stats/jit_in256_stats.npz  
export INCEPTION_WEIGHTS=/path/to/inception_weights  
export IN256_FID_STATS=/path/to/in256_fid_stats  
```

<a name="Datasets"></a>
## Datasets

Training uses the [ImageNet-1K](https://image-net.org/) dataset (ILSVRC2012). Set the `DATA_PATH` environment variable to the training split:

```bash
export DATA_PATH=/path/to/imagenet/train
```

FID evaluation uses precomputed ImageNet statistics from `clean-fid`.  
For our results on the molecules dataset, please refer to the paper.

<a name="Training"></a>
## Training

![](./assets/architecture.png)

Single-node (with 8 GPUs) training procedure on Imagenet 256x256:

```bash
# Using the launch script (auto-detects GPUs and data path)
bash run_scripts/train.sh

# Or directly with torchrun
torchrun --nproc_per_node=8 main_train.py \
    --config configs/imagenet.yaml \
    --data-path $DATA_PATH \
    --experiment-name lets-unite
```

To resume from a checkpoint, additionally provide the path of the checkpoint with argument: `--ckpt`.

Key config fields (see `configs/imagenet.yaml`):

| Field | Default | Description |
|:------|:-------:|:------------|
| `total_batch_size` | 1024 | Effective batch size across all GPUs and accumulation steps |
| `grad_accum_steps` | 2 | Gradient accumulation steps (batch/GPU = total / world_size / accum) |
| `flow_steps_per_recon` | 3 | Flow matching steps per reconstruction forward pass |
| `flow_mini_batch` | 4 | Chunk size for flow steps (controls peak memory) |
`torch_compile_decoder` | true | torch.compile the decoder for faster training |

**Evaluation during training:**  FID is evaluated automatically at the end of each `fid_epoch` interval (default: every 40 epochs). The evaluation uses adaptive CFG scale search: starting from the current best scale, it tests neighboring scales and updates the best. At `fid_sweep_epoch` intervals (default: 120 epochs), a more comprehensive sweep is performed across multiple CFG intervals and normalization orders.


### Reproducing Paper Results
To reproduce the paper results for UNITE-B (with 3 flow mini batches per each reconstruction step)on a single-node on ImageNet-1K 256×256, use the  config
[`configs/imagenet.yaml`](configs/imagenet.yaml).


| | |
|:--|:--|
| **Architecture** | Base encoder (130.6M params) + Base decoder (86.2M params) |
| **Total params** | 217.6M (all trainable) |
| **Hardware** | 1 node × 8 NVIDIA H200 (140 GB each) |
| **Effective batch size** | 1024 (64 per GPU × 2 gradient accumulation steps) |
| **Precision** | BF16 mixed precision |
| **Optimizer** | Muon |
| **Training speed** | ~26 min/epoch |


<br/>

UNITE has an advesarial nature due to joint optimization of reconstruction and denoising losses. See paper Sec. 3.3 for more details. The training curves should be similar to the following graph: 

![](./assets/training_dynamics.png)
<br/>


FID and Inception Score (IS) are computed with adaptive CFG scale search — starting from the current best scale, neighboring scales are evaluated and the best is kept. All evaluations use the EMA model.



![](./assets/fid_curve.png)



<a name="PretrainedCheckpoints"></a>
## Pretrained Checkpoints

| Model | Encoder | Decoder | Total Params | Epochs | FID-50K | Checkpoint |
|:------|:-------:|:-------:|:------------:|:------:|:--:|:----------:|
| United-B | Base | Base | 217.6M | 240 | 2.12 | [Download Link](https://www.dropbox.com/scl/fi/6vbzwqiu9rqnmwm4cj5my/UNITE-B.pt?rlkey=yrqoyiw1zymqe6pn8rf0ss1d6&st=xt8hmz46&dl=0) |
| United-L | Large | Base | 589.0M | 120 | 1.73 | [Download Link](https://www.dropbox.com/scl/fi/mvqbhq3oubog4gtswnthw/UNITE-L.pt?rlkey=ekn3m0b5wb26512dc901rj9w7&st=x91h020h&dl=0) |

Checkpoint includes model weights (encoder + decoder), EMA state, optimizer state, and scheduler state for seamless resumption. Both models were trained with 14 flow iterations per single reconstruction step, so `flow_steps_per_recon = 14`


<a name="Evaluation"></a>
## Evaluation

We will share the inference evaluation for generation and reconstruction FID and IS soon. Till then enjoy the following results from XL model.

![](./assets/generation_visualization.png)
<br/>

<a name="Citation"></a>
## Citation

```bibtex
@article{duggal2026unite,
  title={End-to-End Training for Unified Tokenization and Latent Denoising},
  author={Shivam Duggal and Xingjian Bai and Zongze Wu and Richard Zhang and Eli Shechtman and Antonio Torralba and Phillip Isola and William T. Freeman},
  journal={arXiv preprint arXiv:2603.22283},
  year={2026}
}
```
