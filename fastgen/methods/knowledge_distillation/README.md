# Knowledge Distillation Methods

Learn from pre-computed teacher denoising trajectories.

## KD (Knowledge Distillation)

**File:** [`KD.py`](KD.py) | **Reference:** [Luhman & Luhman, 2021](https://arxiv.org/abs/2101.02388)

MSE loss between student prediction and teacher output: `L = ||f(x_t, t) - x_0^teacher||²`

**Data Requirements:**
- Single-step: `{"real": clean, "noise": noise, "condition": cond}`
- Multi-step: `{"real": clean, "path": [B, steps, C, H, W], "condition": cond}`

**Key Parameters:**
- `student_sample_steps`: Number of student steps
- `sample_t_cfg.t_list`: Timesteps (must align with path)

**Configs:** [`WanT2V/config_kd.py`](../../configs/experiments/WanT2V/config_kd.py), [`SDXL/config_kd.py`](../../configs/experiments/SDXL/config_kd.py), [`CogVideoX/config_kd.py`](../../configs/experiments/CogVideoX/config_kd.py)

---

## CausalKD

**File:** [`KD.py`](KD.py) | **Reference:** [Yin et al., 2024](https://arxiv.org/abs/2412.07772)

KD for causal video models with inhomogeneous timesteps and autoregressive generation.

**Data Requirements:**
- `{"real": [B,C,T,H,W], "path": [B,steps,C,T,H,W], "condition": cond}`

**Key Parameters:**
- `context_noise`: Noise for cached context
- See also key parameters of KD above

**Configs:** [`WanT2V/config_kd_path.py`](../../configs/experiments/WanT2V/config_kd_path.py), [`CosmosPredict2/config_kd_causal.py`](../../configs/experiments/CosmosPredict2/config_kd_causal.py)

---

## 3-Stage Pipeline (ODE Pair Init → CausalKD → Self-Forcing)

The full Self-Forcing pipeline for causal video models consists of three stages:

### Stage 1: Generate ODE Pairs

Generate clean videos with a bidirectional teacher and capture the actual ODE trajectory at target timesteps.

```bash
PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 --standalone \
    scripts/generate_ode_pairs.py \
    --config fastgen/configs/experiments/CosmosPredict2/config_dmd2.py \
    --prompt_file <prompts.txt> --output_dir ODE_PAIRS/cosmos_t2w \
    --num_steps 35 --guidance_scale 3.0 \
    - trainer.ddp=True model.guidance_scale=3.0
```

Output per sample: `latent.pth` (clean), `path.pth` (ODE trajectory), `prompt.txt` (text), `video.mp4` (decoded reference).

### Stage 2: CausalKD Training

Train a causal student network on ODE pair data with MSE regression.

```bash
PYTHONPATH=$(pwd) torchrun --nproc_per_node=8 --standalone \
    train.py --config=fastgen/configs/experiments/CosmosPredict2/config_kd_causal.py \
    - trainer.fsdp=True dataloader_train.data_dir=ODE_PAIRS/cosmos_t2w
```

### Stage 2→3 Bridge: Extract Weights

Extract the student network weights from the CausalKD checkpoint.

```bash
python scripts/extract_net_weights.py \
    --input <save_dir>/checkpoints/0010000.pth --output ode_init.pt
```

### Stage 3: Self-Forcing Training

GAN + AR rollout training using the CausalKD-initialized student.

```bash
PYTHONPATH=$(pwd) torchrun --nproc_per_node=8 --standalone \
    train.py --config=fastgen/configs/experiments/CosmosPredict2/config_sf.py \
    - trainer.fsdp=True
```

The `config_sf.py` loads `ode_init.pt` via `pretrained_student_net_path`.

### Consolidated Script

All stages can also be run via the consolidated script:

```bash
bash scripts/train_cosmos_sf.sh 1 --prompt_file prompts.txt --output_dir ODE_PAIRS/cosmos_t2w
bash scripts/train_cosmos_sf.sh 2
bash scripts/train_cosmos_sf.sh bridge --input <ckpt>.pth --output ode_init.pt
bash scripts/train_cosmos_sf.sh 3
```
