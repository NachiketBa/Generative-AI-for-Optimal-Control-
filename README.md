# Generative AI for: LTI, Minthreat, and Zermelo Navigation Problems

> VAEs and GANs trained to generate synthetic trajectory data across three optimal control problems, with physics-informed loss terms where applicable.

---

## What this is

This project applies generative models to three optimal control problems where real training trajectories are expensive to compute. Each model learns the distribution of those trajectories and generates new synthetic ones.

Eight scripts are included across three problem domains:

| Script | Model | Problem |
|---|---|---|
| `S_VAE_LTI.py` | Standard VAE | LTI system (noisy trajectories) |
| `Split_VAE_LTI.py` | Split-VAE | LTI system (noisy + noiseless) |
| `S_VAE_minthreat.py` | Standard VAE | Minimum threat path planning (noisy) |
| `Split_VAE_minthreat.py` | Split-VAE | Minimum threat path planning (noisy + noiseless) |
| `S_VAE_zermelo.py` | Standard VAE | Zermelo navigation (min-time) |
| `Z_VAE_zermelo.py` | Physics-informed VAE | Zermelo navigation + Hamiltonian constraint |
| `SGAN_zermelo.py` | Standard GAN | Zermelo navigation (min-time) |
| `ZGAN_zermelo.py` | Physics-informed GAN | Zermelo navigation + Hamiltonian + heading angle constraints |

---

## Problem Backgrounds

### LTI System
A Linear Time-Invariant system where each trajectory is a 100,100-dimensional vector (100 time steps x 100 state/control features, flattened). Two datasets exist: noisy (`set09`) and noiseless (`realcase02`), and the Split-VAE handles both at once.

### Minthreat Path Planning
A minimum-threat trajectory planning problem where an agent navigates from a start to a goal while avoiding high-threat regions. Each trajectory is a 2057-dimensional vector. Noisy and noiseless variants exist here too.

### Zermelo Navigation
A minimum-time navigation problem where a vessel moves through a current field from `[0.0, 0.8]` to `[-0.8, -0.9]` with a speed V = 0.05. Trajectories are 400-dimensional (7 state/costate variables x 25 discretization points, plus heading angles). The physics constraint is the Pontryagin Hamiltonian, which must equal zero along optimal trajectories. The Z-VAE and ZGAN models enforce this directly in the loss.

---

## Model Architectures

### Standard VAE (S-VAE)

Used for: LTI, Minthreat, and Zermelo.

```
Input (n_features-dim trajectory)
    |
Encoder: [Linear -> (LayerNorm) -> ReLU] x N -> [mu, log_var]
    |
Reparameterize: z = mu + eps * sigma,  eps ~ N(0, I)
    |
Decoder: [Linear -> (LayerNorm) -> ReLU] x N -> x_hat
    |
Output (n_features-dim reconstructed trajectory)
```

**Loss:**
```
L = MSE(x_hat, x)  +  KL[ q(z|x) || N(0, I) ]
```

LayerNorm is used in the LTI version due to its very small dataset. The Zermelo and Minthreat S-VAEs use plain ReLU stacks without normalization.

---

### Split-VAE

Used for: LTI and Minthreat. Trains on noisy and noiseless trajectories simultaneously. A regularization penalty on noiseless samples pushes z1 toward zero, nudging the model to represent noise as a structured deviation in z1.

```
Input (noisy or noiseless trajectory) + label (0 = noisy, 1 = noiseless)
    |
Encoder -> [z1_mu, z1_logvar, z2_mu, z2_logvar]
    |
Reparameterize z1, z2
    |
Decoder([z1, z2]) -> x_hat
```

**Loss:**
```
L = MSE(x_hat, x)
  + KL[q(z1) || N(0, I)]
  + KL[q(z2) || N(0, I)]
  + lambda_reg * ||z1_mu||^2   (applied only to noiseless samples)
```
---

### Z-VAE (Physics-Informed VAE)

Used for: Zermelo only. Same encoder-decoder structure as the S-VAE, but the decoder output is passed through a Hamiltonian calculator before computing the loss. This penalizes trajectories that violate Pontryagin's necessary conditions for optimality.

**Hamiltonian (per discretization node):**
```
H = 1 + p_x * V * cos(theta) + p_x * f_x
      + p_y * V * sin(theta) + p_y * f_y
```

**Loss:**
```
L = MSE(x_hat, x)
  + KL[ q(z|x) || N(0, I) ]
  + 3 * MSE(H, 0)
```

The weight of 3 on the physics term was set empirically.

---

### Standard GAN (SGAN)

Used for: Zermelo. Generator maps a 20-dim uniform latent vector to a 175-dim trajectory. Discriminator classifies real vs. fake using the first 50 features. Both networks use LeakyReLU(0.1) with Dropout(0.2), trained with SGD and BCE loss.

```
z ~ Uniform(-1, 1)^20  ->  Generator  ->  fake trajectory (175-dim)
real trajectory (175-dim)  ->  Discriminator  ->  real/fake score
```

**Generator loss:**
```
L_G = BCE(D(G(z)), 1)
```

**Discriminator loss:**
```
L_D = BCE(D(x_real), 1) + BCE(D(G(z)), 0)
```

---

### ZGAN (Physics-Informed GAN)

Two variants:

**ZGAN1** adds a Hamiltonian penalty to the generator loss:
```
L_G = BCE(D(G(z)), 1) + MSE(H(G(z)), 0)
```

**ZGAN2** adds both the Hamiltonian penalty and a heading angle consistency term:
```
heading angle: psi = atan(p_y / p_x)

L_G = BCE(D(G(z)), 1)
    + MSE(H(G(z)), 0)
    + MSE(psi(G(z)), theta(G(z)))
```

The discriminator in both ZGAN variants operates on the first 50 features only, same as SGAN.

---

## Hyperparameters

### VAE Models

| Parameter | LTI S-VAE | LTI Split-VAE | Minthreat S-VAE | Minthreat Split-VAE | Zermelo S-VAE | Z-VAE |
|---|---|---|---|---|---|---|
| `n_features` | 100,100 | 100,100 | 2,057 | 2,057 | 400 | 400 |
| `latent_size` | 64 | 20+20 | 32 | 20+20 | 32 | 32 |
| `hidden_dims` | 225-196-125-100-81 | 625-400 | 225-196-125-100-81 | 625-400-225 | 324-225-196-125-100-81 | 225-196-125-100-81 |
| `epochs` | 1000 | 1000 | 1000 | 1000 | 2000 | 2000 |
| `batch_size` | 32 | 32 | 32 | 32 | 32 | 32 |
| `learning_rate` | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| `lambda_reg` | n/a | 1 | n/a | 5 | n/a | n/a |
| Training samples | 500 | 500 noisy + 10,000 noiseless | 200 | 200 noisy + 200 noiseless | 500 | 500 |

### GAN Models (Zermelo only)

| Parameter | SGAN | ZGAN1 | ZGAN2 |
|---|---|---|---|
| `n_features` | 175 | 175 | 175 |
| `input_dimG` (latent) | 20 | 20 | 20 |
| Generator layers | 64-100-225-400-625-900 | 64-100-225-400-625-900 | 64-100-225-400-625-900 |
| Discriminator layers | 900-625-400-225-100-25 | 900-625-400-225-100 | 900-625-400-225-100 |
| Discriminator input | first 50 features | first 50 features | first 50 features |
| `epochs` | 500 | 500 | 500 |
| `batch_size` | 64 | 64 | 64 |
| Optimizer | SGD (lr=0.01) | SGD (lr=0.01) | SGD (lr=0.01) |
| Physics loss | none | MSE(H, 0) | MSE(H, 0) + MSE(psi, theta) |

---


## Design notes

**LayerNorm in LTI VAEs only.** The LTI dataset has only 500 training samples, making batch statistics unreliable. LayerNorm normalizes per sample so it stays stable at any batch size. The Minthreat and Zermelo models use plain ReLU stacks, where the larger datasets make this unnecessary.

**Split-VAE regularization weights differ by problem.** The penalty on z1 for noiseless samples is 1 for LTI and 5 for Minthreat, because the signal-to-noise characteristics differ between the two domains. Higher lambda pushes the noiseless encoder more aggressively toward zero in z1.

**Hamiltonian weight of 3 in Z-VAE.** The reconstruction and KL losses dominate early in training. Weighting the physics term at 3x compensates for its smaller raw magnitude and keeps it from being crowded out by the other two terms.

**Discriminator sees only first 50 features in GANs.** The full 175-dim Zermelo trajectory includes states, costates, and heading angles. The discriminator operates on position and heading states only (columns 0-49), which are the most directly observable. Physics constraints on the remaining features are handled through the generator loss terms instead.

**SGD for GAN training.** Adam can cause mode collapse in GANs on small datasets. SGD with lr=0.01 trains more slowly but produces more stable convergence for these trajectory distributions.

**Best-checkpoint saving in VAE scripts.** The Minthreat S-VAE and all Zermelo VAEs track the lowest training loss and save that checkpoint. Generated samples come from the best checkpoint, not the final epoch.

---


