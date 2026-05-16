# CSE425_Project
# Unsupervised Neural Network for Multi-Genre Music Generation
> **Course:** CSE425 / EEE474 - Neural Networks | **BRAC University, Spring 2026**

---

## Authors

| Afra Musarrat Diya | 22201157 | CSE | BRAC University, Dhaka |
| Sayeba Nasir | 22201354 | CSE | BRAC University, Dhaka |
| Farjana Sikder Tisha | 22201352 | CSE | BRAC University, Dhaka |

---

## Overview

Our project follows a pipeline of **4 deep unsupervised generative models** for multi-genre music generation from raw MIDI data. Music sequences are represented as binary 88-key piano rolls at `fs = 16 steps/sec`, and four core architectures are trained end-to-end:

```
LSTM Autoencoder    Variational Autoencoder (VAE)     Transformer Decoder       RLHF Fine-Tuning
     (Task 1)               (Task 2)                       (Task 3)               (Task 4)
```

A critical finding of this work is that **Mean Squared Error (MSE) loss fails on sparse piano roll data** (sparsity ≈ 94.7%), causing models to predict silence from epoch 1. This is resolved using **Focal Loss** with positive class weighting (`γ = 2.0`, `pos_weight = 20`), which is an important fix for any system trained on binary piano roll representations.

---

## Datasets

| Dataset |  Genre Coverage |

| MAESTRO v3.0.0 | Classical Piano |
| Lakh MIDI Dataset (clean_midi) | Rock, Pop, Jazz, Electronic, Blues, Country, R&B |

---

## Preprocessing Pipeline

Raw MIDI files are processed in a 5-step pipeline:

1. **Piano Roll Extraction** : `pretty_midi` at `fs = 16` sliced to 88-key range (A0–C8), shape `(88, T)`
2. **Binarisation** : All non-zero velocities mapped to 1.0
3. **Windowing** : shape `(88, 128)` per segment
4. **Sparsity Filtering**
5. **Split Assignment** 

---

## Evaluation Metrics

### Pitch Histogram Similarity (PHS)
All notes mapped to pitch class , producing a normalised histogram.

**Lower == better.** 

### Rhythm Diversity Score

**Higher == better.**

### Human Listening Score
10 participants rated each generated audio sample on a 1–5 Likert scale across three dimensions: melodic coherence, rhythmic quality, and overall musicality. Audio rendered to MP3 via software synthesiser and distributed through Google Forms.

---

## Results

### Task 1 : LSTM Autoencoder

| Metric | Value |

| Final Training Loss (Focal) | 0.3038 |
| Final Validation Loss | 0.3970 |
| Final Test Loss | 0.4049 |
| Notes per Generated Sample | 50–106 |
| Duration per Sample | 4–13 seconds |
| Total Parameters | 1,955,928 |

> **Note:** The abstract and conclusion cite a training loss of 0.2746 (an earlier checkpoint value); the authoritative result from Table IV in the report is **0.3038**.

### Task 2 : Variational Autoencoder

| Metric | Value |

| Final Training Loss | 0.6705 |
| Final Validation Loss | 0.6801 |
| Converged KL Divergence | 0.0527 |
| Rhythm Diversity Score (generated samples) | 0.683 |
| Rhythm Diversity Score (cross-model eval) | 0.036 |
| Repetition Ratio | 0.000 |
| Posterior Collapse | None |

### Task 3 : Transformer Decoder

| Metric | Value |

| Final Training Loss | 0.7933 |
| Final Validation Loss | 0.8156 |
| Final Test Loss | 0.8765 |
| Test Perplexity | **2.40** |
| Random Generator Baseline Perplexity | 91.0 |
| Markov Chain Baseline Perplexity | 2.06 |
| Rhythm Diversity | 0.737 |
| Repetition Ratio | 0.026 |
| Human Listening Score | 3.55 / 5.0 |
| Total Parameters | 817,499 |
| Generated Compositions | 10 × 512 tokens (~32s each) |
| Genre Conditioning | 8 genres |

### Task 4 : RLHF Fine-Tuning

| Metric | Before RL | After RL |

| Average Reward | 0.554 | **0.588** (~6.1% gain) |
| Valid MIDI Samples | 2 / 10 | **8 / 10** |
| Human Listening Score | 2.50 | **3.68** |
| Pitch Histogram Similarity | 1.398 | **1.016** |

---

## Cross-Model Comparison

| Model | Loss | Perplexity | PHS | Rhythm Div. | Rep. Ratio | Genre | HLS |

| Random Generator | - | 91.0 | 0.372 | 0.067 | 0.000 | None | 1.10 |
| Markov Chain | - | 2.06 | 1.393 | 0.067 | 0.067 | Weak | 2.30 |
| Task 1: LSTM AE | 0.2746 | - | 0.276 | 0.320 | 0.000 | Single | 3.38 |
| Task 2: VAE | 0.6705 | - | 0.700 | 0.036 | 0.009 | Multi | 3.46 |
| Task 3: Transformer | 0.8765 | 2.40 | 1.229 | 0.737 | 0.026 | Strong | 3.55 |
| Task 4: RL Before | - | - | 1.398 | 0.597 | 0.040 | Strong | 2.50 |
| Task 4: RL After | - | - | 1.016 | 0.245 | 0.052 | Strong | 3.68 |

---

## Individual Contributions

### Afra Musarrat Diya ( 22201157 )
- Built the full **preprocessing pipeline**: MIDI-to-piano-roll conversion, binarisation, 128-step windowing, sparsity filtering, and dataset splitting for both MAESTRO and Lakh MIDI
- Conducted **Exploratory Data Analysis** (pitch, velocity, duration, note count distributions) and identified the critical 94.7% sparsity finding that shaped the loss function choice for all tasks
- Designed and trained the **LSTM Autoencoder (Task 1)**: 2-layer encoder/decoder, hidden size, latent dim; implemented the latent replication strategy across all decoder steps
- Identified the **MSE silence failure mode** and resolved it with Focal Loss; justified the binarisation threshold for class-imbalance-suppressed note probabilities
- Final training / validation / test losses; generated verified MIDI samples 
- **Report sections**: Introduction, Datasets & Preprocessing, Task 1 Architecture & Results, Focal Loss vs MSE discussion, Task 1 contribution to Conclusion

---

### Sayeba Nasir ( 22201354 )
- Designed, Trained the **Transformer Decoder (Task 3)**: tokenisation scheme, sinusoidal positional encoding, learnable genre embeddings for 8 genres, causal masking; testing perplexity
- Implemented sampling; generated 10 long-form compositions across all 8 genres
- Implemented **RLHF Fine-Tuning (Task 4)** via REINFORCE (200 steps, lr=5×10⁻⁵): designed composite reward function note density + duration adequacy + pitch diversity; found improved valid MIDI output from and Human Listening Score
- Conducted the **Human Listening Survey**: rendered MIDI to MP3, distributed to 10 participants, aggregated perceptual ratings across all model conditions
- **Report sections**: Transformer Architecture, Results, RLHF Discussion, Human Listening Score metric definition, Model Comparison section, Task 3 & Task 4 contribution to Conclusion

---

### Farjana Sikder Tisha  ( 22201352 )
- Implemented both **Baseline Models**: Random Generator and Markov Chain, establishing reference benchmarks for all subsequent comparisons
- Designed and trained the **Variational Autoencoder (Task 2)**: probabilistic latent space (dim 128) with reparameterisation trick and ELBO objective combining Focal Loss reconstruction and KL divergence
- Designed the **KL annealing schedule** confirmed non-trivial converged KL 
- Did latent interpolation experiment confirming smooth latent space continuity; generated 8 genre-labelled samples with Rhythm Diversity and Repetition Ratio
- **Report sections**: VAE Architecture & Results, all Evaluation Metrics definitions (PHS, Rhythm Diversity, Repetition Ratio), KL Annealing discussion, System Architectural Trade-offs (VAE), Evaluation Metric Interpretation, Limitations, Directions toward Autoregressive Generation, Task 2 contribution to Conclusion

---

## Project Structure

```
CSE425_Project/                          (main branch)
│
├── Pre-processing/          # MIDI preprocessing pipeline & EDA
├── Task 1/                  # LSTM Autoencoder
├── TASK 2/                  # Variational Autoencoder
├── TASK 3/                  # Transformer Decoder
├── TASK 4/                  # RLHF Fine-Tuning (REINFORCE)
├── models/                  # Saved model checkpoints
├── outputs/                 # Generated MIDI & audio outputs
├── Baseline_Final.ipynb     # Random Generator & Markov  baselines
└── README.md
```

---

## References

1. Hawthorne et al., *"Enabling factorized piano music modeling and generation with the MAESTRO dataset"*, ICLR 2019.
2. Lin et al., *"Focal Loss for Dense Object Detection"*, ICCV 2017.
3. Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017.
4. Kingma & Welling, *"Auto-Encoding Variational Bayes"*, ICLR 2014.

---

<div align="center">
  <sub>Built by Afra Musarrat Diya · Sayeba Nasir · Farjana Sikder Tisha — BRAC University, Spring 2026</sub>
</div>
