# CSE425_Project
# Unsupervised Neural Network for Multi-Genre Music Generation
> **Course:** CSE425 / EEE474 - Neural Networks | **BRAC University, Spring 2026**

---

## Authors

| Name | ID | Department | University |
|------|-----|-----------|------------|
| Afra Musarrat Diya | 22201157 | CSE | BRAC University, Dhaka |
| Sayeba Nasir | 22201354 | CSE | BRAC University, Dhaka |
| Farjana Sikder Tisha | 22201352 | CSE | BRAC University, Dhaka |

---

## Overview

Our project follows a pipeline of **4 deep unsupervised generative models** for multi-genre music generation from raw MIDI data. Music sequences are represented as binary piano rolls at `fs = 16 steps/sec`, and four core architectures:

```
LSTM Autoencoder     Variational Autoencoder (VAE)     Transformer Decoder     RLHF Fine-Tuning
     (Task 1)               (Task 2)                       (Task 3)               (Task 4)
```

A critical finding of this work is that **Mean Squared Error (MSE) loss fails on sparse piano roll data** (sparsity ≈ 94.7%), causing models to predict silence from epoch 1. This is resolved using **Focal Loss** with positive class weighting (`γ = 2.0`, `pos_weight = 20`), which is an important fix for any system trained on binary piano roll representations.

---

## Datasets

| Dataset | Genre Coverage |
|---------|---------------|
| MAESTRO v3.0.0 | Classical Piano |
| Lakh MIDI Dataset (clean_midi) | Rock, Pop, Jazz, Electronic, Blues, Country, R&B |

---

## Preprocessing Pipeline

Raw MIDI files are processed in a 5-step pipeline:

| Step | Description |
|------|-------------|
| 1. Piano Roll Extraction | `pretty_midi` at `fs = 16`, shape `(88, T)` |
| 2. Binarisation | All non-zero velocities mapped to 1.0 |
| 3. Windowing | Non-overlapping 128-step windows → shape `(88, 128)` per segment |
| 4. Sparsity Filtering  |
| 5. Split Assignment  |

---

## Evaluation Metrics

| Metric | Formula | Direction |
|--------|---------|-----------|
| Pitch Histogram Similarity (PHS) | `H(p,q) = Σ \|p_i - q_i\|` over 12 pitch classes | Lower is better |
| Rhythm Diversity Score | `D_rhythm = #unique_durations / #total_notes` | Higher is better |
| Repetition Ratio | `R = #repeated_4-grams / #total_4-grams` | 0.1 to 0.5 is coherent |
| Human Listening Score | 1–5 Likert scale averaged over 10 participants (melodic coherence, rhythmic quality, overall musicality) | Higher is better |

---

## Results

### Task 1 : LSTM Autoencoder

| Metric | Value |
|--------|-------|
| Final Training Loss (Focal) | 0.3038 |
| Final Validation Loss | 0.3970 |
| Final Test Loss | 0.4049 |
| Notes per Generated Sample | 50–106 |
| Duration per Sample | 4–13 seconds |
| Total Parameters | 1,955,928 |

> The abstract and conclusion cite a training loss of 0.2746 (an earlier checkpoint value); the authoritative result from Table IV in the report is **0.3038**.

---

### Task 2 : Variational Autoencoder

| Metric | Value |
|--------|-------|
| Final Training Loss | 0.6705 |
| Final Validation Loss | 0.6801 |
| Converged KL Divergence | 0.0527 |
| Rhythm Diversity Score (generated samples) | 0.683 |
| Rhythm Diversity Score (cross-model eval) | 0.036 |
| Repetition Ratio | 0.000 |
| Posterior Collapse | None |

---

### Task 3 : Transformer Decoder

| Metric | Value |
|--------|-------|
| Final Training Loss | 0.7933 |
| Final Validation Loss | 0.8156 |
| Final Test Loss | 0.8765 |
| Test Perplexity | 2.40 |
| Random Generator Baseline Perplexity | 91.0 |
| Markov Chain Baseline Perplexity | 2.06 |
| Rhythm Diversity | 0.737 |
| Repetition Ratio | 0.026 |
| Human Listening Score | 3.55 / 5.0 |
| Total Parameters | 817,499 |
| Generated Compositions | 10 × 512 tokens |
| Genre Conditioning | 8 genres |

---

### Task 4 : RLHF Fine-Tuning

| Metric | Before RL | After RL |
|--------|-----------|----------|
| Average Reward | 0.554 | **0.588** |
| Valid MIDI Samples | 2 / 10 | **8 / 10** |
| Human Listening Score | 2.50 | **3.68**  |
| Pitch Histogram Similarity | 1.398 | **1.016** |

---

## Cross-Model Comparison

| Model | Loss | Perplexity | PHS | Rhythm Div. | Rep. Ratio | Genre | HLS |
|-------|------|-----------|-----|-------------|------------|-------|-----|
| Random Generator | - | 91.0 | 0.372 | 0.067 | 0.000 | None | 1.10 |
| Markov Chain | - | 2.06 | 1.393 | 0.067 | 0.067 | Weak | 2.30 |
| Task 1: LSTM AE | 0.2746 | - | 0.276 | 0.320 | 0.000 | Single | 3.38 |
| Task 2: VAE | 0.6705 | - | 0.700 | 0.036 | 0.009 | Multi | 3.46 |
| Task 3: Transformer | 0.8765 | 2.40 | 1.229 | 0.737 | 0.026 | Strong | 3.55 |
| Task 4: RL Before | - | - | 1.398 | 0.597 | 0.040 | Strong | 2.50 |
| Task 4: RL After | - | - | 1.016 | 0.245 | 0.052 | Strong | 3.68 |

---

## Individual Contributions

### Afra Musarrat Diya : 22201157

| Area | Details |
|------|---------|
| Preprocessing Pipeline | MIDI-to-piano-roll conversion, binarisation, 128-step windowing, sparsity filtering, and dataset splitting for both MAESTRO and Lakh MIDI |
| Exploratory Data Analysis | Pitch, velocity, duration, note count distributions; identified critical 94.7% sparsity finding that shaped the loss function choice for all tasks |
| LSTM Autoencoder (Task 1) | 2-layer encoder/decoder, hidden size, latent dim ); implemented latent replication strategy across all decoder steps |
| Loss Function Fix | Identified MSE silence failure mode; resolved with Focal Loss (γ=2.0, pos_weight=20); justified binarisation threshold for class-imbalance-suppressed note probabilities |
| Generated 5 verified MIDI samples per window |
| Report Sections | Introduction, Datasets & Preprocessing, Task 1 Architecture & Results, Focal Loss vs MSE discussion, Task 1 contribution to Conclusion |

---

### Sayeba Nasir : 22201354

| Area | Details |
|------|---------|
| Transformer Decoder (Task 3) | Tokenisation scheme (vocab size 91), sinusoidal positional encoding, learnable genre embeddings for 8 genres, causal masking, testing perplexity |
| Generation | Temperature sampling; generated 10 long-form compositions across all 8 genres |
| RLHF Fine-Tuning (Task 4) | Reinforce algorithm, 200 steps, lr=5×10⁻⁵; composite reward (note density + duration adequacy + pitch diversity) |
| Human Listening Survey | Rendered MIDI to MP3 ; distributed to 10 participants; aggregated perceptual ratings across all model conditions |
| Report Sections | Transformer Architecture & Results, RLHF Discussion, Human Listening Score metric definition, Model Comparison section, Task 3 & Task 4 contribution to Conclusion |

---

### Farjana Sikder Tisha : 22201352

| Area | Details |
|------|---------|
| Baseline Models | Random Generator and Markov Chain; established reference benchmarks for all subsequent comparisons |
| Variational Autoencoder (Task 2) | Probabilistic latent space, reparameterisation trick, ELBO objective combining Focal Loss reconstruction and KL divergence |
| KL Annealing | Linear schedule β: 0→1 over 15 warmup epochs to prevent posterior collapse; confirmed non-trivial converged KL |
| Latent Interpolation | 8-step experiment confirming smooth latent space continuity; generated 8 genre-labelled samples with Rhythm Diversity and Repetition Ratio|
| Report Sections | VAE Architecture & Results, all Evaluation Metrics definitions (PHS, Rhythm Diversity, Repetition Ratio), KL Annealing discussion, System Architectural Trade-offs, Evaluation Metric Interpretation, Limitations, Directions toward Autoregressive Generation, Task 2 contribution to Conclusion |

---

## Project Structure

```
CSE425_Project/                    (main branch)
│
├── Pre-processing/                # MIDI preprocessing pipeline & EDA
├── Task 1/                        # LSTM Autoencoder
├── TASK 2/                        # Variational Autoencoder
├── TASK 3/                        # Transformer Decoder
├── TASK 4/                        # RLHF Fine-Tuning (REINFORCE)
├── models/                        # model checkpoints
├── outputs/                       # Generated MIDI & audio outputs
├── Baseline_Final.ipynb           # Random Generator & Markov Chain baselines
└── README.md
```

---

## References

| # | Citation |
|---|---------|
| 1 | Hawthorne et al., *"Enabling factorized piano music modeling and generation with the MAESTRO dataset"*, ICLR 2019 |
| 2 | Lin et al., *"Focal Loss for Dense Object Detection"*, ICCV 2017 |
| 3 | Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017 |
| 4 | Kingma & Welling, *"Auto-Encoding Variational Bayes"*, ICLR 2014 |

---

<div align="center">
  <sub>by Afra Musarrat Diya · Sayeba Nasir · Farjana Sikder Tisha . BRAC University, Spring 2026</sub>
</div>
