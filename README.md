# CSE425_Project

# Unsupervised Neural Network for Multi-Genre Music Generation

> **Course:** CSE425 / EEE474 — Neural Networks | **BRAC University, Spring 2026**

---

## Authors

| Name | Department | University |
|------|-----------|------------|
| Afra Musarrat Diya | CSE | BRAC University, Dhaka |
| Sayeba Nasir | CSE | BRAC University, Dhaka |
| Farjana Sikder Tisha | CSE | BRAC University, Dhaka |

---

## Overview

This project follows a pipeline of **four deep unsupervised generative models** for multi-genre music generation from raw MIDI data. Music sequences are represented as binary 88-key piano rolls at `fs = 16 steps/sec`, and three core architectures are trained end-to-end:

```
LSTM Autoencoder  →  Variational Autoencoder (VAE)  →  Transformer Decoder  →  RLHF Fine-Tuning
     (Task 1)               (Task 2)                       (Task 3)               (Task 4)
```

An interesting finding of this work is that **Mean Squared Error (MSE) loss fails on sparse piano roll data**. This is resolved using **Focal Loss** with positive class weighting (by `γ = 2.0`, `pos_weight = 20`), which is important factor for any system trained on binary piano roll representations.


### Datasets

| Dataset | Genre Coverage | Link |
|---------|---------------|------|
| MAESTRO v3.0.0 | Classical Piano |
| Lakh MIDI Dataset | Multi-Genre |



## 🧪 Evaluation Metrics

### Pitch Histogram Similarity (PHS)
Maps notes to pitch classes (`pitch % 12`) and computes L1 distance between generated and reference 12-element histograms. **Lower is better.**

### Rhythm Diversity Score
```
D_rhythm = #unique_durations / #total_notes
```
**Higher is better.**

### Repetition Ratio
4-gram pattern repetition over the pitch sequence of generated output.
```
R = #repeated_4grams / #total_4grams
```
### Human Listening Score
10 participants rated each generated audio sample on a 1–5 Likert scale across three dimensions: melodic coherence, rhythmic quality, and overall musicality. Audio rendered via MuseScore and distributed through Google Forms.

---

## Results

### Task 1 — LSTM Autoencoder

| Metric | Value |
|--------|-------|
| Final Training Loss (Focal) | 0.2746 |
| Final Validation Loss | 0.3970 |
| Final Test Loss | 0.4049 |
| Notes per Generated Sample | 50–106 |
| Duration per Sample | 4–13 seconds |
| Total Parameters | 1,955,928 |

### Task 2 — Variational Autoencoder

| Metric | Value |
|--------|-------|
| Final Training Loss | 0.6705 |
| Final Validation Loss | 0.6801 |
| Converged KL Divergence | 0.0527 |
| Rhythm Diversity Score | 0.683 |
| Repetition Ratio | 0.000 |
| Posterior Collapse | None  |

### Task 3 — Transformer

| Metric | Value |
|--------|-------|
| Test Perplexity | **2.40** |
| Random Baseline Perplexity | 91.0 |
| Markov Baseline Perplexity | 2.06 |
| Rhythm Diversity | 0.737 |
| Human Listening Score | 3.55 / 5.0 |
| Generated Compositions | 10 × 512 tokens (~32s each) |

### Task 4 — RLHF Fine-Tuning

| Metric | Before RL | After RL |
|--------|-----------|----------|
| Average Reward | 0.555 | **0.605** (+9.1%) |
| Valid MIDI Samples | 2 / 10 | **8 / 10** |
| Human Listening Score | 2.50 | **3.68** |
| Pitch Histogram Similarity | — | 1.016 |


## References

1. Hawthorne et al., *"Enabling factorized piano music modeling and generation with the MAESTRO dataset"*, ICLR 2019.
2. Lin et al., *"Focal Loss for Dense Object Detection"*, ICCV 2017.
3. Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017.
4. Kingma & Welling, *"Auto-Encoding Variational Bayes"*, ICLR 2014.


<div align="center">
  <sub> by Afra Musarrat Diya · Sayeba Nasir · Farjana Sikder Tisha — BRAC University, 2026</sub>
</div>
