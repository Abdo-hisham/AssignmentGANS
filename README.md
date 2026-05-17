# DSAI 490 – Assignment 2: Dates Generator

Four generative models: **LSTM · CVAE · CGAN · Diffusion**

---

## 1. Setup

### Option A – Conda (recommended)
```bash
conda env create -f environment.yml
conda activate dates_gen
```

### Option B – pip
```bash
pip install -r requirements.txt
```

---

## 2. Data

If `data/data.txt` was provided, place it in the `data/` folder.  
Otherwise, generate synthetic data:
```bash
python data/generate_data.py
```

---

## 3. Train

### Train all models at once:
```bash
python train_all.py
```

### Or train individually:
```bash
python model/lstm/train.py
python model/vae/train.py
python model/cgan/train.py
python model/diffusion/train.py
```

Weights are saved to `model/<name>/weights/`.

---

## 4. Predict (required format)

```bash
python model/predict.py -i data/example_input.txt -o predictions.txt
```

Optionally choose a model:
```bash
python model/predict.py -i data/example_input.txt -o predictions.txt --model vae
python model/predict.py -i data/example_input.txt -o predictions.txt --model cgan
python model/predict.py -i data/example_input.txt -o predictions.txt --model diffusion
```

---

## 5. Evaluate + Plots

```bash
python model/evaluate.py --model lstm --n 1000
python model/evaluate.py --model vae  --n 1000
python model/evaluate.py --model cgan --n 1000
python model/evaluate.py --model diffusion --n 300
```

Loss curves and pass-rate plots are saved to `model/plots/`.

---

## Project Structure

```
dates_generator/
├── data/
│   ├── data.txt               # Training data
│   ├── example_input.txt      # Example inputs (conditions only)
│   └── generate_data.py       # Synthetic data generator
├── utils/
│   ├── tokenizer.py           # Custom tokenizer & vocab
│   ├── date_utils.py          # Date validation & fallback search
│   └── dataset.py             # PyTorch Dataset / DataLoader
├── model/
│   ├── predict.py             # ← Required inference entry-point
│   ├── evaluate.py            # Evaluation + plotting
│   ├── lstm/
│   │   ├── model.py           # LSTM autoregressive model
│   │   └── train.py
│   ├── vae/
│   │   ├── model.py           # CVAE model
│   │   └── train.py
│   ├── cgan/
│   │   ├── model.py           # CGAN Generator + Discriminator
│   │   └── train.py
│   └── diffusion/
│       ├── model.py           # Discrete diffusion denoiser
│       └── train.py
├── train_all.py               # Train all models sequentially
├── environment.yml
└── requirements.txt
```

---

## Model Summaries

| Model | Type | Architecture | Key idea |
|-------|------|--------------|----------|
| **LSTM** | Autoregressive | Embedding → LSTM → FC | Teacher-forced seq-to-seq; generates tokens one at a time |
| **CVAE** | Latent variable | Encoder+Decoder MLP | Samples z from learned posterior; 10 independent heads |
| **CGAN** | Adversarial | Generator + Discriminator | Gumbel-softmax for discrete gradients; label smoothing |
| **Diffusion** | Denoising | Transformer denoiser | Token corruption + Transformer; ancestral sampling |

## Evaluation Metric

**Condition Pass Rate** = fraction of generated dates where all 4 conditions are met.  
This is preferred over accuracy because the generation problem has many valid outputs per input.
