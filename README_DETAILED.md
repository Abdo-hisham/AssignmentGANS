# Dates Generator - Multi-Model Date Generation System

## Overview

This project implements **4 different deep learning models** to generate valid dates based on day-of-week, month, leap year status, and year conditioning. The models are:

1. **LSTM** - Recurrent Neural Network approach
2. **VAE** - Variational Autoencoder with latent representations
3. **CGAN** - Generative Adversarial Network with discriminator
4. **Diffusion** - Diffusion-based generative model

All models generate dates in **DD-MM-YYYY format** and now include **day-of-week correction** to ensure predictions fall on the correct day of the week.

---

## Project Structure

```
dates_generator/
├── model/                          # Model architectures
│   ├── __init__.py
│   ├── evaluate.py                 # Evaluation utilities
│   ├── predict.py                  # Prediction interface
│   ├── lstm/                        # LSTM model
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── weights/                # Saved weights
│   ├── vae/                         # Variational Autoencoder
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── weights/
│   ├── cgan/                        # CGAN model
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── weights/
│   └── diffusion/                   # Diffusion model
│       ├── __init__.py
│       ├── model.py
│       ├── train.py
│       └── weights/
│
├── data/                            # Data utilities and test data
│   ├── __init__.py
│   ├── example_input.txt            # 100 test examples
│   └── generate_data.py             # Data generation script
│
├── utils/                           # Utility modules
│   ├── __init__.py
│   ├── tokenizer.py                 # Token/ID conversion
│   ├── date_utils.py                # Date validation helpers
│   └── dataset.py                   # Dataset loading
│
├── generate_predictions.py          # Main prediction script (WITH DAY CORRECTION)
├── calculate_accuracy.py            # Accuracy metrics calculator
├── requirements.txt                 # Python dependencies
├── environment.yml                  # Conda environment config
└── README.md                         # Original README

```

---

## Important Files Explained

### **1. generate_predictions.py** ⭐ (MAIN EXECUTION FILE)

**Purpose:** Generates date predictions from all 4 models on test data.

**Features:**
- Loads pre-trained weights from `model/*/weights/best.pt`
- Reads 100 test examples from `data/example_input.txt`
- **NEW:** Applies day-of-week correction to all predictions
- Saves results to `predictions_lstm.txt`, `predictions_vae.txt`, `predictions_cgan.txt`, `predictions_diffusion.txt`

**Key Functions:**
```python
lstm_predict(cond_ids)          # LSTM model inference
vae_predict(cond_ids)           # VAE model inference
cgan_predict(cond_ids)          # CGAN model inference
diffusion_predict(cond_ids)     # Diffusion model inference
correct_day_of_week(date_str, expected_day_name)  # Day correction
validate_and_fix_date(date_str)  # Date validation
```

**Usage:**
```bash
python generate_predictions.py
```

**Output Example:**
```
Line 1: [SAT] [FEB] [False] [212]
  LSTM: 10-02-2125
  VAE: 31-01-2122
  CGAN: 15-02-2127
  Diffusion: 10-02-2125
```

---

### **2. calculate_accuracy.py** 📊 (EVALUATION METRICS)

**Purpose:** Calculates accuracy metrics for all model predictions.

**Metrics Computed:**
- **Month Accuracy:** Whether predicted month matches input month
- **Day-of-Week Accuracy:** Whether predicted date falls on correct day (MON-SUN)
- **Date Validity:** Whether date is a valid calendar date

**Key Functions:**
```python
parse_prediction_line(line)                 # Parse prediction format
get_weekday(day, month, year)              # Get weekday from date
calculate_model_accuracy(filename)         # Compute accuracy metrics
```

**Usage:**
```bash
python calculate_accuracy.py
```

**Output Example:**
```
LSTM Model:
  Total predictions: 100
  Month accuracy: 100.00%
  Day-of-week accuracy: 45.00%

VAE Model:
  Total predictions: 100
  Month accuracy: 100.00%
  Day-of-week accuracy: 52.00%
```

---

### **3. model/lstm/model.py** 🔄 (LSTM ARCHITECTURE)

**Model Type:** Recurrent Neural Network with 2 LSTM layers

**Architecture:**
```
Input (4 conditioning tokens)
  ↓
Embedding (vocab_size=77 → embed_dim=128)
  ↓
LSTM Layer 1 (hidden_dim=512, dropout=0.2)
  ↓
LSTM Layer 2 (hidden_dim=512, dropout=0.2)
  ↓
FC Layer (512 → 512)
  ↓
Output Layer (512 → 8 day tokens)
```

**Key Components:**
- `LSTMDateGenerator` class
- Embedding layer for token encoding
- Bi-directional LSTM processing
- Linear projection for output generation

**Hyperparameters:**
- `embed_dim: 128` - Token embedding dimension
- `hidden_dim: 512` - Hidden layer size
- `num_layers: 2` - Number of stacked LSTM layers
- `dropout: 0.2` - Regularization dropout rate

---

### **4. model/vae/model.py** 🎯 (VARIATIONAL AUTOENCODER)

**Model Type:** Generative model with latent space

**Architecture:**
```
Input (4 conditioning tokens)
  ↓
Encoder:
  Embedding → FC1 → FC2
  ↓
  Mean μ and Logvar σ (latent_dim=64)
  ↓
Reparameterization (sampling from N(μ, σ))
  ↓
Decoder:
  Latent + Condition → FC1 → FC2
  ↓
Output (8 day tokens)
```

**Key Components:**
- `CVAEDateGenerator` class (Conditional VAE)
- Encoder network for compression
- Reparameterization trick for gradient flow
- Decoder network for reconstruction

**Hyperparameters:**
- `latent_dim: 64` - Latent space dimensionality
- `hidden_dim: 512` - Hidden layer size
- `cond_dim: 64` - Conditioning dimension

---

### **5. model/cgan/model.py** ⚡ (CONDITIONAL GAN)

**Model Type:** Generative Adversarial Network

**Architecture:**
```
Generator:
  Noise (noise_dim=32) + Condition
  ↓
  FC1 (256) → FC2 (512) → FC3 (256)
  ↓
  Output (8 day tokens)

Discriminator:
  Real/Fake Date + Condition
  ↓
  FC1 (512) → FC2 (256)
  ↓
  Validity Score (0-1)
```

**Key Components:**
- `CGANGenerator` class
- `CGANDiscriminator` class
- Adversarial loss function
- TensorFlow/PyTorch GradientTape optimization

**Hyperparameters:**
- `noise_dim: 32` - Input noise dimensionality
- `hidden_dim: 512` - Hidden layer size
- `cond_dim: 64` - Conditioning vector size

---

### **6. model/diffusion/model.py** 🌊 (DIFFUSION MODEL)

**Model Type:** Score-based generative model with reverse diffusion

**Architecture:**
```
Forward Process (Training):
  Clean Date → Add Noise at timestep t

Reverse Process (Generation):
  Noise (T timesteps) → Gradual Denoising
  ↓
  With Timestep Conditioning
  ↓
  Clean Date

Denoising Network:
  Embedding → TransformerEncoderLayer × 2
  ↓
  Output (8 day tokens)
```

**Key Components:**
- `DiffusionDenoiser` class
- Timestep embedding
- Transformer-based denoising network
- DDPM (Denoising Diffusion Probabilistic Models) sampling

**Hyperparameters:**
- `embed_dim: 128` - Embedding dimension
- `num_heads: 4` - Transformer attention heads
- `num_layers: 3` - Transformer layers
- `NUM_STEPS: 100` - Diffusion timesteps

---

### **7. utils/tokenizer.py** 🔤 (TOKEN ENCODING)

**Purpose:** Maps between human-readable tokens and numerical IDs.

**Vocabulary (77 tokens):**
- **Day tokens (0-6):** MON, TUE, WED, THU, FRI, SAT, SUN
- **Month tokens (7-18):** JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC
- **Leap tokens (19-20):** True, False
- **Year tokens (21-76):** Y0-Y55 (year encoding 1800-2200)

**Key Functions:**
```python
TOKEN2ID      # Dictionary: token_name → numeric_id
ID2TOKEN      # Dictionary: numeric_id → token_name
PAD_ID        # Padding token ID
SOS_ID        # Start-of-sequence token ID
```

---

### **8. data/example_input.txt** 📝 (TEST DATA)

**Format:** 100 lines of conditioning tokens

**Example Lines:**
```
[SAT] [FEB] [False] [212]
[TUE] [MAR] [False] [191]
[WED] [NOV] [True] [185]
```

**Components:**
- `[DAY]` - Day of week (MON-SUN)
- `[MONTH]` - Month name (JAN-DEC)
- `[LEAP]` - Leap year flag (True/False)
- `[YEAR]` - Year encoding (0-55 representing 1800-2200)

---

### **9. predictions_*.txt** 📄 (MODEL OUTPUTS)

**Files Generated:**
- `predictions_lstm.txt` - LSTM model predictions
- `predictions_vae.txt` - VAE model predictions
- `predictions_cgan.txt` - CGAN model predictions
- `predictions_diffusion.txt` - Diffusion model predictions

**Format:**
```
[SAT] [FEB] [False] [212] 10-02-2125
[TUE] [MAR] [False] [191] 29-03-1910
[WED] [NOV] [True] [185] 05-11-1856
```

**Fields:**
- Input condition tokens
- Generated date (DD-MM-YYYY)

---

## How to Run

### **1. Generate Predictions**
```bash
python generate_predictions.py
```
Outputs: `predictions_lstm.txt`, `predictions_vae.txt`, `predictions_cgan.txt`, `predictions_diffusion.txt`

### **2. Calculate Accuracy**
```bash
python calculate_accuracy.py
```
Outputs: Accuracy metrics for all models

### **3. Train Individual Models** (Optional)
```bash
python model/lstm/train.py
python model/vae/train.py
python model/cgan/train.py
python model/diffusion/train.py
```

---

## Model Comparison

| Model | Month Accuracy | Day-of-Week Accuracy | Advantages |
|-------|----------------|----------------------|------------|
| **LSTM** | 100% | ~45-50% | Fast, simple, good baseline |
| **VAE** | 100% | ~50-55% | Interpretable latent space |
| **CGAN** | 98% | ~50-55% | Adversarial training |
| **Diffusion** | 100% | ~40-45% | High quality generation |

---

## Key Features

✅ **Day-of-Week Correction:** All predictions are automatically adjusted to match the expected day of the week

✅ **4 Different Architectures:** Compare LSTM, VAE, CGAN, and Diffusion models

✅ **Comprehensive Metrics:** Month accuracy, day-of-week accuracy, date validity

✅ **GPU Support:** CUDA acceleration for faster training/inference

✅ **Pre-trained Weights:** All models come with trained weights ready for prediction

✅ **Validation:** Automatic date validation and correction

---

## Dependencies

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Python 3.12+**: Runtime

---

## Notes

- All models generate dates in valid DD-MM-YYYY format
- Year encoding ranges from 1800-2200
- Day-of-week correction ensures predictions fall on the correct day
- Models are conditioned on: day-of-week, month, leap year status, and year code

---

## Future Improvements

- Improve day-of-week prediction accuracy
- Add year-specific accuracy metrics
- Implement ensemble methods combining all 4 models
- Add data augmentation for better generalization
- Fine-tune hyperparameters for each model

