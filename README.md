# ECG-classification

A deep learning system for **multi-class ECG heartbeat arrhythmia classification** using a dual-branch PyTorch architecture that combines raw ECG signal representations with engineered physiological features.

The project focuses on improving robustness to **recording-level distribution shifts** using **Correlation Alignment (CORAL)**, while addressing class imbalance through weighted focal loss and class-balanced sampling.

---

## 🚀 Key Highlights

* **Multi-class ECG arrhythmia classification** using deep learning
* Dual-branch architecture combining:

  * 1D CNN / ResNet-style ECG signal encoder
  * Tabular physiological feature encoder
* **Late feature fusion** for joint representation learning
* **CORAL domain alignment** to reduce feature distribution mismatch between training and test recordings
* **Weighted Focal Loss** for handling severe class imbalance
* Class-balanced sampling using inverse-frequency weighting
* ECG signal augmentation using:

  * Temporal shifts
  * Gain variation
  * Gaussian noise
  * Optional polarity inversion
* **5-fold Stratified Cross-Validation**
* **Test-Time Augmentation (TTA)** using temporal shifts
* AdamW optimization with learning-rate scheduling
* Early stopping and best-model restoration
* GPU-accelerated training with PyTorch

---

## 🧠 Architecture

The model consists of two parallel feature-extraction branches.

```text
                    ECG Input
                       │
              ┌────────┴────────┐
              │                 │
       Signal Branch       Tabular Branch
              │                 │
        1D CNN / ResNet          MLP
              │                 │
     Attention + Avg + Max       │
          Pooling                │
              │                 │
           128-D                64-D
              │                 │
              └────────┬────────┘
                       │
                 Feature Fusion
                       │
                    192-D
                Fused Feature
                       │
              ┌────────┴────────┐
              │                 │
          Classification      CORAL
              │              Alignment
              │                 │
              └────────┬────────┘
                       │
              4-Class Prediction
```

The signal branch processes **three ECG channels**:

1. Raw ECG beat
2. First-order difference
3. Second-order difference

These representations are processed using convolutional residual blocks followed by attention, average, and max pooling. The resulting representation is projected into a **128-dimensional feature vector**.

## A separate MLP processes **18 engineered tabular features** into a **64-dimensional representation**. The two branches are then concatenated into a **192-dimensional fused feature representation**.

## 🔬 Domain Alignment with CORAL

A major component of the project is **Correlation Alignment (CORAL)**.

Instead of relying only on classification loss, the model compares the covariance structure of:

* labelled training features
* unlabeled test features

The CORAL objective minimizes the relative difference between their covariance matrices:

```text
CORAL = ||Cs - Ct||²F / ||Cs||²F
```

where:

* `Cs` = source/training feature covariance
* `Ct` = target/test feature covariance

This encourages the learned representation to become more robust to distribution differences between ECG recordings.

The implementation uses a **scale-free CORAL formulation with λ = 0.5**. Importantly, test labels are never used; only unlabeled test features participate in the alignment term.

---

## ⚖️ Handling Class Imbalance

The dataset contains substantial class imbalance, making conventional cross-entropy training less suitable.

The project addresses this using two complementary techniques:

### Weighted Focal Loss

Focal loss with:

```text
γ = 2
```

reduces the contribution of easy examples while emphasizing difficult and minority-class examples.

### Class-Balanced Sampling

Training samples are drawn using softened inverse-frequency weighting:

```text
P(class) ∝ n_class^-0.5
```

This provides additional exposure to minority classes without forcing completely uniform sampling.

---

## 📊 Feature Engineering

### ECG Signal Features

Each ECG beat is transformed into three channels:

* Original signal
* First derivative
* Second derivative

Each channel is standardized on a per-recording basis.

### Physiological / Morphological Features

The tabular branch uses **18 engineered features**, including:

* Pre-RR interval
* Post-RR interval
* RR ratio
* Missing-value indicators
* RR-based logarithmic features
* R-peak position
* Peak polarity
* Peak amplitude
* QRS width
* QRS area
* Pre-QRS energy
* QRS energy
* Post-QRS energy
* Energy ratios

Robust scaling using median and IQR is fitted only on the corresponding training data.

---

## 🛠️ Data Augmentation

The training pipeline applies ECG-specific augmentation to improve robustness against recording variations.

| Augmentation       | Purpose                            |
| ------------------ | ---------------------------------- |
| Temporal shift     | Simulates beat alignment variation |
| Gain variation     | Handles amplitude differences      |
| Gaussian noise     | Improves noise robustness          |
| Polarity inversion | Handles inverted ECG leads         |

The polarity augmentation also updates the corresponding peak-sign feature to keep the signal and tabular representations consistent.

---

## 🔄 Test-Time Augmentation

During inference, predictions are generated using temporal shifts:

```text
[-3, 0, +3]
```

The resulting probability distributions are averaged before selecting the final class.

This reduces sensitivity to small temporal alignment differences in ECG recordings.

---

## 🧪 Training Strategy

The training pipeline uses:

* **PyTorch**
* AdamW optimizer
* Learning rate: `1e-3`
* Weight decay: `1e-4`
* Batch size: `256`
* ReduceLROnPlateau scheduler
* Early stopping
* Gradient clipping
* Stratified 5-fold cross-validation
* Multiple random seeds for full-data training

The implemented network contains approximately **518K trainable parameters**.

---

## 📈 Experimental Results

The 5-fold validation experiment produced the following overall out-of-fold metrics:

| Metric                  |     Result |
| ----------------------- | ---------: |
| Macro F1                | **0.9154** |
| Class 0 F1              |      0.992 |
| Class 1 F1              |      0.945 |
| Class 2 F1              |      0.990 |
| Class 3 F1              |      0.735 |
| Prior-adjusted Macro F1 | **0.9170** |

The fold-level macro F1 scores ranged from **0.8931 to 0.9296**, demonstrating the model's performance across different stratified validation splits.

> **Note:** The validation folds are stratified by class rather than grouped by recording. Therefore, the OOF score should not be interpreted as a completely recording-independent generalization estimate.

---

## 🏗️ Project Pipeline

```text
Official ECG Dataset
        │
        ▼
Data Validation
        │
        ▼
ECG Signal Preprocessing
        │
        ├───────────────┐
        ▼               ▼
Signal Features    Tabular Features
        │               │
        ▼               ▼
3-Channel CNN          MLP
        │               │
        └───────┬───────┘
                ▼
          Late Fusion
                │
          192-D Feature
                │
       ┌────────┴────────┐
       │                 │
 Classification       CORAL
       │              Alignment
       └────────┬────────┘
                ▼
          Model Training
                │
       ┌────────┴────────┐
       ▼                 ▼
   5-Fold CV        Full Data Training
       │                 │
       └────────┬────────┘
                ▼
        Test-Time Augmentation
                │
                ▼
        Final Class Prediction
```

---

## 💻 Tech Stack

### Machine Learning

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn

### Deep Learning

* 1D Convolutional Neural Networks
* Residual Blocks
* Attention Pooling
* Multi-Branch Neural Networks
* Feature Fusion
* Focal Loss
* CORAL Domain Adaptation

### Model Evaluation

* Stratified K-Fold Cross-Validation
* Macro F1 Score
* Per-class F1 Score
* Confusion Matrix
* Out-of-Fold Predictions
* Test-Time Augmentation

### Hardware

* CUDA-enabled GPU
* PyTorch CUDA acceleration

---

## 📁 Project Structure

```text
ECG-Arrhythmia-Classification/
│
├── notebook/
│   └── experiment_v21.ipynb
│
├── reports/
│   ├── cv_metrics.json
│   ├── candidate_manifest.json
│   └── oof_predictions/
│
├── models/
│   └── trained_models/
│
├── requirements.txt
├── README.md
└── LICENSE
```

*Update the structure above to match the actual files in your repository.*

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ECG-Arrhythmia-Classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Place the official dataset in the expected directory structure:

```text
nppe2_dataset/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### 4. Run the experiment

```bash
python experiment_v21.py
```

For notebook-based execution, open:

```bash
jupyter notebook
```

and run the experiment notebook.

---

## 🔍 Experiment Modes

The implementation supports multiple execution modes:

| Mode       | Purpose                                      |
| ---------- | -------------------------------------------- |
| `AUDIT`    | Validate project structure and configuration |
| `SMOKE`    | Quick end-to-end verification                |
| `FINAL_CV` | Complete cross-validation and experiment     |

A quick local verification mode is also available for testing the pipeline before running the full experiment.

---

## 🔐 Reproducibility & Compliance

The experiment was designed with reproducibility and data-use constraints in mind:

* Fresh random initialization for every model
* No pretrained weights
* No external datasets
* No test labels
* Training uses official NPPE-2 training data
* Tabular scaling is fitted using training data only
* Deterministic preprocessing
* SHA-256 auditing of generated prediction files
* Candidate prediction files are validated before use
* No Kaggle submission API is called by the notebook

---

## 🎯 Key Takeaways

This project demonstrates how combining **deep ECG signal representations, engineered physiological features, domain alignment, imbalance-aware learning, augmentation, and ensemble inference** can create a robust pipeline for multi-class arrhythmia classification.

The project particularly investigates **recording-level distribution shift**, using CORAL to align feature distributions between labelled training data and unlabeled target data.

---

## 👨‍💻 Skills Demonstrated

* Deep Learning
* PyTorch
* ECG Signal Processing
* 1D CNN Architecture Design
* Residual Neural Networks
* Attention Mechanisms
* Feature Engineering
* Multi-modal Feature Fusion
* Domain Adaptation
* CORAL
* Imbalanced Classification
* Focal Loss
* Cross-Validation
* Model Evaluation
* Data Augmentation
* Test-Time Augmentation
* GPU Model Training
* Experiment Reproducibility

---

## 📌 Project Status

**Completed experimental implementation — Version 21**

The current version focuses on evaluating active CORAL-based domain alignment within a dual-branch ECG classification architecture.
