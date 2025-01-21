# Blood Pressure Monitoring from Naturally Recorded Speech Sounds: Advancements and Future Prospects

## Authors
- **Fikret Arı** (Ankara University, Electrical & Electronics Engineering, Ankara, Turkey)
- **Haydar Ankışhan** (Ankara University, Stem Cell Institute, Ankara, Turkey)
- **Blaise B. Frederick** (Mclean Hospital, MIC, Harvard University, USA)
- **Lia M. Hocke** (Harvard Medical School, Department of Psychiatry, USA)
- **Sinem B. Erdoğan** (Acıbadem University, Biomedical Engineering, İstanbul, Turkey)

**Corresponding Author**: Haydar Ankışhan, ankishan@ankara.edu.tr

---

## Abstract
This repository hosts the code and dataset for the study **"Blood Pressure Monitoring from Naturally Recorded Speech Sounds: Advancements and Future Prospects"**, which introduces an artificial intelligence-based method to predict blood pressure (BP) using speech recordings in everyday scenarios. The system utilizes smartphones, avoiding the need for additional hardware.

Key achievements:
- Classification accuracy: **92.13% for systolic BP**, **94.49% for diastolic BP** using Adaptive Synthetic Sampling (ADASYN).
- Methodology: Hyperparameter-tuned ML models, Vowel Onset Point (VOP) detection, and a 1x43-D statistical feature vector.
- Applications: Telehealth, IoT devices, and remote BP monitoring.

---

## Key Features
- **Novelty**: Predict BP levels from natural speech without cuff-based measurements.
- **Techniques**: Synthetic Minority Oversampling Technique (SMOTE) and ADASYN for data imbalance.
- **Feature Extraction**: Statistical features (MFCC, Skewness, Kurtosis, etc.) and demographic data.
- **Vowel Onset Point Detection**: Automatic detection for speech segments rich in physiological information.

---

## Dataset
The dataset comprises speech recordings and physiological data from **95 participants**:
- **Participants**: 40 females, 55 males (Age: 20–70 years).
- **Recording Statistics**:
  - Systolic BP (SBP): {Min: 91 mmHg, Max: 153 mmHg}.
  - Diastolic BP (DBP): {Min: 35 mmHg, Max: 98 mmHg}.
  - Exclusion criteria: Neurological issues, caffeine/alcohol intake, etc.

---

## Methodology
### Data Preprocessing
1. **Vowel Onset Points (VOP)**: Detected from sentences with 20 ms frames, ensuring optimal feature extraction.
2. **Feature Extraction**: Statistical and spectral features derived from speech recordings, focusing on vowels due to their high spectral energy.

### Machine Learning Pipeline
1. **Oversampling**: SMOTE and ADASYN to balance the dataset.
2. **Feature Selection**: Using techniques like SULOV and Recursive XGBoost for dimensionality reduction.
3. **Classification Models**: Hyperparameter-tuned classifiers to differentiate between normal and high BP.

---

## Usage
### Prerequisites
- Python 3.8+
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `matplotlib`

### Installation
Clone the repository:
```bash
git clone https://github.com/your-repo-name/blood-pressure-speech-monitoring.git
cd blood-pressure-speech-monitoring
