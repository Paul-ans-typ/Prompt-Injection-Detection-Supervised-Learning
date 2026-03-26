# Detection of Prompt Injection Attacks in LLM-Based Software Services

## Overview

Large Language Models (LLMs) embedded in software services (like customer support chatbots and code-generation tools) introduce new security risks, most notably **prompt injection attacks**. In these attacks, malicious users craft inputs that override system instructions, bypass safety controls, or extract sensitive information.

This project implements an offline, supervised machine learning pipeline to detect these text-based prompt injection attacks. It establishes a traditional machine learning baseline and compares it against a fine-tuned Deep Learning approach to classify prompts as either benign or malicious.

## Dataset

This project uses the **Malicious Prompt Detection Dataset (MPDD)**.

* **Size:** 39,234 text prompts.
* **Class Balance:** Perfectly balanced (50% Benign, 50% Malicious).
* **Format:** Binary classification (`isMalicious` label).

## Repository Structure

```text
prompt-injection-detection/
├── .gitignore
├── .env.example             # Template for environment variables (Kaggle API key)
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
│
├── data/                    # Ignored by Git
│   ├── raw/                 # Raw MPDD.csv dataset
│   └── processed/           # 80/10/10 Train/Val/Test splits
│
├── models/                  # Ignored by Git (Saved model weights & vectorizers)
│
├── notebooks/               # Jupyter notebooks for EDA and evaluation visuals
│
└── src/                     # Core Python scripts
    ├── download_data.py     # Script to securely download the dataset
    ├── preprocess.py        # Data cleaning and train/test splitting
    ├── train_baseline.py    # TF-IDF + Logistic Regression training
    ├── train_bert.py        # BERT fine-tuning using PyTorch
    └── evaluate.py          # Model comparison and confusion matrix generation

```

## Models Implemented

1. **Baseline Model (TF-IDF + Logistic Regression):** Chosen for its lightweight footprint, interpretability, and speed. Uses a 10,000 max-feature TF-IDF vectorizer.
2. **Deep Learning Model (BERT):** A `bert-base-uncased` transformer model fine-tuned for sequence classification. Trained using PyTorch with an AdamW optimizer.

## Results & Evaluation

Both models were evaluated on an unseen test set of 3,924 prompts.

* **Logistic Regression Baseline:**
* Achieved ~95.2% overall accuracy.
* **False Negatives:** 145
* **False Positives:** 44


* **Fine-Tuned BERT:**
* Achieved ~97.5% overall accuracy.
* **False Negatives:** 37
* **False Positives:** 62



**Key Takeaway:** While the Logistic Regression baseline performed surprisingly well, the BERT model is significantly better suited for security contexts. BERT reduced False Negatives (actual malicious prompts that slipped through undetected) by nearly 75% compared to the baseline. In a security environment, minimizing False Negatives is critical to preventing successful prompt injection attacks.

## How to Run

1. **Environment Setup:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt

```


2. **Download Data:**
You will need your own Kaggle API key to download the dataset. Set up your `.env` file with your credentials (refer to `.env.example`), then run:
```bash
python src/download_data.py

```


3. **Data Preprocessing:**
Process the raw data and generate the train/val/test splits:
```bash
python src/preprocess.py

```


4. **Train Models:**
```bash
python src/train_baseline.py
python src/train_bert.py

```


5. **Evaluate:**
```bash
python src/evaluate.py

```



## Future Work

As this project expands, planned future improvements include:

* **Real-time API Integration:** Wrapping the trained BERT model in a FastAPI or Flask application to serve as a middleware security layer for an actual LLM application.
* **Testing on Unseen Attack Vectors:** Evaluating the model against newer, out-of-distribution jailbreak techniques (e.g., base64 encoding attacks, multi-language bypasses).
* **Model Quantization:** Reducing the memory footprint of the BERT model using techniques like ONNX or TensorRT to decrease inference latency in production.
* **Exploring Smaller LLMs for Detection:** Testing local, uncensored 7B/8B parameter models as few-shot classifiers to see if they outperform fine-tuned BERT representations.