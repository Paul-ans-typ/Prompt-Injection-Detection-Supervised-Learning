## Project Overview: Detecting LLM Prompt Injection

As Large Language Models (LLMs) become the backbone of modern software services—from customer support bots to document analyzers—they face a growing security threat: **Prompt Injection Attacks**. These attacks involve malicious users crafting inputs designed to hijack the model's logic, bypass safety filters, or leak sensitive data.

This project focuses on building a robust defense layer for LLM-integrated applications. Our team is developing and comparing two distinct approaches to identify and block these attacks before they ever reach the model:

1. **Baseline Approach:** A lightweight, high-speed **TF-IDF + Logistic Regression** model.
2. **Advanced Approach:** A fine-tuned **BERT-based Transformer** classifier for deep semantic understanding.

### Key Objectives

* **Security for Software Services:** Tailoring detection specifically for real-world LLM deployments.
* **Supervised Learning:** Utilizing labeled datasets (like MalPID) to train binary classifiers (Benign vs. Malicious).
* **Comparative Analysis:** Evaluating the trade-offs between "traditional ML" and "deep learning" in terms of accuracy, latency, and computational cost.
* **Actionable Metrics:** Measuring success through Precision, Recall, and F1-score to ensure minimal "false positives" (blocking legitimate users).

### 🛠️ Tech Stack

* **Language:** Python
* **ML/DL:** PyTorch, Hugging Face Transformers, Scikit-learn
* **Data:** Pandas, NumPy, TF-IDF Tokenization
* **Environment:** Jupyter Notebook / Google Colab

---

### The Team

**Lead:** Maulik Jadav | **Deputy:** Vatsal Nirmal
**Contributors:** Josie Lorenz, Anish Paul Singareddy, Mani Sindhu Vemaraju, Shivasmaran Rajashekar, Tawanda Nyagumbo, Mohammad Jani Basha Shaik
