# Detection of Prompt Injection Attacks in LLM-Based Software Services Using Supervised Machine Learning

## Project Overview

Large Language Models (LLMs) are increasingly embedded in software services such as customer support chatbots, document assistants, and code-generation tools. While these systems improve usability and automation, they introduce new security risks, particularly prompt injection attacks.

This project focuses on the offline detection of prompt injection attacks in LLM-integrated software services. By framing this as a binary text classification problem (benign vs. malicious), we implement, evaluate, and compare two supervised machine learning approaches:

1. **Baseline Model:** TF-IDF feature extraction combined with Logistic Regression.
2. **Deep Learning Model:** A pre-trained Transformer model (BERT-based) fine-tuned for sequence classification.

Performance is evaluated using standard classification metrics including accuracy, precision, recall, and F1-score to determine the trade-offs between computational cost and detection robustness.

## Project Team

* **Group Leader:** Maulik Jadav
* **Deputy Leader:** Vatsal Nirmal
* **Group Members:** Josie Lorenz, Anish Paul Singareddy, Mani Sindhu Vemaraju, Shivasmaran Rajashekar, Tawanda Nyagumbo, Mohammad Jani Basha Shaik

## Repository Structure

```text
prompt-injection-detection/
├── .gitignore
├── .env.example             # Template for environment variables (Kaggle API key)
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
│
├── data/                    # Ignored by Git
│   ├── raw/                 # Raw dataset downloaded from Kaggle
│   └── processed/           # Cleaned and tokenized data ready for modeling
│
├── models/                  # Ignored by Git (Saved model weights)
│
├── notebooks/               # Jupyter notebooks for EDA and prototyping
│
└── src/                     # Core Python scripts
    └── download_data.py     # Script to securely download the dataset
```

## Environment Setup

It is highly recommended to run this project within an isolated Conda environment, particularly when deploying on a High-Performance Computing (HPC) cluster.

**1. Clone the repository:**

```bash
git clone git@github.com:YourUsername/prompt-injection-detection.git
cd prompt-injection-detection
```

**2. Create and activate the Conda environment:**

```bash
module load anaconda3  # Use the specific module load command for your HPC
conda create -n prompt_env python=3.10
conda activate prompt_env
```

**3. Install the required dependencies:**

```bash
pip install -r requirements.txt
```

## Data Acquisition

*Note: The project proposal initially cited the MalPID dataset. As the authors did not publicly release that dataset, this implementation utilizes the [Malicious Prompt Detection Dataset (MPDD)](https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd), a balanced dataset of ~40,000 prompts aggregated from real-world prompt injection benchmarks.*

To download the data automatically using the provided script, you must configure your Kaggle API credentials.

**1. Generate a Kaggle API Token:**

* Log into Kaggle.com.
* Navigate to **Settings** -> **API** -> **Generate New Token**.
* Copy the provided token string.

**2. Configure the local environment variables:**

* In the root directory of this project, duplicate the example environment file:

  ```bash
  cp .env.example .env
  ```

* Open the newly created `.env` file and replace the placeholder value with your copied token.
* *Security Note: The `.env` file is explicitly ignored by Git to ensure your credentials remain local and secure.*

**3. Execute the download script:**
Ensure you are in the project root directory and your Conda environment is active, then run:

```bash
python src/download_data.py
```

This script will authenticate via the Kaggle API, download the MPDD dataset, and extract the CSV files into the `data/raw/` directory.

## Next Steps (Current Project Stage)

* Exploratory Data Analysis (EDA) and text preprocessing.
* Implementation of the TF-IDF + Logistic Regression baseline script.
* Development of the BERT fine-tuning script for HPC GPU execution.
