import os
from pathlib import Path
from dotenv import load_dotenv  # <-- NEW: Imports the dotenv library

def download_kaggle_dataset():
    project_root = Path(__file__).resolve().parent.parent
    raw_data_dir = project_root / "data" / "raw"
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # <-- NEW: Load the secret key from the .env file BEFORE importing kaggle
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path) 
    
    try:
        import kaggle
    except OSError as e:
        print("ERROR: Kaggle API token not found.")
        print("Please ensure your .env file has the KAGGLE_API_TOKEN set.")
        return

    dataset_name = "mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd"
    print(f"Downloading dataset '{dataset_name}'...")
    
    kaggle.api.dataset_download_files(
        dataset_name, 
        path=str(raw_data_dir), 
        unzip=True
    )
    
    print("\n✅ Download and extraction complete!")
    print(f"Check the contents of {raw_data_dir} to see your CSV files.")

if __name__ == "__main__":
    download_kaggle_dataset()