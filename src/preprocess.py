import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    input_path = 'data/raw/MPDD.csv'
    output_dir = 'data/processed/'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Stratified split to maintain the 50/50 class balance
    # Split 1: 80% Train, 20% Temp (for Val and Test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.20, 
        random_state=42, 
        stratify=df['isMalicious']
    )
    
    # Split 2: Split the 20% Temp evenly into 10% Validation and 10% Test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=42, 
        stratify=temp_df['isMalicious']
    )
    
    print("Saving splits to processed directory...")
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Train set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()