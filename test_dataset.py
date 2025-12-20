import pandas as pd
import os

# Check if file exists
dataset_path = 'dataset/IMDBDataset.csv'
if os.path.exists(dataset_path):
    print(f" Dataset file found at: {dataset_path}")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    print(f"\n Dataset loaded successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sentiment distribution:")
    print(df['sentiment'].value_counts())
    print(f"\n  Sample review:")
    print(f"  {df['review'].iloc[0][:200]}...")
else:
    print(f" Dataset file not found at: {dataset_path}")
    print(f"  Current working directory: {os.getcwd()}")
