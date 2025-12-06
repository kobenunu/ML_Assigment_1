import pandas as pd
import os
import logging
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(module)s.%(funcName)s:: %(message)s",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

def load_data(dataset_name):
    """Loads a dataset from the regression/datasets directory."""
    logger.info(f"Loading '{dataset_name}' dataset...")
    path = f"./regression/datasets/{dataset_name}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)

def save_data(df, dataset_name):
    """Saves a processed dataset to the regression/datasets/processed directory."""
    logger.info(f"Saving processed '{dataset_name}' dataset...")
    processed_dir = './regression/datasets/processed'
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(f"{processed_dir}/{dataset_name}_processed.csv", index=False)

def preprocess_dataset(dataset_name, target_column, drop_columns=None, cat_cols=None, actions=None):
    """Generic function to preprocess a dataset."""
    df = load_data(dataset_name)

    if drop_columns:
        df = df.drop(columns=drop_columns, errors='ignore')

    if actions:
        for action in actions:
            df = action(df)

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    if target_column in df.columns:
        df = df.rename(columns={target_column: 'target'})
    else:
        raise KeyError(f"Target column '{target_column}' not found in the DataFrame")

    save_data(df, dataset_name)
    return df

# --- Custom Action Functions ---

def clean_laptop_data(df):
    """Cleans Ram and Weight columns in the laptop dataset."""
    df = df.copy()
    if 'Ram' in df.columns:
        df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    if 'Weight' in df.columns:
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
    return df

def map_gender(df):
    """Maps gender to numerical values."""
    df = df.copy()
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    return df

# --- Specific Preprocessing Functions ---

def preprocess_fiat_data():
    """Preprocesses the Fiat 500 dataset."""
    logger.info("--- Preprocessing Fiat 500 Dataset ---")
    preprocess_dataset(
        dataset_name='another-fiat-500-dataset-1538-rows',
        target_column='price',
        drop_columns=['lat', 'lon'],
        cat_cols=['model']
    )

def preprocess_food_delivery_data():
    """Preprocesses the food delivery time dataset."""
    logger.info("--- Preprocessing Food Delivery Dataset ---")
    preprocess_dataset(
        dataset_name='food-delivery-time-prediction',
        target_column='Delivery_Time_min',
        drop_columns=['Order_ID'],
        cat_cols=['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
    )

def preprocess_laptop_price_data():
    """Preprocesses the laptop price dataset."""
    logger.info("--- Preprocessing Laptop Price Dataset ---")
    preprocess_dataset(
        dataset_name='laptop-price-prediction-dataset',
        target_column='Price',
        drop_columns=['Unnamed: 0'],
        cat_cols=['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys'],
        actions=[clean_laptop_data]
    )

def preprocess_second_hand_car_data():
    """Preprocesses the second-hand car dataset."""
    logger.info("--- Preprocessing Second-Hand Car Dataset ---")
    preprocess_dataset(
        dataset_name='second-hand-used-cars-data-set-linear-regression',
        target_column='current price',
        drop_columns=['v.id']
    )

def preprocess_bank_churn_data_regression():
    """Preprocesses the bank churn dataset for regression."""
    logger.info("--- Preprocessing Bank Churn Dataset for Regression ---")
    preprocess_dataset(
        dataset_name='bank_churn',
        target_column='estimated_salary',
        drop_columns=["customer_id", "churn"],
        cat_cols=["country"],
        actions=[map_gender]
    )

def main():
    """Runs all preprocessing steps for the regression datasets."""
    logger.info("Starting preprocessing for all regression datasets...")
    preprocess_fiat_data()
    preprocess_food_delivery_data()
    preprocess_laptop_price_data()
    preprocess_second_hand_car_data()
    preprocess_bank_churn_data_regression()
    logger.info("All regression datasets have been processed and saved.")

if __name__ == "__main__":
    main()