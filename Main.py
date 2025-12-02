import logging
from rich.logging import RichHandler
logger = logging.getLogger(__name__)
import pandas as pd
import os
from run_cross_validation import run_cross_validation
from run_sensitivity_analysis import run_sensitivity_analysis
from GenerateVisualizations import generate_all_visualizations

def map_gender(df):
    df = df.copy()
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    return df

def map_sex(df):
    df = df.copy()
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
    return df

def map_blood_pressure(df):
    df = df.copy()

    if 'Blood Pressure' in df.columns:
        df[['Systolic', 'Diastolic']] = (
            df['Blood Pressure'].str.split('/', expand=True).astype(int)
        )
        df = df.drop(columns=['Blood Pressure'])

    return df

def load_data(dataset_name):
    logger.info(f"Loading {dataset_name} dataset...")
    # Placeholder for actual data loading logic
    df = pd.read_csv(f"./datasets/{dataset_name}.csv")
    return df

def save_data(df, dataset_name):
    logger.info(f"Saving processed {dataset_name} dataset...")
    processed_dir = './datasets/processed'
    df.to_csv(f"{processed_dir}/{dataset_name}_processed.csv", index=False)

def preprocess_dataset(dataset_name, drop_columns: list, cat_cols: list, target_column: str, actions: list):
    df = load_data(dataset_name)

    # Ensure columns to drop exist in the DataFrame
    df = df.drop(columns=drop_columns)

    # Ensure categorical columns exist in the DataFrame
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Apply actions and ensure they return a DataFrame
    if len(actions) > 0:
        for action in actions:
            df = action(df)

    # Ensure the target column exists before renaming
    if target_column in df.columns:
        df = df.rename(columns={target_column: 'target'})
    else:
        raise KeyError(f"Target column '{target_column}' not found in the DataFrame")

    save_data(df, dataset_name)
    
def preprocess_drug_consumption_data(dataset_name):
    preprocess_dataset(dataset_name,
                       drop_columns=['ID'],
                       cat_cols =["Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Choc", "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine", "Legalh", "LSD", "Meth", "Mushrooms", "Nicotine", "Semer", "VSA"],
                       target_column='Cannabis',
                       actions=[])

def preprocess_heart_attack_data(dataset_name):
    preprocess_dataset(dataset_name,
                       drop_columns=["Patient ID", "Continent", "Hemisphere"],
                       cat_cols =["Country","Diet"],
                       target_column="Heart Attack Risk", actions = [map_sex, map_blood_pressure])
    
def preprocess_bank_churn_data(dataset_name):
    preprocess_dataset(dataset_name,
                       drop_columns=["customer_id"],
                       cat_cols =["country"],
                       target_column="churn", actions = [map_gender])
    
def preprocess_weather_data(dataset_name):
    preprocess_dataset(dataset_name,
                       drop_columns=[],
                       cat_cols =["Location", "Season", "Cloud Cover"],
                       target_column="Weather Type",
                       actions=[])
    
def preprocess_diabetes_data(dataset_name):
    preprocess_dataset(dataset_name,
                       drop_columns=[],
                       cat_cols=[],
                       target_column="Diabetes",
                       actions=[])
    

def preprocess_data():
    preprocess_bank_churn_data('bank_churn')
    preprocess_drug_consumption_data('drug_consumption')
    preprocess_heart_attack_data('heart_attack')
    preprocess_weather_data('weather')
    preprocess_diabetes_data('diabetes')

def get_all_files_in_path(path):
    """
    Get all files in the given directory path.

    Args:
        path (str): The directory path to search for files.

    Returns:
        list: A list of file paths in the directory.
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,
    format="%(module)s.%(funcName)s:: %(message)s",
    handlers=[RichHandler()]
    )
    logging.getLogger("sklearn").setLevel(logging.INFO)
    logging.getLogger("pandas").setLevel(logging.INFO)

    processed_dir = './datasets/processed'
    os.makedirs(processed_dir, exist_ok=True)
    preprocess_data() # Uncomment this line if preprocessing is needed
    
    #run_cross_validation()
    run_sensitivity_analysis()
    #generate_all_visualizations()



