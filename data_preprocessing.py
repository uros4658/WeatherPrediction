import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Handle missing values, outliers, and encode categorical variables if necessary
    # This is just a placeholder. You'll need to fill in with your actual preprocessing steps.
    df = df.dropna()

    # Split the data into a training set and a test set
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df
