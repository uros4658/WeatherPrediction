import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(df, window_size=5):
    # Handle missing values
    df = df.dropna()

    # Create windows of data
    X, y = [], []
    features = ['Humidity', 'Wind Speed']
    target = 'Temperature'
    
    for i in range(len(df) - window_size):
        X.append(df[features].iloc[i:i + window_size].values)
        y.append(df[target].iloc[i + window_size])
    
    X = np.array(X)
    y = np.array(y)

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
