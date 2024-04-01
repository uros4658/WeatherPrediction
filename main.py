import pandas as pd
import data_preprocessing
import model_training
import model_evaluation

# Load the data
df = pd.read_csv('weather_data.csv')

# Preprocess the data
train_df, test_df = data_preprocessing.preprocess_data(df)

# Train the model
model = model_training.train_model(train_df)

# Evaluate the model
mae = model_evaluation.evaluate_model(model, test_df)

print(f"Mean Absolute Error: {mae}")
