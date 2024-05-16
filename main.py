import pandas as pd
import data_preprocessing
import model_training
import model_evaluation

# Load the data
df = pd.read_csv('weather_data.csv')

# Preprocess the data
train_df, test_df = data_preprocessing.preprocess_data(df)

# Train the model
rf_model, lstm_model = model_training.train_model(train_df)

# Evaluate the model
rf_mae, lstm_mae = model_evaluation.evaluate_model(rf_model, lstm_model, test_df)

print(f"Random Forest Mean Absolute Error: {rf_mae}")
print(f"LSTM Mean Absolute Error: {lstm_mae}")
