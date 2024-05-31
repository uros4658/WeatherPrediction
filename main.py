import pandas as pd
import data_preprocessing
import model_training
import model_evaluation

# Load the data
df = pd.read_csv('weather_data.csv')

# Preprocess the data
X_train, X_test, y_train, y_test = data_preprocessing.preprocess_data(df, window_size=5)

# Train the models
rf_model, lstm_model = model_training.train_model(X_train, y_train)

# Evaluate the models
rf_mae, lstm_mae = model_evaluation.evaluate_model(rf_model, lstm_model, X_test, y_test)

print(f"Random Forest Mean Absolute Error: {rf_mae}")
print(f"LSTM Mean Absolute Error: {lstm_mae}")
