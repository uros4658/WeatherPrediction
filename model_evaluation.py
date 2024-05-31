from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np

def evaluate_model(rf_model, lstm_model, X_test, y_test):
    # Flatten the test data for the Random Forest model
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Make predictions on the test set using the Random Forest model
    rf_predictions = rf_model.predict(X_test_flat)

    # Calculate the Mean Absolute Error of the Random Forest predictions
    rf_mae = mean_absolute_error(y_test, rf_predictions)

    # Make predictions on the test set using the LSTM model
    lstm_predictions = lstm_model.predict(X_test)

    # Calculate the Mean Absolute Error of the LSTM predictions
    lstm_mae = mean_absolute_error(y_test, lstm_predictions)

    return rf_mae, lstm_mae
