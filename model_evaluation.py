from sklearn.metrics import mean_absolute_error
import joblib


def evaluate_model(rf_model, lstm_model, test_df):
    # Define your features and target variable
    features = ['Humidity', 'Wind Speed']
    target = 'Temperature'

    # Prepare your test data
    X_test = test_df[features]
    Y_test = test_df[target].values

    # Make predictions on the test set using the Random Forest model
    rf_predictions = rf_model.predict(X_test)

    # Calculate the Mean Absolute Error of the Random Forest predictions
    rf_mae = mean_absolute_error(Y_test, rf_predictions)

    # Prepare your test data for the LSTM model
    # X_test_lstm should be a 3D array of shape (n_samples, n_timesteps, n_features)
    X_test_lstm = X_test.values.reshape(-1, 1, len(features))

    # Make predictions on the test set using the LSTM model
    lstm_predictions = lstm_model.predict(X_test_lstm)

    # Calculate the Mean Absolute Error of the LSTM predictions
    lstm_mae = mean_absolute_error(Y_test, lstm_predictions)

    return rf_mae, lstm_mae
