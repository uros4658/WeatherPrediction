from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import joblib


def train_model(train_df):
    # Define your features and target variable
    features = ['Humidity', 'Wind Speed']
    target = 'Temperature'

    # Prepare your data
    X = train_df[features]
    Y = train_df[target]

    # Define the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model
    rf_model.fit(X, Y)

    # Save the trained model
    joblib.dump(rf_model, 'rf_trained_model.pkl')

    # Now let's train the LSTM model
    # Prepare your data
    X_lstm = X.values.reshape(-1, 1, len(features))
    Y_lstm = Y.values

    # Set the number of features in your data
    n_features = len(features)

    # Define the LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(None, n_features)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')

    # Fit the model
    lstm_model.fit(X_lstm, Y_lstm, epochs=200, verbose=0)

    # Save the trained model
    lstm_model.save('lstm_trained_model.h5')

    return rf_model, lstm_model
