from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

def train_model(train_df):
    # Define your features and target variable
    features = ['Humidity', 'Wind Speed']
    target = 'Temperature'

    # Prepare your data
    X = train_df[features].values.reshape(-1, 1, len(features))
    Y = train_df[target].values

    # Set the number of features in your data
    n_features = len(features)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(None, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fit the model
    model.fit(X, Y, epochs=200, verbose=0)

    # Save the trained model
    model.save('trained_model.h5')  # Change this line

    return model
