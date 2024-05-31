from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib

def train_model(X_train, y_train):
    # Define your Random Forest model
    features = X_train.shape[2]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Flatten the data for the Random Forest model
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    # Fit the model
    rf_model.fit(X_train_flat, y_train)

    # Save the trained model
    joblib.dump(rf_model, 'rf_trained_model.pkl')

    # Define the LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], features)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')

    # Fit the model
    lstm_model.fit(X_train, y_train, epochs=200, verbose=0)

    # Save the trained model
    lstm_model.save('lstm_trained_model.h5')

    return rf_model, lstm_model
