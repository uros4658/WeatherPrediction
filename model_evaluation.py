from keras.models import load_model
from sklearn.metrics import mean_absolute_error

def evaluate_model(model, test_df):
    # Define your features and target variable
    features = ['Humidity', 'Wind Speed']
    target = 'Temperature'

    # Prepare your test data
    # X_test should be a 3D array of shape (n_samples, n_timesteps, n_features)
    # Y_test should be a 2D array of shape (n_samples, n_outputs)
    X_test = test_df[features].values.reshape(-1, 1, len(features))
    Y_test = test_df[target].values

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate the Mean Absolute Error of the predictions
    mae = mean_absolute_error(Y_test, predictions)

    return mae
