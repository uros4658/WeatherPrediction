from keras.models import load_model
import joblib

# Load the model
model = joblib.load('trained_model.pkl')

# Print the model's architecture
model.summary()
