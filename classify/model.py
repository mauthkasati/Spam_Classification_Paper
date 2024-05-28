import joblib
import os

# Load the model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'spam_classifier_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

model = joblib.load(model_path)