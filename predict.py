from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import joblib
# The following imports are not strictly needed in predict.py but were in previous version
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer

# Assuming extract_url_features and calculate_continuation_rate are in a file named features.py
from features import extract_url_features, calculate_continuation_rate


def predict_url(url, model_path):
    """
    Predicts if a given URL is phishing using the loaded model.

    Args:
        url (str): The URL to predict.
        model_path (str): Path to the trained model file (.joblib).

    Returns:
        str: "Phishing" or "Benign" based on the model's prediction.
    """
    try:
        loaded_model = joblib.load(model_path)
        print(f"Model loaded successfully from: {model_path}")

        extracted_features = extract_url_features(url)
        print("\nExtracted Features:")
        print(extracted_features)

        features_df = pd.DataFrame([extracted_features])

        # Ensure the order of columns matches the training data
        # Drop columns that are text/identifiers and not used by the model's pipeline
        features_for_prediction = features_df.drop(columns=['URL', 'Domain', 'Title'], errors='ignore')

        # Make the prediction
        prediction = loaded_model.predict(features_for_prediction)

        # Correct interpretation of the prediction (0 = Benign, 1 = Phishing)
        if prediction[0] == 1:
            return "Phishing"
        else:
            return "Benign"

    except FileNotFoundError:
        return f"Error: Model file not found at {model_path}"
    except Exception as e:
        return f"An error occurred during prediction: {e}"

if __name__ == '__main__':
    model_file = 'phishing_detection_model.joblib' # Changed to main model name

    while True:
        user_url = input("Enter the URL to check (or type 'exit' to quit): ")
        if user_url.lower() == 'exit':
            break
        prediction_result = predict_url(user_url, model_file)
        print(f"\nThe predicted label for '{user_url}' is: {prediction_result}\n")