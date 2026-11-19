from flask import Flask, request, render_template
import pandas as pd
import joblib
import predict # Import your predict.py which contains extract_url_features

app = Flask(__name__)
model_path = 'phishing_detection_model.joblib' # Changed to main model name

def load_model(model_path):
    """Loads the trained model."""
    try:
        loaded_model = joblib.load(model_path)
        print(f"Model loaded successfully from: {model_path}")
        return loaded_model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

model = load_model(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    extracted_features = None
    if request.method == 'POST':
        url_to_check = request.form['url']
        if model:
            # Call extract_url_features from the predict module (which imports features.py)
            extracted_features_series = predict.extract_url_features(url_to_check)
            extracted_features = extracted_features_series.to_dict()
            features_df = pd.DataFrame([extracted_features_series])
            
            # Ensure the order of columns matches the training data
            # Drop columns that are text/identifiers and not used by the model's pipeline
            features_for_prediction = features_df.drop(columns=['URL', 'Domain', 'Title'], errors='ignore')
            
            prediction_array = model.predict(features_for_prediction)
            
            # Correct interpretation of the prediction (0 = Benign, 1 = Phishing)
            if prediction_array[0] == 1:
                prediction = "Phishing"
            else:
                prediction = "Benign"
        else:
            prediction = "Model not loaded."
    return render_template('index.html', prediction=prediction, features=extracted_features)

if __name__ == '__main__':
    app.run(debug=True)