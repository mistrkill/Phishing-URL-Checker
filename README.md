Phishing URL Detector

A Random Forest model that classifies URLs as phishing or benign, with a simple Flask web app for checking links live.

Trained on the PhiUSIIL phishing URL dataset (~235k URLs) with 99.9% accuracy on a held-out test set (27 errors out of 47k URLs). Features come from the URL itself (length, special characters, HTTPS, etc.) and from the page content (title, favicon, forms, etc.). I removed the dataset’s URLSimilarityIndex feature because it leaked the label, and pruned redundant features using a correlation matrix.

Includes:

data_process.py, model_train.py: dataset processing and training
features.py, predict.py: feature extraction and prediction for any URL
app.py: Flask interface
testing/autotester.py: Selenium script that runs a list of URLs through the app and saves results to CSV

Built for the Fontys AI for Society minor.