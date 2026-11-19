from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import csv

# Configuration
FLASK_APP_URL = "http://127.0.0.1:5000/"  # Your Flask app's URL
URL_LIST_FILE = "urls_to_test.txt"      # File containing URLs to test (one per line)
OUTPUT_CSV_FILE = "test_results.csv"     # File to save the test results

def test_url(driver, url):
    driver.get(FLASK_APP_URL)
    url_input = driver.find_element(By.ID, "url")
    submit_button = driver.find_element(By.XPATH, "//input[@type='submit']")

    url_input.send_keys(url)
    submit_button.click()
    time.sleep(1)  # Give the app time to process and display the result

    try:
        prediction_element = driver.find_element(By.CLASS_NAME, "prediction")
        prediction = prediction_element.text.split(": ")[1].strip()
    except Exception:
        prediction = "Error: Prediction not found"

    try:
        features_element = driver.find_element(By.CLASS_NAME, "features")
        features_text = features_element.text.replace("\n", "; ")
    except Exception:
        features_text = "Error: Features not found"

    return {"url": url, "prediction": prediction, "features": features_text}

if __name__ == "__main__":
    driver = webdriver.Firefox()  # Or webdriver.Chrome()

    try:
        with open(URL_LIST_FILE, "r") as f:
            urls_to_test = [line.strip() for line in f if line.strip()]

        test_results = []
        for url in urls_to_test:
            print(f"Testing: {url}")
            result = test_url(driver, url)
            test_results.append(result)
            print(f"  Prediction: {result['prediction']}")

        # Save results to CSV
        with open(OUTPUT_CSV_FILE, "w", newline="") as csvfile:
            fieldnames = ["url", "prediction", "features"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(test_results)

        print(f"\nTest results saved to: {OUTPUT_CSV_FILE}")

    finally:
        driver.quit()