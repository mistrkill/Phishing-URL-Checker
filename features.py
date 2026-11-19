import pandas as pd
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import re

# Helper for string similarity for Domain/URLTitleMatchScore
def jaccard_similarity_word(s1, s2):
    """Calculates Jaccard similarity between two strings based on words."""
    words1 = set(re.findall(r'\b\w+\b', s1.lower()))
    words2 = set(re.findall(r'\b\w+\b', s2.lower()))
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0

def calculate_continuation_rate(url):
    if not url:
        return 0
    max_consecutive = 0
    current_consecutive = 1
    for i in range(len(url) - 1):
        if url[i] == url[i + 1]:
            current_consecutive += 1
        else:
            max_consecutive = max(max_consecutive, current_consecutive)
            current_consecutive = 1
    max_consecutive = max(max_consecutive, current_consecutive)
    return max_consecutive / len(url) if len(url) > 0 else 0

def extract_url_features(url):
    parsed_url = urlparse(url)
    hostname = parsed_url.netloc
    path = parsed_url.path

    features = {}

    # --- URL-based features (from data_process.py's columns_to_keep) ---
    features['URLLength'] = len(url)
    features['Domain'] = hostname # Kept for reference, dropped before model prediction
    features['DomainLength'] = len(hostname) if hostname else 0
    features['TLD'] = hostname.split('.')[-1].lower() if hostname and '.' in hostname else ''
    features['CharContinuationRate'] = calculate_continuation_rate(url)
    features['SpacialCharRatioInURL'] = sum(1 for char in url if not char.isalnum() and char not in '=%?&./:-_') / len(url) if len(url) > 0 else 0
    features['DegitRatioInURL'] = sum(1 for char in url if char.isdigit()) / len(url) if len(url) > 0 else 0
    features['IsHTTPS'] = 1 if parsed_url.scheme == 'https' else 0

    # Initialize content-based features with defaults (in case fetching fails)
    # These must match the features from data_process.py's columns_to_keep
    content_based_defaults_init = {
        'LineOfCode': 0, 'HasTitle': 0, 'DomainTitleMatchScore': 0, 
        'HasFavicon': 0, 'Robots': 0, 'IsResponsive': 0, 'HasDescription': 0,
        'NoOfiFrame': 0, 'HasExternalFormSubmit': 0, 'HasSocialNet': 0,
        'HasSubmitButton': 0, 'HasHiddenFields': 0, 'HasPasswordField': 0,
        'Bank': 0, 'Pay': 0, 'HasCopyrightInfo': 0, 'NoOfImage': 0, 'NoOfCSS': 0,
        'NoOfJS': 0, 'NoOfSelfRef': 0, 'NoOfExternalRef': 0
    }
    features.update(content_based_defaults_init)

    try:
        response = requests.get(url, timeout=5, allow_redirects=True, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Content-based features (matching data_process.py's columns_to_keep)
        features['LineOfCode'] = len(response.text.splitlines())
        
        page_title = soup.title.string.strip() if soup.title and soup.title.string else ''
        features['HasTitle'] = 1 if page_title else 0
        features['Title'] = page_title # Kept for reference, dropped before model prediction

        # Implemented DomainTitleMatchScore with Jaccard Similarity (as per discussion)
        features['DomainTitleMatchScore'] = jaccard_similarity_word(hostname, page_title)
        # URLTitleMatchScore was removed from the final columns_to_keep in data_process.py, so don't calculate here.

        features['HasFavicon'] = 1 if soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon') else 0
        
        robots_url = urljoin(f"{parsed_url.scheme}://{parsed_url.netloc}", 'robots.txt')
        try:
            robots_response = requests.head(robots_url, timeout=2)
            features['Robots'] = 1 if robots_response.status_code == 200 else 0
        except requests.exceptions.RequestException:
            features['Robots'] = 0

        features['IsResponsive'] = 1 if soup.find('meta', attrs={'name': 'viewport'}) else 0
        features['HasDescription'] = 1 if soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'}) else 0
        
        # NoOfPopup was removed from final features.
        features['NoOfiFrame'] = len(soup.find_all('iframe'))
        features['HasExternalFormSubmit'] = 1 if soup.find('form', action=lambda x: x and not urlparse(x).netloc == parsed_url.netloc) else 0
        
        social_keywords = ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com', 't.me'] # Refined social media domains
        features['HasSocialNet'] = 1 if any(urlparse(a.get('href', '')).netloc in k for a in soup.find_all('a', href=True) for k in social_keywords) else 0

        features['HasSubmitButton'] = 1 if soup.find('input', type='submit') or soup.find('button', type='submit') else 0
        features['HasHiddenFields'] = len(soup.find_all('input', type='hidden'))
        features['HasPasswordField'] = 1 if soup.find('input', type='password') else 0

        features['Bank'] = 1 if re.search(r'\b(bank|banco|banking|financial|credit)\b', response.text.lower()) else 0
        features['Pay'] = 1 if re.search(r'\b(pay|payment|checkout|invoice|bill)\b', response.text.lower()) else 0
        # Crypto was removed from final features.
        
        features['HasCopyrightInfo'] = 1 if re.search(r'©|copyright', response.text, re.IGNORECASE) else 0
        features['NoOfImage'] = len(soup.find_all('img'))
        features['NoOfCSS'] = len(soup.find_all('link', rel='stylesheet'))
        features['NoOfJS'] = len(soup.find_all('script'))
        features['NoOfSelfRef'] = len(soup.find_all('a', href=lambda x: x and x.startswith('#')))
        features['NoOfEmptyRef'] = len(soup.find_all('a', href=''))
        features['NoOfExternalRef'] = len(soup.find_all('a', href=lambda x: x and urlparse(x).netloc and urlparse(x).netloc != parsed_url.netloc))
        
        # NoOfURLRedirect was removed from data_process.py columns_to_keep
        # NoOfSelfRedirect was removed from data_process.py columns_to_keep

        # TLDLegitimateProb and URLCharProb were removed from final features.
        
    except requests.exceptions.RequestException as e:
        # Fallback for when web fetching fails - only include features that were in final list
        print(f"Error fetching or processing {url}: {e}")
        error_defaults = {
            'LineOfCode': 0, 'HasTitle': 0,
            'DomainTitleMatchScore': 0, 'HasFavicon': 0,
            'Robots': 0, 'IsResponsive': 0, 'HasDescription': 0,
            'NoOfiFrame': 0, 'HasExternalFormSubmit': 0, 'HasSocialNet': 0,
            'HasSubmitButton': 0, 'HasHiddenFields': 0, 'HasPasswordField': 0,
            'Bank': 0, 'Pay': 0, 'HasCopyrightInfo': 0, 'NoOfImage': 0,
            'NoOfCSS': 0, 'NoOfJS': 0, 'NoOfSelfRef': 0, 'NoOfEmptyRef': 0,
            'NoOfExternalRef': 0
        }
        features.update(error_defaults)

    # --- Ensure consistent order and presence of ALL features the model expects ---
    # This list MUST EXACTLY MATCH `columns_to_keep` from `data_process.py`
    # MINUS 'URL', 'Domain', 'Title', 'label'
    final_feature_keys_for_model = [
        'URLLength', 'DomainLength', 'TLD', 'CharContinuationRate',
        'SpacialCharRatioInURL', 'DegitRatioInURL', 'IsHTTPS', 'LineOfCode',
        'HasTitle', 'DomainTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive',
        'HasDescription', 'NoOfiFrame', 'HasExternalFormSubmit', 'HasSocialNet',
        'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay',
        'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef',
        'NoOfExternalRef'
    ]
    
    # Add 'URL' and 'Title' back if they are needed for display but not for model input
    # They are then dropped in predict.py/app.py before passing to the model pipeline.
    # 'Domain' is already part of the features dict and will be dropped.
    reference_columns = ['URL', 'Title']
    
    ordered_features = {k: features.get(k, 0) for k in final_feature_keys_for_model}
    # Add reference columns back to ordered_features if needed for display
    for ref_col in reference_columns:
        if ref_col in features:
            ordered_features[ref_col] = features[ref_col]
        else:
            ordered_features[ref_col] = '' # Default empty string if not calculated

    # Convert to Series and return.
    return pd.Series(ordered_features)


if __name__ == '__main__':
    # Test with example URLs
    test_urls = [
        "https://www.google.com/", # Legitimate
        "https://secure-login.example.com.login.php", # Phishing-like
        "https://www.facebook.com/", # Legitimate, previously misclassified
        "http://www.badsite.xyz/phish.html", # Phishing-like (HTTP, bad TLD)
        "https://docs.microsoft.com/en-us/windows/security/threat-protection/microsoft-defender-antivirus/microsoft-defender-antivirus-in-windows", # Legitimate, long URL
        "http://www.some-legitimate-site.com/broken-link-example" # Legit but might fail request
    ]

    for user_url in test_urls:
        print(f"\n--- Extracting features for: {user_url} ---")
        extracted_features = extract_url_features(user_url)
        print("\nExtracted Features:")
        print(extracted_features.to_string())
        print("-" * 50)