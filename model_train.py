import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import numpy as np

def train_model(data_path, model_output_path):
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully from: {data_path}")
        print(f"Original DataFrame Info:\n{df.info()}")
        print(f"\nFirst 5 rows of the DataFrame:\n{df.head()}")
        print(f"\nValue counts for 'label' column (before conversion):\n{df['label'].value_counts()}")

        label_mapping = {1: 0, 0: 1} # 1 (original benign) -> 0 (new benign), 0 (original phishing) -> 1 (new phishing)
        df['label_numeric'] = df['label'].map(label_mapping).fillna(-1).astype(int)

        print(f"\nValue counts for 'label_numeric' column (after mapping):\n{df['label_numeric'].value_counts()}")

        for col in ['URLLength', 'DomainLength', 'TLDLength', 'NoOfSubDomain', 'NoOfObfuscatedChar',
                    'NoOfLettersInURL', 'NoOfDegitsInURL', 'NoOfEqualsInURL', 'NoOfQMarkInURL',
                    'NoOfAmpersandInURL', 'NoOfOtherSpecialSpecialCharsInURL', 'LineOfCode',
                    'LargestLineLength', 'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef',
                    'NoOfEmptyRef', 'NoOfExternalRef', 'NoOfURLRedirect', 'NoOfSelfRedirect',
                    'NoOfiFrame', 'NoOfPopup', 'HasHiddenFields']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('int32', errors='ignore') 
        
        for col in ['IsDomainIP', 'HasObfuscation', 'IsHTTPS', 'HasTitle', 'HasFavicon',
                    'Robots', 'IsResponsive', 'HasDescription', 'HasExternalFormSubmit',
                    'HasSocialNet', 'HasSubmitButton', 'HasPasswordField', 'Bank', 'Pay',
                    'Crypto', 'HasCopyrightInfo']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('int8', errors='ignore')

        for col in ['CharContinuationRate', 'ObfuscationRatio', 'LetterRatioInURL',
                    'DegitRatioInURL', 'SpacialCharRatioInURL', 'TLDLegitimateProb',
                    'URLCharProb', 'DomainTitleMatchScore', 'URLTitleMatchScore']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32', errors='ignore')

        print("\nDataFrame Info after Type Optimization:\n")
        print(df.info())

        columns_to_drop_for_X = ['URL', 'label', 'Title', 'Domain', 'label_numeric']
        X = df.drop(columns=[col for col in columns_to_drop_for_X if col in df.columns], errors='ignore')
        y = df['label_numeric']

        categorical_features = ['TLD']
        numerical_features = X.select_dtypes(include=np.number).columns.tolist() 
        
        if 'TLD' in numerical_features:
            numerical_features.remove('TLD')
        if 'TLD' not in X.columns and 'TLD' in categorical_features:
            categorical_features.remove('TLD')
        if 'TLD' not in X.columns:
            print("Error: 'TLD' column not found in features after dropping. Please check data_process.py output.")
            return

        numerical_transformer = Pipeline(steps=[('fillna', SimpleImputer(strategy='mean'))])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)])

        model = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print(f"\nShape of y_train: {y_train.shape}, Value counts in y_train:\n{pd.Series(y_train).value_counts()}")
        print(f"\nShape of y_test: {y_test.shape}, Value counts in y_test:\n{pd.Series(y_test).value_counts()}")

        print("\nTraining the Random Forest model with TLD encoding and refined features...")
        model.fit(X_train, y_train)
        print("Model training complete.")

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)

        print("\nModel Evaluation on Test Set:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(confusion)

        # --- Display Feature Importances ---
        print("\n--- Feature Importances ---")
        try:
            numerical_features_out = model.named_steps['preprocessor'].named_transformers_['num'].get_feature_names_out(numerical_features)
            categorical_features_out = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
            
            final_feature_names = list(numerical_features_out) + list(categorical_features_out)
            
            importances = model.named_steps['classifier'].feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': final_feature_names, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            print(feature_importance_df.head(20).to_string())
        except Exception as e:
            print(f"Could not display feature importances: {e}")
            import traceback
            traceback.print_exc()

        joblib.dump(model, model_output_path)
        print(f"\nTrained model with TLD encoding and refined features saved to: {model_output_path}")

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    data_file = 'phiusil_full_features.csv' 
    model_file = 'phishing_detection_model.joblib' # Changed to main model name
    train_model(data_file, model_file)