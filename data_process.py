import pandas as pd

def create_full_features_dataset(input_csv_path, output_csv_path):
    """
    Loads the PhiUSIIL dataset, selects the specified full set of features
    (excluding URLSimilarityIndex and other identified low-value/redundant features),
    and saves them to a new CSV file.

    Args:
        input_csv_path (str): The path to the original PhiUSIIL_Phishing_URL_Dataset.csv.
        output_csv_path (str): The path to save the new CSV file with full features.
    """
    # Finalized list of features to KEEP after correlation analysis and review
    columns_to_keep = [
        'URL', # Kept for reference, will be dropped in model_train.py
        'URLLength',
        'Domain', # Kept for reference, will be dropped in model_train.py
        'DomainLength',
        'TLD', # Will be one-hot encoded in model_train.py
        'CharContinuationRate',
        'SpacialCharRatioInURL',
        'DegitRatioInURL', # This was negative correlation with label
        'IsHTTPS',
        'LineOfCode',
        'HasTitle', # Will be dropped in model_train.py (as its text content)
        'DomainTitleMatchScore', 
        'HasFavicon',
        'Robots',
        'IsResponsive',
        'HasDescription',
        'NoOfiFrame',
        'HasExternalFormSubmit',
        'HasSocialNet',
        'HasSubmitButton',
        'HasHiddenFields',
        'HasPasswordField',
        'Bank',
        'Pay',
        'HasCopyrightInfo',
        'NoOfImage',
        'NoOfCSS', # Low correlation, but often kept as a basic content feature
        'NoOfJS',
        'NoOfSelfRef',
        'NoOfExternalRef',
        'label' # Our target variable
    ]
    
    try:
        df_original = pd.read_csv(input_csv_path)
        print(f"Original data loaded successfully from: {input_csv_path}. Shape: {df_original.shape}")

        # Drop URLSimilarityIndex and any other columns that are in the original but not in columns_to_keep
        # This handles cases where original CSV might have more columns than we want
        # Or specifically remove URLSimilarityIndex if it exists
        if 'URLSimilarityIndex' in df_original.columns:
            df_original = df_original.drop(columns=['URLSimilarityIndex'])
            print(f"Removed 'URLSimilarityIndex' from the original DataFrame.")
        
        # Ensure that all columns_to_keep are actually present in the DataFrame before selecting
        missing_columns = [col for col in columns_to_keep if col not in df_original.columns]
        if missing_columns:
            raise KeyError(f"The following required columns are missing from the input CSV: {missing_columns}")

        # Select only the desired columns for the full features
        df_full = df_original.loc[:, columns_to_keep].copy() 

        df_full.to_csv(output_csv_path, index=False)
        print(f"Successfully created '{output_csv_path}' with the specified full features. Shape: {df_full.shape}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv_path}' not found.")
    except KeyError as e:
        print(f"Error: A column issue occurred: {e}. Please check column names in the original CSV or your 'columns_to_keep' list.")
    except Exception as e:
        print(f"An unexpected error occurred during data processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    input_file_path = 'PhiUSIIL_Phishing_URL_Dataset.csv'
    output_full_features_path = 'phiusil_full_features.csv'
    create_full_features_dataset(input_file_path, output_full_features_path)