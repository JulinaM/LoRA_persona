import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# --- Configuration ---
# List of all personality traits to be processed
ALL_TARGET_COLUMNS = ['cOPN']
TEXT_COLUMN = "STATUS"
LIWC_DATA_FILE = "/users/PGS0218/julina/projects/LoRA_persona/data/LIWC_mypersonality_v2.csv"


def load_and_prepare_data(file_path, text_col, target_col):
    """
    Loads data from a CSV and prepares it for a specific target personality trait.
    """
    print(f"Loading and preparing data for target: {target_col}")
    df = pd.read_csv(file_path)

    # REFINEMENT: Clean up column dropping for better readability
    # 1. Drop any 'Unnamed' columns that might exist from CSV saving/loading
    df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True, errors='ignore')

    # 2. Rename the current target column to 'label'
    df.rename(columns={target_col: 'label'}, inplace=True)

    # 3. Identify and drop the other non-target traits and the text column
    all_traits = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
    traits_to_drop = [trait for trait in all_traits if trait != target_col]
    df_processed = df.drop(columns=[text_col] + traits_to_drop, errors='ignore')

    # Pre-split check to prevent errors if a label has only one class
    if df_processed['label'].nunique() < 2:
        print(f"Warning: Not enough class diversity for {target_col}. Skipping this trait.")
        return None, None

    train_df, test_df = train_test_split(df_processed, test_size=0.2, random_state=42, stratify=df_processed['label'])
    print("Data preparation complete.")
    return train_df, test_df


# --- Main Execution Loop ---
# Dictionary to store the accuracy for each trait
all_results = {}

for target_trait in ALL_TARGET_COLUMNS:
    print("\n" + "="*60)
    print(f" PROCESSING TRAIT: {target_trait} ".center(60, "="))
    print("="*60)

    # Call the modified function for the current trait in the loop
    train_df, test_df = load_and_prepare_data(LIWC_DATA_FILE, TEXT_COLUMN, target_trait)

    # If data preparation failed (e.g., not enough classes), skip to the next trait
    if train_df is None:
        all_results[target_trait] = "Skipped"
        continue

    # Separate features (X) and labels (y)
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']

    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    # DEBUG: Check label distribution AFTER splitting
    print("\nLabel distribution in y_train:")
    print(y_train.value_counts())

    # --- SVM Training ---
    print("\nTraining SVM model...")
    try:
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X_train, y_train)
        print("SVM model training complete.")
    except Exception as e:
        print(f"An error occurred during SVM training for {target_trait}: {e}")
        all_results[target_trait] = "Failed (SVM Training Error)"
        continue

    # --- Evaluation ---
    predictions_svm = svm_model.predict(X_test)
    accuracy_svm = accuracy_score(y_test, predictions_svm)

    print(f"\nRESULTS FOR {target_trait}:")
    print(f"SVM + LIWC Accuracy: {accuracy_svm:.4f}")

    # Store the result
    all_results[target_trait] = accuracy_svm

# --- Final Summary ---
print("\n" + "="*60)
print(" FINAL SUMMARY OF ACCURACIES ".center(60, "="))
print("="*60)
for trait, accuracy in all_results.items():
    if isinstance(accuracy, float):
        print(f"{trait:<10}: {accuracy:.4f}")
    else:
        print(f"{trait:<10}: {accuracy}")
print("="*60)