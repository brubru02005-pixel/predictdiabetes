import pandas as pd
from data_preprocessing import DiabetesDataPreprocessor
from feature_selection import select_features_for_dataset
import sys

def main():
    # Load and preprocess combined dataset as in app.py
    preprocessor = DiabetesDataPreprocessor()
    result = preprocessor.preprocess_dataset("COMBINED", apply_smote_flag=True)
    if result is None:
        print("Failed to preprocess dataset.")
        return
    X_train, X_test, y_train, y_test, info = result

    # Use mutual_info feature selection (top 6 features)
    X_train_selected, X_test_selected, selector, summary = select_features_for_dataset(
        X_train, X_test, y_train, 'COMBINED', feature_names=info['features'], method='mutual_info', k=6
    )

    X_test_path = sys.argv[1] if len(sys.argv) > 1 else 'X_test.csv'
    y_test_path = sys.argv[2] if len(sys.argv) > 2 else 'y_test.csv'

    pd.DataFrame(X_test_selected).to_csv(X_test_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False)
    print(f"Exported X_test to {X_test_path}")
    print(f"Exported y_test to {y_test_path}")

if __name__ == "__main__":
    main()
