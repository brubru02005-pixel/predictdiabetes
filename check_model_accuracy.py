import sys
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Usage: py check_model_accuracy.py <model_path> <X_test_path> <y_test_path>
# Example: py check_model_accuracy.py models/Stack-RF-TUNED_COMBINED.joblib X_test.csv y_test.csv

def main():
    if len(sys.argv) != 4:
        print("Usage: py check_model_accuracy.py <model_path> <X_test_path> <y_test_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_test_path = sys.argv[3]

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"Loading X_test from: {X_test_path}")
    X_test = pd.read_csv(X_test_path)
    print(f"Loading y_test from: {y_test_path}")
    y_test = pd.read_csv(y_test_path).squeeze()

    print("Predicting...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
