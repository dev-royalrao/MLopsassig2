# src/test.py
"""
Load saved model and test set, compute test accuracy and print result.
"""
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved')
MODEL_PATH = os.path.join(BASE_DIR, "savedmodel.pth")
TEST_PATH = os.path.join(BASE_DIR, "test_data.npz")

def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError("Run src/train.py first to create model and test data.")

    clf = joblib.load(MODEL_PATH)
    data = np.load(TEST_PATH)
    X_test = data['X_test']
    y_test = data['y_test']
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
