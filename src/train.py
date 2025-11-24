# src/train.py
"""
Train a DecisionTreeClassifier on Olivetti faces and save:
 - saved/savedmodel.pth  (joblib)
 - saved/test_data.npz   (X_test, y_test)
"""
import os
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    data = fetch_olivetti_faces()
    X, y = data.data, data.target  # X shape = (400, 4096)
    # 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    model_path = os.path.join(OUTPUT_DIR, "savedmodel.pth")
    joblib.dump(clf, model_path)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "test_data.npz"),
                        X_test=X_test, y_test=y_test)
    print(f"Model saved to: {model_path}")
    print(f"Test data saved to: {os.path.join(OUTPUT_DIR, 'test_data.npz')}")

if __name__ == "__main__":
    main()
