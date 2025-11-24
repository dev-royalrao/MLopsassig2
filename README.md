# ML Ops - Olivetti Faces (DecisionTree) — End-to-End

This repo contains:
- `src/train.py` — trains DecisionTreeClassifier and saves model
- `src/test.py` — loads model and test data and prints accuracy
- `app/` — Flask app to accept image upload and predict class
- `Dockerfile` — builds container for Flask app
- `.github/workflows/ci.yml` — GitHub Actions workflow to run train & test
- `k8s/` — Kubernetes manifests for deployment

## Branching strategy (assignment required)
- `main` — initial README and .gitignore
- `dev` — model development (train/test)
- `docker_cicd` — docker + deployment related work (Flask, Dockerfile, k8s)

## Local dev (Ubuntu / WSL / Mac)
```bash
# create virtualenv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# train
python src/train.py

# run tests
python src/test.py

# run flask app locally
cd app
python app.py
# or: gunicorn --bind 0.0.0.0:5000 app:app --workers 2
