import os
import sys
import subprocess
import logging

# NIDS ML pipeline (beginner notes):
# - preprocess.py loads the CSV, cleans it, then keeps a stratified 200k-row working set
#   (multiclass Attack Type proportions preserved). It splits train/test; test is for metrics only.
# - Hyperparameter tuning uses 30% of the training data for speed; train.py retrains on the
#   full training portion with the best saved parameters.
# - evaluate.py reports metrics on the held-out test set only.

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Required folders
folders = [
    "models",
    "reports",
    "data/processed"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Pipeline files
pipeline = [
    "scripts/preprocess.py",
    "scripts/feature_selection.py",
    "scripts/hyperparameter_tuning.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/predict.py"
]

# Running scripts
def run_script(script):

    print(f"\n{'=' * 60}")
    print(f"Running: {script}")
    print(f"{'=' * 60}\n")

    try:

        subprocess.run(
            [sys.executable, script],
            check=True
        )

        print(f"\nCompleted: {script}\n")

    except subprocess.CalledProcessError as e:

        print(f"\nError while running {script}")
        print(e)

        sys.exit(1)

# Main pipeline
if __name__ == "__main__":

    print("\nStarting NIDS ML Pipeline...\n")

    for script in pipeline:
        run_script(script)

    print("\nPipeline Completed Successfully")