import os
import sys
import subprocess
import logging


# Beginner Notes

# This file runs the complete NIDS ML pipeline step by step.

# Step 1: preprocess.py
# Loads CSV data, cleans it, and prepares train/test data.

# Step 2: feature_selection.py
# Finds important features using Random Forest.

# Step 3: hyperparameter_tuning.py
# Finds the best settings for ML models.

# Step 4: train.py
# Trains the final model using best settings.

# Step 5: evaluate.py
# Tests the model and creates reports.

# Step 6: predict.py
# Runs prediction using the trained model.


# Logging setup

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


# Create folders one by one
for folder in folders:

    os.makedirs(
        folder,
        exist_ok=True
    )


# Pipeline script files

pipeline = [
    "scripts/preprocess.py",
    "scripts/feature_selection.py",
    "scripts/hyperparameter_tuning.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/predict.py"
]


# Function to run one script

def run_script(script):

    print(f"\n{'=' * 60}")

    print(f"Running: {script}")

    print(f"{'=' * 60}\n")

    try:

        # Run the current Python script
        subprocess.run(
            [sys.executable, script],
            check=True
        )

        print(f"\nCompleted: {script}\n")

    except subprocess.CalledProcessError as e:

        print(f"\nError while running {script}")

        print(e)

        sys.exit(1)


# Main pipeline starts here

if __name__ == "__main__":

    print("\nStarting NIDS ML Pipeline...\n")

    # Run each script one by one
    for script in pipeline:

        run_script(script)

    print("\nPipeline Completed Successfully")