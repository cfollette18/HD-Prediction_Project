import os

# List of scripts to run in order
scripts = [
    "1_fetch_dataset.py",
    "2_preprocess_data.py",
    "3_split_data.py",
    "4_train_model.py",
    "5_evaluate_model.py",
    "6_interpret_model.py"
]

# Run each script
for script in scripts:
    print(f"\nRunning {script}...")
    print("=" * 50)
    os.system(f"python {script}")
    print(f"Finished running {script}.")
    print("=" * 50 + "\n")

print("All scripts executed successfully!")