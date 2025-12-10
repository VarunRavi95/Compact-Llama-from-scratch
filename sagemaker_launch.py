import sagemaker
from sagemaker.pytorch import PyTorch
import os

# Configuration
# Replace with your actual role ARN if different
ROLE = "arn:aws:iam::359077243653:role/service-role/AmazonSageMaker-ExecutionRole-20251115T201648"
INSTANCE_TYPE = "ml.c5.2xlarge" # Consider "ml.g4dn.xlarge" for GPU training

# Hyperparameters matching train_fineweb.py arguments
hyperparams = {
    "subset": "sample-10BT",
    "val-split": 0.01,
    "batch-size": 4,
    "block-size": 128,
    "epochs": 1,
    "learning-rate": 6e-4,
    "device": "cpu", # Change to 'cuda' if using a GPU instance
    "max-train-samples": 1000, # Limit samples for quick testing
    "output-dir": "/opt/ml/model", # Standard SageMaker model output directory
    "num-workers": 4,
}

def launch_training():
    print(f"Launching SageMaker training job with role: {ROLE}")
    print(f"Instance type: {INSTANCE_TYPE}")
    
    estimator = PyTorch(
        entry_point="train_fineweb.py",
        source_dir=".",  # Uploads the current directory including config.py, model.py, etc.
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version="2.2",
        py_version="py310",
        hyperparameters=hyperparams,
        environment={
            "WANDB_DISABLED": "1", # Disable W&B unless you have it configured
        },
    )

    # Start the training job
    estimator.fit()

if __name__ == "__main__":
    launch_training()
