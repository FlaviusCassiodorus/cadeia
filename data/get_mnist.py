# You need to install
# pip install numpy tensorflow tensorflow-datasets
# Possibly importlib_resources as well

import numpy as np
import tensorflow_datasets as tfds
from pathlib import Path

def download_mnist(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download MNIST
    train_ds, test_ds = tfds.load(
        "mnist",
        split=["train", "test"],
        as_supervised=True,
        batch_size = -1
    )

    # Convert TensorFlow tensors to NumPy arrays
    X_train, y_train = tfds.as_numpy(train_ds)
    X_test, y_test = tfds.as_numpy(test_ds)

    # Remove the final channel dimension: (N, 28, 28, 1) -> (N, 28, 28)
    X_train = X_train.squeeze(axis=-1).astype(np.float32) / 255.0
    X_test = X_test.squeeze(axis=-1).astype(np.float32) / 255.0

    # Encode labels once so each row matches the classifier's 10 logits.
    y_train = np.eye(10, dtype=np.float32)[np.asarray(y_train)]
    y_test = np.eye(10, dtype=np.float32)[np.asarray(y_test)]

    # Save to .npy files
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)

    print(f"Saved to: {output_dir.resolve()}")
    print(f"X_train: {X_train.shape}, range [{X_train.min()}, {X_train.max()}]")
    print(f"X_test:  {X_test.shape}, range [{X_test.min()}, {X_test.max()}]")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / Path("data/mnist")
    X_train, y_train, X_test, y_test = download_mnist(output_dir)

