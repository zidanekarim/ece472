from dataclasses import InitVar, dataclass, field
from datasets import load_dataset
import numpy as np
from pathlib import Path
import pickle

def _load_pickle_batch(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(file_path, "rb") as f:
        entry = pickle.load(f, encoding="bytes")
    # Shape: (10000, 3072) -> transpose to (10000, 32, 32, 3)
    raw_data = entry[b"data"]
    labels = np.array(entry[b"labels"], dtype=np.int32)
    images = raw_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def _load_from_local_batches(
    batches_dir: Path,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    train_x_list, train_y_list = [], []
    for i in range(1, 6):
        x, y = _load_pickle_batch(batches_dir / f"data_batch_{i}")
        train_x_list.append(x)
        train_y_list.append(y)

    train_images = (
        np.concatenate(train_x_list, axis=0).astype(np.float32) / 255.0
    )
    train_labels = np.concatenate(train_y_list, axis=0)

    test_images, test_labels = _load_pickle_batch(batches_dir / "test_batch")
    test_images = test_images.astype(np.float32) / 255.0

    return (train_images, train_labels), (test_images, test_labels)


def _load_from_huggingface() -> ( # this function is used if the dataset isnt found locally (since the dataset isnt pushed to git)
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
):
    print("Local CIFAR-10 directory not found. Loading from Hugging Face...")
    from datasets import load_dataset

    dataset = load_dataset("uoft-cs/cifar10")

    train_images = (
        np.stack(
            [np.array(img, dtype=np.float32) for img in dataset["train"]["img"]]
        )
        / 255.0
    )
    train_labels = np.array(dataset["train"]["label"], dtype=np.int32)

    test_images = (
        np.stack(
            [np.array(img, dtype=np.float32) for img in dataset["test"]["img"]]
        )
        / 255.0
    )
    test_labels = np.array(dataset["test"]["label"], dtype=np.int32)

    return (train_images, train_labels), (test_images, test_labels)


def find_cifar_dir() -> Path | None:
    """Finds the cifar-10 folder regardless of where python/just is executed."""
    current_file_dir = Path(__file__).resolve().parent  # .../src/hw04
    repo_root = current_file_dir.parents[
        1
    ]  # .../ (two levels up from src/hw04)

    search_paths = [
        current_file_dir / "data" / "cifar-10-batches-py",
        repo_root / "data" / "cifar-10-batches-py",
        Path("data/cifar-10-batches-py").resolve(),
        Path("../data/cifar-10-batches-py").resolve(),
    ]

    for p in search_paths:
        if p.is_dir() and (p / "data_batch_1").exists():
            return p
    return None


def load_cifar10():
    cifar_dir = find_cifar_dir()

    if cifar_dir is not None:
        print(f"Loading local CIFAR-10 from: {cifar_dir}")
        return _load_from_local_batches(cifar_dir)

    print(
        "Local CIFAR-10 directory not found anywhere. Loading from Hugging Face..."
    )
    return _load_from_huggingface()

def augment_batch(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random horizontal flip + 4-pixel reflect padding & random crop to 32x32."""
    B, H, W, C = images.shape

    # 1. Random horizontal flip (50% probability)
    flips = rng.integers(0, 2, size=B)
    for i in range(B):
        if flips[i]:
            images[i] = np.fliplr(images[i])

    # 2. 4-pixel pad & random crop back to 32x32[cite: 1]
    padded = np.pad(images, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    cropped = np.empty_like(images)
    for i in range(B):
        top = rng.integers(0, 9)
        left = rng.integers(0, 9)
        cropped[i] = padded[i, top : top + H, left : left + W, :]

    return cropped


@dataclass
class Data:
    batch_size: int
    data_dir: str = "data/cifar-10-batches-py"
    augment: bool = True  

    def __post_init__(self):
        (self.train_images, self.train_labels), (
            self.test_images,
            self.test_labels,
        ) = load_cifar10()

        self.num_train_samples = len(self.train_images)
        self.index = np.arange(self.num_train_samples)

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        choices = rng.choice(self.index, size=batch_size, replace=False)
        batch_x = self.train_images[choices].copy()
        batch_y = self.train_labels[choices]

        if self.augment:
            batch_x = augment_batch(batch_x, rng)

        return batch_x, batch_y