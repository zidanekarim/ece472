from dataclasses import InitVar, dataclass, field
from datasets import load_dataset
import numpy as np


def preprocess(batch):
    images = [np.array(img, dtype=np.float32) for img in batch["image"]]
    
    images = np.stack(images, axis=0)
    images = np.expand_dims(images, axis=-1)  # add channel dimension for grayscale images, since our flax conv wants format: batch, hieght, width, channels
    images = images / 255.0  # we normalize bc the pixel values are between 0 and 255, and we want them to be between 0 and 1. this is usual for image data when taking from huggingface or similar  

    labels = np.array(batch["label"], dtype=np.int32)
    return {"image": images, "label": labels}


def load_mnist_dataset():
    dataset = load_dataset("ylecun/mnist")
    train_data = dataset["train"].map(preprocess, batched=True)
    validation_data = dataset["test"].map(preprocess, batched=True)

    return train_data, validation_data

@dataclass
class Data:
    """Handles generation of synthetic data for linear regression."""

    batch_size: int

    def __post_init__(self):
        self.train_data, self.validation_data = load_mnist_dataset()
        self.train_data.set_format(type="numpy", columns=["image", "label"]) # numpy technique to speed computation, nothing crazy
        self.validation_data.set_format(type="numpy", columns=["image", "label"]) # numpy technique to speed computation, nothing crazy

        #self.train_images = self.train_data["image"]
        #self.train_labels = self.train_data["label"]
        batch = self.train_data[:]
        self.train_images = np.asarray(batch["image"], dtype=np.float32)
        self.train_labels = np.asarray(batch["label"], dtype=np.int32)

        self.num_train_samples = len(self.train_data)
        self.index = np.arange(self.num_train_samples)

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size, replace=False)
        batch = self.train_data[choices]
        return batch["image"], batch["label"]