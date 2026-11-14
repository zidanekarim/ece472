from dataclasses import InitVar, dataclass, field
from datasets import load_dataset
import numpy as np
import jax.numpy as jnp
import jax
from flax import nnx
from dataclasses import InitVar, dataclass, field
from datasets import load_dataset
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def _load_from_huggingface():
    dataset = load_dataset("mteb/banking77")
    train_texts = dataset["train"]["text"]
    train_labels = jnp.array(dataset["train"]["label"])

    test_texts = dataset["test"]["text"]
    test_labels = jnp.array(dataset["test"]["label"])

    return train_texts, train_labels, test_texts, test_labels


@dataclass
class Data:
    model: SentenceTransformer
    sigma: float
    rng: InitVar[np.random.Generator]
    num_classes: int = 77
    x: jax.Array = field(init=False)
    y: jax.Array = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):

        self.train_texts, self.train_labels, self.test_texts, self.test_labels = (
            _load_from_huggingface()
        )
        self.train_emb_np = self.model.encode(
            self.train_texts,
            batch_size=64,  # 64 for SentenceTransformers
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self.test_emb_np = self.model.encode(
            self.test_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        self.train_embeddings, self.test_embeddings = (
            jnp.array(self.train_emb_np),
            jnp.array(self.test_emb_np),
        )
        self.x = self.train_embeddings
        self.y = self.train_labels
        self.index = np.arange(len(self.x))

        self.dimension = self.x.shape[1]  # 384

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[jax.Array, jax.Array]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()
