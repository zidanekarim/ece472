from datasets import load_dataset
import jax
import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .plotting import plot_sae_results
from .training import train_sae_models


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.key(settings.random_seed)
    data_key, _ = jax.random.split(key)
    np_rng = np.random.default_rng(np.asarray(jax.random.key_data(data_key)))

    embeddings_model = SentenceTransformer(
        "hotchpotch/bekko-embedding-v1-a8m",
        backend="openvino",
        model_kwargs={
            "file_name": "openvino/openvino_model.xml"
        },  # loads the official OpenVINO artifact
    )

    data = Data(model=embeddings_model, rng=np_rng, sigma=settings.data.sigma_noise)

    latent_dims = [768, 1536, 3072]
    k = 20

    # Train and checkpoint SAEs across 2x, 4x, and 8x overcomplete dictionaries
    models, results = train_sae_models(
        data=data,
        settings_training=settings.training,
        latent_dims=latent_dims,
        k=k,
        epochs=80,
    )



