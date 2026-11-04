import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .logging import configure_logging



def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # # JAX PRNG
    # key = jax.random.key(settings.random_seed)
    # data_key, model_key = jax.random.split(key)
    # np_rng = np.random.default_rng(np.asarray(jax.random.key_data(data_key)))
    # data = Data(
    #     model=TransformerBlock(
    #         d_model=settings.model.d_model,
    #         d_ff=settings.model.d_ff,
    #         num_heads=settings.model.num_heads,
    #         use_bias=settings.model.use_bias,
    #         rngs=nnx.Rngs(model_key),
    #     ),
    #     rng=np_rng,
    #     num_features=settings.data.num_features,
    #     num_samples=settings.data.num_samples,
    #     sigma=settings.data.sigma_noise,
    # )

    # model = TransformerBlock(
    #     d_model=settings.model.d_model,
    #     d_ff=settings.model.d_ff,
    #     num_heads=settings.model.num_heads,
    #     use_bias=settings.model.use_bias,
    #     rngs=nnx.Rngs(model_key),
    # )
    # #log.debug("Initial model", model=model.model)

    # optimizer = nnx.Optimizer(
    #     model, optax.adamw(settings.training.learning_rate), wrt=nnx.Param
    # )

    # train(model, optimizer, data, settings.training, np_rng, causal=settings.model.causal)

    # log.debug("Trained model", model=model.model)
    
    """
    This file is essentially unused. All functionality comes from evaluate
    """