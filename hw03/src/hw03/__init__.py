import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import LinearModel, NNXLinearModel, Classifer
from .plotting import compare_linear_models, plot_fit
from .training import train


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.key(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.asarray(jax.random.key_data(data_key)))


    log.debug("Data generating model", model=data_generating_model)

    data = Data(
        batch_size=settings.training.batch_size
    )

    model = Classifer(
        input_channels=1, # mnist specs
        layer_channels=[32, 64], 
        kernel_sizes=[(3, 3), (3, 3)], 
        strides=[1, 1],
        num_classes=10, # digits 0-9
        rngs=np_rng
    )
    log.debug("Initial model", model=model.model)

    optimizer = nnx.Optimizer(
        model, optax.adamw(settings.training.learning_rate), wrt=nnx.Param
    )

    train(model, optimizer, data, settings.training, np_rng)

    log.debug("Trained model", model=model.model)

    # compare_linear_models(data.model, model.model)

    # if settings.data.num_features == 1:
    #     plot_fit(model, data, settings.plotting)
    # else:
    #     log.info("Skipping plotting for multi-feature models.")
