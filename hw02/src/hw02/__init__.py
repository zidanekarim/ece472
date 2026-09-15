import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import LinearModel, NNXLinearModel, SwiGLUMLP, MLP
from .plotting import plot_training_samples
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

    # data_generating_model = LinearModel(
    #     weights=np_rng.integers(low=0, high=5, size=(settings.data.num_features)),
    #     bias=2,
    # )
    #log.debug("Data generating model", model=data_generating_model)

    data = Data(
        noise=settings.data.sigma_noise,
    )
    data.two_spiral_generator(rng=np_rng)

    num_hidden_layers = 3
    hidden_layer_width = 64
    data_x, data_y = data.X, data.Y
    model1 = MLP(
        num_inputs=2,             # x1, x2
        num_outputs=2,            #
        num_hidden_layers=num_hidden_layers,
        hidden_layer_width=hidden_layer_width, 
        hidden_activation=nnx.gelu,
        output_activation=nnx.identity,
        rngs=nnx.Rngs(params=model_key),
    )



    #log.debug("Initial model", model=model.model)

    optimizer1 = nnx.Optimizer(
        model1, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    train(model1, optimizer1, data, settings.training, np_rng)

    #log.debug("Trained model", model=model.model)
    plot_training_samples(model1, data, settings.plotting, "MLP")





    model2 = SwiGLUMLP(
        num_inputs=2,             # x1, x2
        num_outputs=2,            #
        num_hidden_layers=num_hidden_layers,
        hidden_layer_width=hidden_layer_width, 
        output_activation=nnx.identity,
        rngs=nnx.Rngs(params=model_key),
    )

    optimizer2 = nnx.Optimizer(
        model2, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    train(model2, optimizer2, data, settings.training, np_rng)

    plot_training_samples(model2, data, settings.plotting, "SwiGLU-MLP")

    # if settings.data.num_features == 1:
    #     plot_fit(model, data, settings.plotting)
    # else:
    #     log.info("Skipping plotting for multi-feature models.")
