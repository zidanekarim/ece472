import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import LinearModel, NNXLinearModel, Classifier

# from .plotting import compare_linear_models, plot_fit
from .training import train, evaluate

def optax_learning_rate_schedule( 
    total_steps: int,  # decay steps + warmup_steps = total_steps
    base_learning_rate: float,
    warmup_percentage: float = 0.10, # allocating 10% of steps to warmup
) -> optax.Schedule:
    warmup_steps = int(total_steps * warmup_percentage)
    decay_steps = total_steps - warmup_steps
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_learning_rate,
        warmup_steps=warmup_steps, # length of linear warmup (https://optax.readthedocs.io/en/latest/api/generated/optax.schedules.warmup_cosine_decay_schedule.html)
        decay_steps=decay_steps, 
        end_value=1e-5,
    )


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

    # log.debug("Data generating model", model=data_generating_model)

    data = Data(batch_size=settings.training.batch_size)
    data_no_augmentation = Data(batch_size=settings.training.batch_size, augmentation=False)

    model = Classifier(
        input_channels=3,  # 1 MNIST, 3 CIFAR for the increased input info
        layer_channels=[32, 64, 128],
        strides=[1, 2, 2],  # originally [1,1], which produced 91.2%
        num_classes=10,  # 10 classes cifar-10
        num_groups=8, # small common factor of the layer channels
        rngs=nnx.Rngs(params=model_key),
    )

    model2 = Classifier(
            input_channels=3,  # 1 MNIST, 3 CIFAR for the increased input info
            layer_channels=[32, 64, 128],
            strides=[1, 2, 2],  # originally [1,1], which produced 91.2%
            num_classes=10,  # 10 classes cifar-10
            num_groups=8, # small common factor of the layer channels
            residual=False,
            rngs=nnx.Rngs(params=model_key),
        )

    learning_rate = optax_learning_rate_schedule(total_steps=settings.training.num_iters, base_learning_rate=settings.training.learning_rate)

    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate=learning_rate, weight_decay=0.0001), wrt=nnx.Param # I switched to adamw here to boost by a couple percentage points
    )

    optimizer2 = nnx.Optimizer(
        model, optax.adamw(learning_rate=settings.training.learning_rate, weight_decay=0.0001), wrt=nnx.Param # I switched to adamw here to boost by a couple percentage points
    )

    history = train(model, optimizer, data, settings.training, np_rng)

    # log.debug("Trained model", model=model.model)

    # compare_linear_models(data.model, model.model)

    # if settings.data.num_features == 1:
    #     plot_fit(model, data, settings.plotting)
    # else:
    #     log.info("Skipping plotting for multi-feature models.")
    test_accuracy = evaluate(model, data.test_images, data.test_labels, batch_size=250)
    print(f"\nTest Accuracy: {test_accuracy * 100}%\n")
        