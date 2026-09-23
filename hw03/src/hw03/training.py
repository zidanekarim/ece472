import jax.numpy as jnp
import numpy as np
import structlog
from flax import nnx
from tqdm import trange
import optax
from .config import TrainingSettings
from .data import Data
from .model import NNXLinearModel

log = structlog.get_logger()


@nnx.jit
def train_step(
    model, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray
):
    """Performs a single training step."""

    def loss_fn(model):
        logits = model(x)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # In-place update of model parameters, same as second assignment
    return loss


def train(
    model: NNXLinearModel,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> None:
    log.info("Starting training", **settings.model_dump())
    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(np_rng, settings.batch_size)
        x, y = jnp.asarray(x_np), jnp.asarray(y_np)

        loss = train_step(model, optimizer, x, y)
        bar.set_description(f"Loss @ {i} => {loss:.6f}")
        bar.refresh()
    log.info("Training finished")


"""Evaluation logic to get test accuracy"""
@nnx.jit
def eval_step(
    model: nnx.Module, x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    logits = model(x)
    predicted_classes = jnp.argmax(logits, axis=-1)
    correct_count = jnp.sum(predicted_classes == y)
    return correct_count, x.shape[0]


def evaluate(
    model: nnx.Module, test_dataset, batch_size: int = 256
) -> float:
    total_correct = 0
    total_samples = 0

    num_samples = len(test_dataset)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch = test_dataset[start_idx:end_idx]

        x = jnp.asarray(batch["image"])
        y = jnp.asarray(batch["label"])

        correct, count = eval_step(model, x, y)
        total_correct += int(correct)
        total_samples += int(count)

    accuracy = total_correct / total_samples
    return accuracy