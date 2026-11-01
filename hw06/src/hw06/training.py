import jax.numpy as jnp
import numpy as np
import structlog
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Data
from .model import TransformerBlock
import optax
log = structlog.get_logger()


@nnx.jit
def train_step(
    model: TransformerBlock, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray, causal: bool = False
):
    """Performs a single training step."""

    def loss_fn(model: TransformerBlock):
        logits = model(x, causal=causal)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # In-place parameter update
    return loss


def train(
    model: TransformerBlock,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
    causal: bool = False
) -> None:
    """Train the model using SGD."""
    log.info("Starting training", **settings.model_dump())
    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(np_rng, settings.batch_size)
        x, y = jnp.asarray(x_np), jnp.asarray(y_np)

        loss = train_step(model, optimizer, x, y, causal=causal)

        bar.set_description(f"Loss @ {i} => {loss:.6f}")
        bar.refresh()
    log.info("Training finished")
