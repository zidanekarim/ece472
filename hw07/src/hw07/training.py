from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import structlog
from flax import nnx

from .config import TrainingSettings
from .data import Data
from .model import SparseAutoencoder

log = structlog.get_logger()


def save_checkpoint(model: nnx.Module, checkpoint_path: Path) -> None:
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(checkpoint_path, state)
    checkpointer.wait_until_finished()
    checkpointer.close()


def normalize_decoder_(sae: SparseAutoencoder) -> None:
    """Enforce unit L2 norm on every dictionary feature vector in W_dec."""
    norms = jnp.linalg.norm(sae.W_dec.value, axis=-1, keepdims=True)
    sae.W_dec.value = sae.W_dec.value / jnp.maximum(norms, 1e-8)


@nnx.jit
def sae_train_step(
    sae: SparseAutoencoder,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    aux_alpha: float = 1.0 / 32.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single Top-k SAE step with unit-norm decoder projection and AuxK dead-latent revival."""

    def loss_fn(model: SparseAutoencoder):
        x_cent = x - model.b_dec.value
        pre_acts = x_cent @ model.W_enc.value + model.b_enc.value

        # Primary Top-k forward pass
        top_vals, top_idx = jax.lax.top_k(pre_acts, model.k)
        top_vals = jax.nn.relu(top_vals)
        z = jnp.zeros_like(pre_acts)
        z = z.at[jnp.arange(pre_acts.shape[0])[:, None], top_idx].set(top_vals)

        x_hat = z @ model.W_dec.value + model.b_dec.value
        residual = x - x_hat
        mse_loss = jnp.mean(jnp.sum(residual**2, axis=-1))

        # Auxiliary loss on inactive latents to prevent dead features at m=1536, 3072
        inactive_mask = z == 0.0
        masked_pre = jnp.where(inactive_mask, pre_acts, -1e9)
        k_aux = min(model.k * 2, model.m_latents - model.k)
        aux_vals, aux_idx = jax.lax.top_k(masked_pre, k_aux)
        aux_vals = jax.nn.relu(aux_vals)
        z_aux = jnp.zeros_like(pre_acts)
        z_aux = z_aux.at[jnp.arange(pre_acts.shape[0])[:, None], aux_idx].set(aux_vals)

        residual_hat = z_aux @ model.W_dec.value
        aux_loss = jnp.mean(jnp.sum((residual - residual_hat) ** 2, axis=-1))

        total_loss = mse_loss + aux_alpha * aux_loss
        return total_loss, (mse_loss, z)

    (_, (mse_loss, z)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(sae)
    optimizer.update(sae, grads)
    normalize_decoder_(sae)

    active_mask = jnp.any(z > 0, axis=0)  # Shape: (m,)
    return mse_loss, z, active_mask




def train_sae_models(
    data: Data,
    settings_training: TrainingSettings,
    latent_dims: list[int] = [768, 1536, 3072],
    k: int = 20,
    epochs: int = 80,
) -> tuple[dict[int, SparseAutoencoder], dict[int, dict[str, float]]]:
    """Train, evaluate, and checkpoint SAEs across m in {768, 1536, 3072}."""
    x_train = data.train_embeddings
    x_test = data.test_embeddings
    n_samples, d_in = x_train.shape
    batch_size = settings_training.batch_size
    n_batches = int(np.ceil(n_samples / batch_size))
    total_steps = epochs * n_batches

    models: dict[int, SparseAutoencoder] = {}
    results: dict[int, dict[str, float]] = {}

    for m in latent_dims:
        log.info(f"Training Top-{k} SAE", m_latents=m, expansion=f"{m // d_in}x")
        sae = SparseAutoencoder(d_in=d_in, m_latents=m, k=k, rngs=nnx.Rngs(67 + m))

        # Center decoder bias to dataset mean
        sae.b_dec.value = jnp.mean(x_train, axis=0)
        normalize_decoder_(sae)

        schedule = optax.cosine_decay_schedule(
            init_value=settings_training.learning_rate, decay_steps=total_steps
        )
        optimizer = nnx.Optimizer(sae, optax.adam(schedule), wrt=nnx.Param)
        rng = np.random.default_rng(67 + m)

        # Track dead latents over the last 10 epochs of training
        recent_active = np.zeros(m, dtype=bool)

        for epoch in range(1, epochs + 1):
            if epoch == epochs - 9:
                recent_active[:] = False

            perm = rng.permutation(n_samples)
            x_shuf = x_train[perm]

            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_samples)
                _, _, batch_active = sae_train_step(sae, optimizer, x_shuf[start:end])
                recent_active |= np.asarray(batch_active)

            if epoch % 20 == 0 or epoch == epochs:
                val_metrics = evaluate_sae(
                    sae, x_test, train_active_tracker=recent_active
                )
                log.info(
                    f"m={m} | Epoch {epoch:02d}",
                    test_r2=f"{val_metrics['r2']:.4f}",
                    test_mse=f"{val_metrics['mse']:.5f}",
                    l0=f"{val_metrics['l0']:.1f}",
                    dead_pct=f"{val_metrics['dead_fraction'] * 100:.2f}%",
                )

        ckpt_dir = Path("./checkpoints/") / f"sae_{m}"
        save_checkpoint(sae, ckpt_dir)
        log.info(f"Checkpointed SAE weights for m={m}", path=str(ckpt_dir))

        models[m] = sae
        results[m] = evaluate_sae(sae, x_test, train_active_tracker=recent_active)

    return models, results