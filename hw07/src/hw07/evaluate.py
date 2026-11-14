"""Standalone evaluation entry point for hw07."""

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import structlog
from flax import nnx
from sentence_transformers import SentenceTransformer

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import SparseAutoencoder
from .plotting import plot_sae_results
from .training import evaluate_sae

log = structlog.get_logger()

def evaluate_sae(
    sae: SparseAutoencoder,
    x_eval: jax.Array,
    train_active_tracker: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute MSE, R^2 (variance explained), mean L0 sparsity, and dead latent fraction."""
    x_hat, z = sae(x_eval)

    # MSE & Fraction of Variance Explained (R^2)
    sq_err = jnp.sum((x_eval - x_hat) ** 2, axis=-1)
    mse = float(jnp.mean(sq_err))

    x_mean = jnp.mean(x_eval, axis=0, keepdims=True)
    total_var = jnp.sum((x_eval - x_mean) ** 2, axis=-1)
    r2 = float(1.0 - jnp.sum(sq_err) / jnp.sum(total_var))

    # L0 norm: average active latents per embedding
    l0 = float(jnp.mean(jnp.sum(z > 0, axis=-1)))

    # Dead latent fraction
    if train_active_tracker is not None:
        dead_frac = float(np.mean(~train_active_tracker))
    else:
        ever_active = jnp.any(z > 0, axis=0)
        dead_frac = float(1.0 - jnp.mean(ever_active))

    return {
        "mse": mse,
        "r2": r2,
        "l0": l0,
        "dead_fraction": dead_frac,
    }


def load_sae_from_checkpoint(
    d_in: int, m_latents: int, k: int, ckpt_dir: Path
) -> SparseAutoencoder:
    skeleton = SparseAutoencoder(
        d_in=d_in, m_latents=m_latents, k=k, rngs=nnx.Rngs(0)
    )
    graphdef, target_state = nnx.split(skeleton)

    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(Path(ckpt_dir).resolve(), target_state)
    checkpointer.close()

    return nnx.merge(graphdef, restored_state)


def inspect_top_features(
    sae: SparseAutoencoder,
    test_embeddings: jax.Array,
    test_texts: list[str],
    test_labels: jax.Array,
    num_features: int = 6,
    top_n_sentences: int = 5,
) -> None:
    _, Z = sae(test_embeddings)
    Z_np = np.asarray(Z)
    labels_np = np.asarray(test_labels)

    active_counts = np.sum(Z_np > 0, axis=0)
    alive_indices = np.where(active_counts > 0)[0]

    sorted_by_freq = alive_indices[np.argsort(active_counts[alive_indices])]
    quantiles = np.linspace(0.15, 0.90, num_features)
    chosen_features = [
        int(sorted_by_freq[int(q * (len(sorted_by_freq) - 1))]) for q in quantiles
    ]

    for feat_idx in chosen_features:
        acts = Z_np[:, feat_idx]
        top_doc_indices = np.argsort(acts)[::-1][:top_n_sentences]
        freq_pct = (active_counts[feat_idx] / len(test_texts)) * 100

        print(
            f"\n=== Latent Feature #{feat_idx} "
            f"(Active on {active_counts[feat_idx]}/{len(test_texts)} test samples | {freq_pct:.2f}%) ==="
        )
        for rank, doc_idx in enumerate(top_doc_indices, 1):
            lbl_id = int(labels_np[doc_idx])
            print(f"  {rank}. [act={acts[doc_idx]:.3f} | label={lbl_id}] {test_texts[doc_idx]}")


def main() -> None:
    configure_logging()
    settings = load_settings()

    key = jax.random.key(settings.random_seed)
    data_key, _ = jax.random.split(key)
    np_rng = np.random.default_rng(np.asarray(jax.random.key_data(data_key)))

    embeddings_model = SentenceTransformer(
        "hotchpotch/bekko-embedding-v1-a8m",
        backend="openvino",
        model_kwargs={"file_name": "openvino/openvino_model.xml"},
    )
    data = Data(model=embeddings_model, rng=np_rng, sigma=settings.data.sigma_noise)

    latent_dims = [768, 1536, 3072]
    k = 20
    base_ckpt_dir = Path("./checkpoints").resolve()
    models = {}
    results = {}

    for m in latent_dims:
        ckpt_dir = base_ckpt_dir / f"sae_{m}"
        if not ckpt_dir.exists():
            log.error(f"Weights not found")
            return

        sae_m = load_sae_from_checkpoint(
            d_in=data.dimension, m_latents=m, k=k, ckpt_dir=ckpt_dir
        )
        models[m] = sae_m
        metrics = evaluate_sae(sae_m, data.test_embeddings)
        results[m] = metrics

        log.info(
            f"Evaluated SAE m={m} ({m // data.dimension}x)",
            test_r2=f"{metrics['r2']:.4f}",
            test_mse=f"{metrics['mse']:.5f}",
            l0=f"{metrics['l0']:.1f}",
            test_dead_pct=f"{metrics['dead_fraction'] * 100:.2f}%",
        )

    settings.plotting.output_dir.mkdir(parents=True, exist_ok=True)
    plot_sae_results(
        latent_dims=latent_dims,
        results=results,
        save_path=settings.plotting.output_dir / "sae_evaluation.svg",
    )

    inspect_top_features(
        sae=models[1536],
        test_embeddings=data.test_embeddings,
        test_texts=data.test_texts,
        test_labels=data.test_labels,
        num_features=6,
        top_n_sentences=5,
    )
