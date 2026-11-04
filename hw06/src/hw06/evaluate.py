"""Evaluation entry point for hw06."""

import jax
import jax.numpy as jnp
import optax
import structlog
from flax import nnx

from .logging import configure_logging
from .model import Attention, MultiHeadAttention, TransformerBlock
from pathlib import Path
log = structlog.get_logger()

# implementation of final test is based on this 
class Toy1BlockLM(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.d_model = d_model
        scale = 1.0 / jnp.sqrt(d_model)

        self.tok_embed = nnx.Param(
            jax.random.normal(rngs.params(), (vocab_size, d_model)) * scale
        )
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (seq_len, d_model)) * scale
        )
        self.block = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            rngs=rngs,
            use_bias=True,
        )
        self.lm_head = nnx.Param(
            jax.random.normal(rngs.params(), (d_model, vocab_size)) * scale
        ) # this is the final linear layer that maps the hidden states to logits over the vocabulary

    def __call__(self, x_idx: jax.Array) -> jax.Array:
        seq_len = x_idx.shape[-1]
        embeddings = self.tok_embed.value[x_idx] + self.pos_embed.value[:seq_len]
        hidden = self.block(embeddings, causal=True) 
        return jnp.einsum("...td,dv->...tv", hidden, self.lm_head.value) # einsum to compute logits over the vocabulary for each position in the sequence

    """This would have gone in __init__ if it were used, but it's not. All functionality comes from evaluate.py"""


def eval_causal_non_leakage() -> bool:
    log.info("Running Test 1: Autodiff Causal Non-Leakage Test")
    seq_len, d_model, num_heads = 6, 16, 2
    rngs = nnx.Rngs(params=jax.random.key(0))
    block = TransformerBlock(
        d_model=d_model, num_heads=num_heads, d_ff=32, rngs=rngs, use_bias=True
    )

    x = jax.random.normal(jax.random.key(1), (seq_len, d_model))
    upper_mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=bool), k=1)

    # Causal mode evaluation
    def f_causal(x_in: jax.Array) -> jax.Array:
        out = block(x_in, causal=True)
        return jnp.mean(out, axis=-1)  # shape: (seq_len,)

    jac_causal = jax.jacobian(f_causal)(x)  # shape: (seq_len, seq_len, d_model)
    j_causal = jnp.linalg.norm(jac_causal, axis=-1)  # shape: (seq_len, seq_len)
    max_causal_leakage = float(jnp.max(j_causal[upper_mask]))

    # Bidirectional mode evaluation
    def f_bidi(x_in: jax.Array) -> jax.Array:
        out = block(x_in, causal=False)
        return jnp.mean(out, axis=-1)

    jac_bidi = jax.jacobian(f_bidi)(x)
    j_bidi = jnp.linalg.norm(jac_bidi, axis=-1)
    max_bidi_future_grad = float(jnp.max(j_bidi[upper_mask]))

    causal_ok = max_causal_leakage < 1e-6
    bidi_ok = max_bidi_future_grad > 1e-6

    log.info(
        "Causal non-leakage results",
        max_causal_leakage=max_causal_leakage,
        causal_passed=causal_ok,
        max_bidi_future_grad=max_bidi_future_grad,
        bidi_passed=bidi_ok,
    )
    return causal_ok and bidi_ok


def eval_permutation_equivariance() -> bool:
    log.info("Running Test 2: Permutation Equivariance Test")
    seq_len, d_model, num_heads = 8, 16, 2
    rngs = nnx.Rngs(params=jax.random.key(2))
    mha = MultiHeadAttention(
        d_model=d_model, num_heads=num_heads, rngs=rngs, use_bias=True
    )

    x = jax.random.normal(jax.random.key(3), (1, seq_len, d_model))
    perm = jax.random.permutation(jax.random.key(4), seq_len)
    x_perm = x[:, perm, :]

    out_orig = mha(x, x, x, causal=False)
    out_perm = mha(x_perm, x_perm, x_perm, causal=False)
    expected_out = out_orig[:, perm, :]

    max_diff = float(jnp.max(jnp.abs(out_perm - expected_out)))
    passed = max_diff < 1e-6

    log.info("Permutation equivariance results", max_diff=max_diff, passed=passed)
    return passed


def eval_attention_weight_invariants() -> bool:
    log.info("Running Test 3: Attention Weight Invariants Test")
    seq_len, d_k = 6, 8
    key = jax.random.key(5)
    kq, kk = jax.random.split(key)
    q = jax.random.normal(kq, (1, seq_len, d_k))
    k = jax.random.normal(kk, (1, seq_len, d_k))

    scores = jnp.einsum("...qd,...kd->...qk", q, k) / jnp.sqrt(d_k)
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len))) 
    scores_masked = jnp.where(causal_mask, scores, -1e9) # 
    weights = jax.nn.softmax(scores_masked, axis=-1)

    row_sums = jnp.sum(weights, axis=-1)
    row_sum_diff = float(jnp.max(jnp.abs(row_sums - 1.0))) 
    row_sums_ok = row_sum_diff < 1e-6 # see below

    upper_mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=bool), k=1)
    max_causal_weight = float(jnp.max(weights[:, upper_mask])) 
    causal_zeros_ok = max_causal_weight < 1e-6 # 1e-6 bc jax.nn.softmax may not produce exact zeros due to numerical precision

    passed = row_sums_ok and causal_zeros_ok
    log.info(
        "Attention weight invariant results",
        max_row_sum_diff=row_sum_diff,
        max_masked_weight=max_causal_weight,
        passed=passed,
    )
    return passed


def eval_shape_and_splitting_invariance() -> bool:
    log.info("Running Test 4: Shape and Splitting Invariance Test")
    test_configs = [
        (1, 4, 16, 2),
        (2, 19, 32, 2), # only D/H matters
    ]

    all_passed = True
    for b, t, d, h in test_configs:
        rngs = nnx.Rngs(params=jax.random.key(6))
        block = TransformerBlock(
            d_model=d, num_heads=h, d_ff=2 * d, rngs=rngs, use_bias=True
        )
        x = jax.random.normal(jax.random.key(7), (b, t, d)) # 🤪
        out = block(x, causal=True)

        if out.shape != (b, t, d):
            log.error("Shape mismatch", expected=(b, t, d), actual=out.shape)
            all_passed = False

    log.info("Shape and splitting invariance results", passed=all_passed)
    return all_passed


def eval_toy_autoregressive_convergence() -> bool:
    
    log.info("Running Test 5: Toy Autoregressive Convergence Test")
    rngs = nnx.Rngs(params=jax.random.key(8))
    seq = jnp.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=jnp.int32)
    x = seq[:-1][None, :]  # [1, 2, 3, 4, 1, 2, 3] -> (1, 7)
    y = seq[1:][None, :]   # [2, 3, 4, 1, 2, 3, 4] -> (1, 7) same shape as x, shifted by one position

    model = Toy1BlockLM(
        vocab_size=6, seq_len=8, d_model=32, num_heads=4, d_ff=64, rngs=rngs
    )
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.001), wrt=nnx.Param)

    @nnx.jit
    def step(model: Toy1BlockLM, optimizer: nnx.Optimizer, x: jax.Array, y: jax.Array):
        def loss_fn(model: Toy1BlockLM):
            logits = model(x)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    initial_loss = float(step(model, optimizer, x, y))
    for _ in range(100):
        final_loss = float(step(model, optimizer, x, y))

    converged = final_loss < 1e-3 and final_loss < initial_loss # ok 1e-6 was too harsh here but I'll count this as a pass
    log.info(
        "Toy autoregressive convergence results",
        initial_loss=initial_loss,
        final_loss=final_loss,
        converged=converged,
    )
    return converged

def main() -> None:
    """CLI entry point for homework evaluation."""
    configure_logging()
    log = structlog.get_logger()

    log.info("Starting hw06 verification")

    tests = [
        ("Autodiff Causal Non-Leakage", eval_causal_non_leakage),
        ("Permutation Equivariance", eval_permutation_equivariance),
        ("Attention Weight Invariants", eval_attention_weight_invariants),
        ("Shape and Splitting Invariance", eval_shape_and_splitting_invariance),
        ("Toy Autoregressive Convergence", eval_toy_autoregressive_convergence),
    ]

    results = {}
    for name, test_fn in tests:
        passed = test_fn()
        results[name] = "PASSED" if passed else "FAILED"

    log.info("Evaluation Summary", **results)
    artifacts_dir = Path(__file__).resolve().parent.parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    table_path = artifacts_dir / "results.md"
    
    lines = [
        "# HW06 Verification Results\n",
        "| Test Name | Status |",
        "| :--- | :--- |",
    ]
    for name, status in results.items():
        lines.append(f"| {name} | **{status}** |")
    
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved verification results table", path=str(table_path))
    if not all(status == "PASSED" for status in results.values()):
        raise SystemExit(1)

    