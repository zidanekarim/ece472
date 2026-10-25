from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class LayerNorm(nnx.Module):

    def __init__(self, d_model, *, eps=1e-6):
        self.eps = eps
        self.gamma = nnx.Param(jnp.ones((d_model,)))
        self.beta = nnx.Param(jnp.zeros((d_model,)))

    def __call__(self, x: jax.Array):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        normalized_x = (x - mean) / jnp.sqrt(variance + self.eps)
        return self.gamma.value * normalized_x + self.beta.value


class FFN(nnx.Module):
    def __init__(self, d_model, d_ff, *, rngs, use_bias):
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias

        scale = 1.0 / jnp.sqrt(d_model)
        self.w1 = nnx.Param(
            jax.random.normal(rngs.params(), (d_model, d_ff)) * scale
        )
        self.w2 = nnx.Param(
            jax.random.normal(rngs.params(), (d_ff, d_model)) * scale
        )

        if self.use_bias:
            self.b1 = nnx.Param(jnp.zeros((d_ff,)))
            self.b2 = nnx.Param(jnp.zeros((d_model,)))
        else:
            self.b1 = None
            self.b2 = None

    def __call__(self, x: jax.Array):
        x = jnp.einsum('...qd,df->...qf', x, self.w1.value)
        if self.use_bias:
            x += self.b1.value
        x = jax.nn.relu(x)
        x = jnp.einsum('...qf,fd->...qd', x, self.w2.value)
        if self.use_bias:
            x += self.b2.value
        return x


class Attention(nnx.Module):
    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array | None, causal: bool=False):
        d_k = q.shape[-1] # dimension of the key
        t_q = q.shape[-2] # sequence length of the query
        t_k = k.shape[-2] # sequence length of the key
        mask = mask if mask is not None else jnp.ones((t_q, t_k))

        scale = 1.0 / jnp.sqrt(d_k)
        scores = jnp.einsum('...qd,...kd->...qk', q, k) * scale

        if causal:
            causal_mask = jnp.tril(jnp.ones((t_q, t_k))) # lower triangular matrix
            mask = mask * causal_mask
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9) # apply mask

        weights = nnx.softmax(scores, axis=-1)
        return jnp.einsum('...qk,...vd->...qd', weights, v)


class MultiHeadAttention(nnx.Module):
    """MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O"""
    """and head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)"""
    def __init__(self, d_model, num_heads, *, rngs, use_bias):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_bias = use_bias

        scale = 1.0 / jnp.sqrt(self.d_model)
        self.w_q = nnx.Param(
            jax.random.normal(rngs.params(), (d_model, d_model)) * scale
        )
        self.w_k = nnx.Param(
            jax.random.normal(rngs.params(), (d_model, d_model)) * scale
        )
        self.w_v = nnx.Param(
            jax.random.normal(rngs.params(), (d_model, d_model)) * scale
        )
        self.w_o = nnx.Param(
            jax.random.normal(rngs.params(), (d_model, d_model)) * scale
        )

        if self.use_bias:
            self.b_q = nnx.Param(jnp.zeros((d_model,))) # query bias
            self.b_k = nnx.Param(jnp.zeros((d_model,))) # key bias
            self.b_v = nnx.Param(jnp.zeros((d_model,))) # value bias
            self.b_o = nnx.Param(jnp.zeros((d_model,))) # output bias
        else:
            self.b_q = None # define for python type checking
            self.b_k = None
            self.b_v = None
            self.b_o = None
        self.attention = Attention() # initialize the attention module

    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array | None = None, causal: bool = False):
        # Linear projections
        q_proj = jnp.einsum('...qd,df->...qf', q, self.w_q.value)
        k_proj = jnp.einsum('...kd,df->...kf', k, self.w_k.value)
        v_proj = jnp.einsum('...vd,df->...vf', v, self.w_v.value)

        if self.use_bias:
            q_proj += self.b_q.value
            k_proj += self.b_k.value
            v_proj += self.b_v.value

        q_proj = q_proj.reshape(q.shape[:-1] + (self.num_heads, self.head_dim))
        k_proj = k_proj.reshape(k.shape[:-1] + (self.num_heads, self.head_dim))
        v_proj = v_proj.reshape(v.shape[:-1] + (self.num_heads, self.head_dim))

        # Attention(Q W_i^Q, K W_i^K, V W_i^V)
        attn_output = self.attention(q_proj, k_proj, v_proj, mask=mask, causal=causal)

        # MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        attn_output = attn_output.reshape(attn_output.shape[:-2] + (self.d_model,))
        output = jnp.einsum('...qd,df->...qf', attn_output, self.w_o.value)

        if self.use_bias:
            output += self.b_o.value

        return output









class TransformerBlock(nnx.Module):
    """
    y = x + Attention(LayerNorm(x))
    Output = y + FFN(LayerNorm(y))
    """

    def __init__(self, d_model, num_heads, d_ff, *, rngs, use_bias, eps=1e-5):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_bias = use_bias
        self.eps = eps

        self.layer1 = LayerNorm(d_model, eps=eps)
        self.layer2 = LayerNorm(d_model, eps=eps)

        self.mha = MultiHeadAttention(d_model, num_heads, rngs=rngs, use_bias=use_bias)
        self.ffn = FFN(d_model, d_ff, rngs=rngs, use_bias=use_bias)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None, causal: bool = False):
        x_norm = self.layer1(x)
        attn_output = self.mha(x_norm, x_norm, x_norm, mask=mask, causal=causal)
        y = x + attn_output

        y_norm = self.layer2(y)
        ffn_output = self.ffn(y_norm)
        output = y + ffn_output

        return output
