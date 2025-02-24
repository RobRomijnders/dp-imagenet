import jax
import jax.numpy as jnp

def construct_logit_clip_fn(clip_fn):
  if clip_fn == "none":
    return lambda x: x
  elif clip_fn == "tanh":
    return lambda x: jnp.tanh(x)
  elif clip_fn == "sigmoid":
    return lambda x: 2 * jax.nn.sigmoid(x)
  elif clip_fn == "blf":
    return lambda x: 2 * (x * jax.nn.sigmoid(x) + jax.nn.sigmoid(x) - x * jax.nn.sigmoid(x)**2) - 1
  else:
    raise ValueError(f"Unknown clip_fn {clip_fn}")
