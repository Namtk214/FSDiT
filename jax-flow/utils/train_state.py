"""Flax TrainState with optimizer, EMA, and serialization."""

import flax
import flax.linen as nn
import jax
import optax
import functools
from typing import Any, Callable

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


def target_update(model, target, tau):
    """EMA update: new_target = tau * model + (1 - tau) * target."""
    new_params = jax.tree.map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target.params
    )
    return target.replace(params=new_params)


class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Callable = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None):
        opt_state = tx.init(params) if tx else None
        return cls(
            step=1, apply_fn=model_def.apply, model_def=model_def,
            params=params, tx=tx, opt_state=opt_state,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        params = params or self.params
        if isinstance(method, str):
            method = getattr(self.model_def, method)
        return self.apply_fn({"params": params}, *args, method=method, **kwargs)

    def save(self):
        return {'params': self.params, 'opt_state': self.opt_state, 'step': self.step}

    def load(self, data):
        return self.replace(**data)
