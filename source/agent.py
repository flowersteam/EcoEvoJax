from flax import linen as nn

import logging
import jax
import jax.numpy as jnp

import itertools
import functools

from typing import Tuple, Callable, List, Optional, Iterable, Any
from flax.struct import dataclass
from evojax.task.base import TaskState
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.util import create_logger
from evojax.util import get_params_format_fn


class MetaRNN_bcppr(nn.Module):
    output_size: int
    out_fn: str
    hidden_layers: list
    encoder_in: bool
    encoder_layers: list

    def setup(self):

        self._num_micro_ticks = 1
        self._lstm = nn.recurrent.LSTMCell()
        self.convs = [nn.Conv(features=4, kernel_size=(3, 3),strides=2),nn.Conv(features=8, kernel_size=(3, 3),strides=2)]

        self._hiddens = [(nn.Dense(size)) for size in self.hidden_layers]
        # self._encoder=nn.Dense(64)
        self._output_proj = nn.Dense(self.output_size)
        if (self.encoder_in):
            self._encoder = [(nn.Dense(size)) for size in self.encoder_layers]

    def __call__(self, h, c, inputs: jnp.ndarray, last_action: jnp.ndarray, reward: jnp.ndarray):
        carry = (h, c)
        # todo replace with scan
        # inputs=self._encoder(inputs)
        out = inputs
        for conv in self.convs:
            out = conv(out)
            out = nn.relu(out)
            out = nn.avg_pool(out, window_shape=(2, 2), strides=(1, 1))

        out = jnp.ravel(out)

        if (self.encoder_in):
            for layer in self._encoder:
                out = jax.nn.tanh(layer(out))

        inputs_encoded = jnp.concatenate([out, last_action, reward])

        for _ in range(self._num_micro_ticks):
            carry, out = self._lstm(carry, inputs_encoded)
        out = jnp.concatenate([inputs_encoded, out])
        for layer in self._hiddens:
            out = jax.nn.tanh(layer(out))
        out = self._output_proj(out)

        h, c = carry
        if self.out_fn == 'tanh':
            out = nn.tanh(out)
        elif self.out_fn == 'softmax':
            out = nn.softmax(out, axis=-1)
        else:
            if (self.out_fn != 'categorical'):
                raise ValueError(
                    'Unsupported output activation: {}'.format(self.out_fn))
        return h, c, out


@dataclass
class metaRNNPolicyState_bcppr(PolicyState):
    lstm_h: jnp.array
    lstm_c: jnp.array
    keys: jnp.array


class MetaRnnPolicy_bcppr(PolicyNetwork):

    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_act_fn: str = "categorical",
                 hidden_layers: list = [],
                 encoder: bool = False,
                 encoder_layers: list = [32, 32],
                 logger: logging.Logger = None):

        if logger is None:
            self._logger = create_logger(name='MetaRNNolicy')
        else:
            self._logger = logger
        model = MetaRNN_bcppr(output_dim, out_fn=output_act_fn, hidden_layers=hidden_layers, encoder_in=encoder,
                              encoder_layers=encoder_layers)
        self.params = model.init(jax.random.PRNGKey(0), jnp.zeros((hidden_dim)), jnp.zeros((hidden_dim)),
                                 jnp.zeros(input_dim), jnp.zeros([output_dim]), jnp.zeros([1]))

        self.num_params, format_params_fn = get_params_format_fn(self.params)
        self._logger.info('MetaRNNPolicy.num_params = {}'.format(self.num_params))
        self.hidden_dim = hidden_dim
        self._format_params_fn = jax.jit(jax.vmap(format_params_fn))
        self._forward_fn = jax.jit(jax.vmap(model.apply))

    def reset(self, states: TaskState) -> PolicyState:
        """Reset the policy.
        Args:
            TaskState - Initial observations.
        Returns:
            PolicyState. Policy internal states.
        """
        keys = jax.random.split(jax.random.PRNGKey(0), states.obs.shape[0])
        h = jnp.zeros((states.obs.shape[0], self.hidden_dim))
        c = jnp.zeros((states.obs.shape[0], self.hidden_dim))
        return metaRNNPolicyState_bcppr(keys=keys, lstm_h=h, lstm_c=c)

    def reset_b(self, obs: jnp.array) -> PolicyState:
        """Reset the policy.
        Args:
            TaskState - Initial observations.
        Returns:
            PolicyState. Policy internal states.
        """
        keys = jax.random.split(jax.random.PRNGKey(0), obs.shape[0])
        h = jnp.zeros((obs.shape[0], self.hidden_dim))
        c = jnp.zeros((obs.shape[0], self.hidden_dim))
        return metaRNNPolicyState_bcppr(keys=keys, lstm_h=h, lstm_c=c)

    def get_actions(self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState):
        params = self._format_params_fn(params)
        h, c, out = self._forward_fn(params, p_states.lstm_h, p_states.lstm_c, t_states.obs, t_states.last_actions,
                                     t_states.rewards)
        return out, metaRNNPolicyState_bcppr(keys=p_states.keys, lstm_h=h, lstm_c=c)