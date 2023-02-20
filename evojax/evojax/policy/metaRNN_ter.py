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

class MetaRNN_t(nn.Module):
    output_size:int
    out_fn:str
    hidden_layers:list
    encoder_in:bool
    encoder_size:int
    def setup(self):

        self._num_micro_ticks = 1
        self._lstm = nn.recurrent.LSTMCell()
        
        self._hiddens=[(nn.Dense(size)) for size in self.hidden_layers]
        self._output_proj = nn.Dense(self.output_size)
        if(self.encoder_in):
            self._encoder=nn.Dense(self.encoder_size)
        



    def __call__(self,h,c, inputs: jnp.ndarray):
        carry=(h,c)
        # todo replace with scan
        
        if(self.encoder_in):
            inputs=jax.nn.tanh(self._encoder(inputs))
        for _ in range(self._num_micro_ticks):
            carry,out= self._lstm(carry,inputs)
        out=jnp.concatenate([out,inputs],axis=-1)
        
        for layer in self._hiddens:
            out=jax.nn.tanh(layer(out))
            
        out = self._output_proj(out)
        h,c=carry
        if self.out_fn == 'tanh':
            out = nn.tanh(out)
        elif self.out_fn == 'softmax':
            out = nn.softmax(out, axis=-1)
        else:
          if(self.out_fn!='categorical'):
            raise ValueError(
                'Unsupported output activation: {}'.format(self.out_fn))
        return h,c,out

@dataclass
class metaRNNPolicyState_t(PolicyState):
    lstm_h:jnp.array
    lstm_c:jnp.array
    keys:jnp.array






class MetaRnnPolicy_t(PolicyNetwork):

    def __init__(self,input_dim: int,
                    hidden_dim: int,
                    output_dim: int,
                    output_act_fn: str ="categorical",
                    hidden_layers: list= [64,64],
                    encoder: bool=False,
                    encoder_size:int=32,
                    logger: logging.Logger=None):


            if logger is None:
                        self._logger = create_logger(name='MetaRNNolicy_ter')
            else:
                        self._logger = logger
            model=MetaRNN_t(output_dim,out_fn=output_act_fn,hidden_layers=hidden_layers,encoder_in=encoder,encoder_size=encoder_size)
            self.params = model.init(jax.random.PRNGKey(0),jnp.zeros((hidden_dim)),jnp.zeros((hidden_dim)), jnp.zeros([ input_dim]))
            
            self.num_params, format_params_fn = get_params_format_fn(self.params)
            self._logger.info('MetaRNNPolicy.num_params = {}'.format(self.num_params))
            self.hidden_dim=hidden_dim
            self._format_params_fn = (jax.vmap(format_params_fn))
            self._forward_fn = (jax.vmap(model.apply))

    def reset(self, states: TaskState) -> PolicyState:
        """Reset the policy.
        Args:
            TaskState - Initial observations.
        Returns:
            PolicyState. Policy internal states.
        """
        keys = jax.random.split(jax.random.PRNGKey(0), states.obs.shape[0])
        h= jnp.zeros((states.obs.shape[0],self.hidden_dim))
        c= jnp.zeros((states.obs.shape[0],self.hidden_dim))
        return metaRNNPolicyState_t(keys=keys,lstm_h=h,lstm_c=c)



    def get_actions(self,t_states: TaskState,params: jnp.ndarray,p_states: PolicyState):
        params = self._format_params_fn(params)
        inp=jnp.concatenate([t_states.obs,t_states.last_action,t_states.reward],axis=-1)
        print(inp.shape)
        h,c,out=self._forward_fn(params,p_states.lstm_h,p_states.lstm_c, inp)
        return out, metaRNNPolicyState_t(keys=p_states.keys,lstm_h=h,lstm_c=c)

