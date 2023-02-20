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

class MetaRNN_b2(nn.Module):
    output_size:int
    out_fn:str
    hidden_layers:list
    encoder:bool
    encoder_size:int
    def setup(self):

        self._num_micro_ticks = 1
        self._lstm_1 = nn.recurrent.LSTMCell()
        self._lstm_2 = nn.recurrent.LSTMCell()
        
        self._hiddens=[(nn.Dense(size)) for size in self.hidden_layers]
        self._output_proj = nn.Dense(self.output_size)
        if(self.encoder):
            self._encoder=nn.Dense(self.encoder_size)
        



    def __call__(self,h_1,c_1,h_2,c_2, inputs: jnp.ndarray,last_action,reward):
        carry_1=(h_1,c_1)
        carry_2=(h_2,c_2)
        # todo replace with scan
        if(self.encoder):
            inputs=jax.nn.tanh(self._encoder(inputs))
        
        inputs=jnp.concatenate([inputs,reward,last_action],axis=-1)
        for _ in range(self._num_micro_ticks):
            carry_1,out= self._lstm_1(carry_1,inputs)
        
        out=jnp.concatenate([out,inputs],axis=-1)    
        #out=jnp.concatenate([out,reward,last_action],axis=-1)
        
        carry_2,out=self._lstm_2(carry_2,out)
        
        #out=jnp.concatenate([out1,out],axis=-1)
        for layer in self._hiddens:
            out=(layer(out))
            
        out = self._output_proj(out)
        
        if self.out_fn == 'tanh':
            out = nn.tanh(out)
        elif self.out_fn == 'softmax':
            out = nn.softmax(out, axis=-1)
        else:
          if(self.out_fn!='categorical'):
            raise ValueError(
                'Unsupported output activation: {}'.format(self.out_fn))
        h_1,c_1=carry_1
        h_2,c_2=carry_2
        return h_1,c_1,h_2,c_2,out

@dataclass
class metaRNNPolicyState_b2(PolicyState):
    h_1:jnp.array
    c_1:jnp.array
    h_2:jnp.array
    c_2:jnp.array
    keys:jnp.array






class MetaRnnPolicy_b2(PolicyNetwork):

    def __init__(self,input_dim: int,
                    hidden_dim_1: int,
                    hidden_dim_2: int,
                    output_dim: int,
                    hidden_layers: list= [64],
                    encoder: bool=False,
                    encoder_size:int=32,
                    output_act_fn: str ="categorical",
                    logger: logging.Logger=None):


            if logger is None:
                        self._logger = create_logger(name='MetaRNNolicy_2layers')
            else:
                        self._logger = logger
            model=MetaRNN_b2(output_dim,out_fn=output_act_fn,hidden_layers=hidden_layers,encoder=encoder,encoder_size=encoder_size)
            self.params = model.init(jax.random.PRNGKey(0),jnp.zeros((hidden_dim_1)),jnp.zeros((hidden_dim_1)),jnp.zeros((hidden_dim_2)),jnp.zeros((hidden_dim_2)), jnp.zeros((input_dim-output_dim-1)),jnp.zeros((output_dim)),jnp.zeros((1, )))
            
            self.num_params, format_params_fn = get_params_format_fn(self.params)
            self._logger.info('MetaRNNPolicy.num_params = {}'.format(self.num_params))
            self.hidden_dim_1=hidden_dim_1
            self.hidden_dim_2=hidden_dim_2
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
        h_1= jnp.zeros((states.obs.shape[0],self.hidden_dim_1))
        c_1= jnp.zeros((states.obs.shape[0],self.hidden_dim_1))
        h_2= jnp.zeros((states.obs.shape[0],self.hidden_dim_2))
        c_2= jnp.zeros((states.obs.shape[0],self.hidden_dim_2))
        return metaRNNPolicyState_b2(keys=keys,h_1=h_1,c_1=c_1,h_2=h_2,c_2=c_2)



    def get_actions(self,t_states: TaskState,params: jnp.ndarray,p_states: PolicyState):
        params = self._format_params_fn(params)
        #inp=jnp.concatenate([t_states.obs,t_states.last_action,t_states.reward],axis=-1)

        h_1,c_1,h_2,c_2,out=self._forward_fn(params,p_states.h_1,p_states.c_1,p_states.h_2,p_states.c_2, t_states.obs,t_states.last_action,t_states.reward)
        return out, metaRNNPolicyState_b2(keys=p_states.keys,h_1=h_1,c_1=c_1,h_2=h_2,c_2=c_2)

