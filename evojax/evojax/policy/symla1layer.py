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


        


#@dataclass
#class LSTMState:
#    h: jnp.array
#    c: jnp.array

@dataclass
class LayerState:
    #lstm_state: LSTMState
    lstm_h: jnp.array
    lstm_c: jnp.array
    fwd_msg: jnp.ndarray
    bwd_msg: jnp.ndarray
    
@dataclass
class SymlaPolicyState:
    layerState:LayerState
    keys:jnp.array






class VSMLRNN(nn.Module):            
    num_micro_ticks:int
    msg_size:int
    output_idx:int
    layer_norm:bool
    reduce:str
    output_fn:str

    def forward_rnn(self,inc_fwd_msg: jnp.ndarray, inc_bwd_msg: jnp.ndarray,fwd_msg:jnp.ndarray,bwd_msg:jnp.ndarray,reward:jnp.ndarray,
                 h:jnp.array,c:jnp.array):
        carry=(h,c)
        inputs = jnp.concatenate([inc_fwd_msg,inc_bwd_msg,fwd_msg, bwd_msg,reward], axis=-1)
        carry,outputs= self._lstm(carry,inputs)
        h,c=carry
        fwd_msg = self._fwd_messenger(outputs)
        bwd_msg = self._bwd_messenger(outputs)
        # replace layer norm
        if self.layer_norm:
            fwd_msg = self._fwd_layer_norm(fwd_msg)
            bwd_msg = self._bwd_layer_norm(bwd_msg)
        return h,c,fwd_msg, bwd_msg

    def setup(self):
      self._lstm = nn.recurrent.LSTMCell()
      self._fwd_messenger = nn.Dense(self.msg_size)
      self._bwd_messenger = nn.Dense(self.msg_size)
      if self.layer_norm:
            self._fwd_layer_norm = nn.LayerNorm((-1,), use_scale=True, use_bias=True)
            self._bwd_layer_norm = nn.LayerNorm((-1,), use_scale=True, use_bias=True)

      dense_vsml= jax.vmap(self.forward_rnn, in_axes=( 0,None,None,0,None,0,0))
      self.dense_vsml = jax.vmap(dense_vsml, in_axes=(None,0,0,None,None,0,0))
      if(self.reduce=="mean"):
          self.reduce_fn=jnp.mean





    def __call__(self, layer_state: LayerState, reward: jnp.ndarray,last_action: jnp.ndarray,
              inp: jnp.ndarray):


        inp=jnp.expand_dims(inp,axis=-1)
        last_action=jnp.expand_dims(last_action,axis=-1)
        incoming_fwd_msg = jnp.pad(inp,((0,0),(0,self.msg_size - 1)))
        incoming_bwd_msg = jnp.pad(last_action, ((0, 0), (0, self.msg_size - 1)))
        ls=layer_state
        lstm_h,lstm_c, fwd_msg, bwd_msg = (ls.lstm_h,ls.lstm_c,
                                        ls.fwd_msg,
                                        ls.bwd_msg)
        for _ in range(self.num_micro_ticks):
            
            lstm_h,lstm_c,fwd_msg, bwd_msg = self.dense_vsml( incoming_fwd_msg,incoming_bwd_msg, fwd_msg, bwd_msg, reward,lstm_h,lstm_c)
            fwd_msg = self.reduce_fn(fwd_msg, axis=1)
            bwd_msg = self.reduce_fn(bwd_msg, axis=0)
        layer_state=LayerState(lstm_h=lstm_h,lstm_c=lstm_c,fwd_msg=fwd_msg,bwd_msg=bwd_msg)
        out = fwd_msg[:, self.output_idx]
        
        if self.output_fn == 'tanh':
            out = nn.tanh(out)
        elif self.output_fn == 'softmax':
            out = nn.softmax(out, axis=-1)
        else:
          if(self.output_fn!='categorical'):
            raise ValueError(
                'Unsupported output activation: {}'.format(self.out_fn))
        

        return layer_state, out
    
    









class SymLA_Policy(PolicyNetwork):
    def __init__(self,input_dim: int,
                    msg_dim: int,
                    hidden_dim:int,
                    output_dim: int,
                    num_micro_ticks: int,
                    output_act_fn: str ="tanh",
                    logger: logging.Logger=None):


            if logger is None:
                        self._logger = create_logger(name='SymLAPolicy')
            else:
                        self._logger = logger
            model=VSMLRNN(num_micro_ticks=num_micro_ticks,msg_size=msg_dim,output_idx=0,output_fn=output_act_fn,reduce="mean",layer_norm=False)
            
    

            self.hidden_dim=hidden_dim
            self.msg_dim=msg_dim
            self.input_dim=input_dim
            self.output_dim=output_dim
            
            self._forward_fn = (jax.vmap(model.apply))
            
            
            #init
            h= jnp.zeros((self.output_dim,self.input_dim,self.hidden_dim))
            c= jnp.zeros((self.output_dim,self.input_dim,self.hidden_dim))
            fwd_msg=jnp.zeros((self.output_dim,self.msg_dim))
            bwd_msg=jnp.zeros((self.input_dim,self.msg_dim))
            layer_state=LayerState(lstm_h=h,lstm_c=c,fwd_msg=fwd_msg,bwd_msg=bwd_msg)
            
            reward=jnp.zeros((1))
            
            last_action=jnp.zeros((output_dim))
            inp=jnp.zeros((input_dim))
    
            self.params = model.init(jax.random.PRNGKey(0),layer_state=layer_state,reward=reward,last_action=last_action,inp=inp)
            self.num_params, format_params_fn = get_params_format_fn(self.params)
            self._logger.info('SymLAPolicy.num_params = {}'.format(self.num_params))
            self._format_params_fn = (jax.vmap(format_params_fn))

    def reset(self, states: TaskState) -> PolicyState:
        """Reset the policy.
        Args:
            TaskState - Initial observations.
        Returns:
            PolicyState. Policy internal states.
        """
        keys = jax.random.split(jax.random.PRNGKey(0), states.obs.shape[0])
        h= jnp.zeros((states.obs.shape[0],self.output_dim,self.input_dim,self.hidden_dim))
        c= jnp.zeros((states.obs.shape[0],self.output_dim,self.input_dim,self.hidden_dim))
        fwd_msg=jnp.zeros((states.obs.shape[0],self.output_dim,self.msg_dim))
        bwd_msg=jnp.zeros((states.obs.shape[0],self.input_dim,self.msg_dim))
        layer_state=LayerState(lstm_h=h,lstm_c=c,fwd_msg=fwd_msg,bwd_msg=bwd_msg)
        return SymlaPolicyState(layerState=layer_state,keys=keys)



    def get_actions(self,t_states: TaskState,params: jnp.ndarray,p_states: PolicyState):
        params = self._format_params_fn(params)
        layer_state,out=self._forward_fn(params,p_states.layerState, t_states.reward,t_states.last_action,t_states.obs)
        return out, SymlaPolicyState(keys=p_states.keys,layerState=layer_state)









