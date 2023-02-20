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


        
def dense(sub_rnn, reduce_fn,inc_fwd_msg,inc_bwd_msg, fwd_msg, bwd_msg, reward,lstm_h,lstm_c):
    """Dense VSML layer.
    Args:
        sub_rnn: Rnn taking as inputs fwd_msg, bwd_msg, state
        reduce_fn: A function used to merge messages
        fwd_msg: Shape [in_channel, msg_size]
        bwd_msg: Shape [out_channel, msg_size]
        state: Shape [ic, oc, slow_size]
    """
    batched = jax.vmap(sub_rnn.apply, in_axes=(None, None, 0,None,None,0,None,0,0))
    batched = jax.vmap(batched,       in_axes=(None, None, None,0,0,None,None,0,0))
    h,c,fwd_msg, bwd_msg = batched(fwd_msg, bwd_msg, state)
    fwd_msg = reduce_fn(fwd_msg, axis=0)
    bwd_msg = reduce_fn(bwd_msg, axis=1)
    return h,c,fwd_msg, bwd_msg

#@dataclass
#class LSTMState:
#    h: jnp.array
#    c: jnp.array

@dataclass
class LayerState:
    #lstm_state: LSTMState
    lstm_h: jnp.array
    lstm_c: jnp.array
    incoming_fwd_msg: jnp.ndarray
    incoming_bwd_msg: jnp.ndarray
    
@dataclass
class symlaPolicyState:
    layerStates:List[LayerState]
    keys:jnp.array


class SubRNN(nn.Module):

    def __init__(self, slow_size: int, msg_size: int, init_rand_proportion: float, layer_norm: bool):
        super().__init__()
        self._lstm = nn.recurrent.LSTMCell(slow_size)
        self._fwd_messenger = nn.Dense(msg_size)
        self._bwd_messenger = nn.Dense(msg_size)
        #replace layer norm
        if layer_norm:
            self._fwd_layer_norm = nn.LayerNorm((-1,), use_scale=True, use_bias=True)
            self._bwd_layer_norm = nn.LayerNorm((-1,), use_scale=True, use_bias=True)
        self.msg_size = msg_size
        self._init_rand_proportion = init_rand_proportion
        self._use_layer_norm = layer_norm
        
        

    def __call__(self, inc_fwd_msg: jnp.ndarray, inc_bwd_msg: jnp.ndarray,
    		 fwd_msg:jnp.ndarray,bwd_msg:jnp.ndarray,reward:jnp.ndarray,
                 h:jnp.array,c:jnp.array,) -> Tuple[jnp.ndarray, jnp.ndarray, hk.LSTMState]:
        
        carry=(h,c)
        inputs = jnp.concatenate([inc_fwd_msg,inc_bwd_msg,fwd_msg, bwd_msg,reward], axis=-1)
        carry,outputs= self._lstm(carry,inputs)
        h,c=carry
        fwd_msg = self._fwd_messenger(outputs)
        bwd_msg = self._bwd_messenger(outputs)
        # replace layer norm
        if self._use_layer_norm:
            fwd_msg = self._fwd_layer_norm(fwd_msg)
            bwd_msg = self._bwd_layer_norm(bwd_msg)
        return h,c,fwd_msg, bwd_msg



class VSMLRNN(nn.Module):

    def __init__(self, layer_specs: List[LayerSpec], num_micro_ticks: int,
                 output_idx: int, backward_pass: bool,
                 separate_backward_rnn: bool, layerwise_rnns: bool):
        super().__init__()
        self._layer_specs = layer_specs
        self._num_micro_ticks = num_micro_ticks
        
        if layerwise_rnns:
            self._sub_rnns = [SubRNN() for _ in layer_specs]
        else:
            self._sub_rnns = [SubRNN()] * len(layer_specs)
            
        self._backward_pass = backward_pass
        self._feed_label = feed_label
        if backward_pass:
            if separate_backward_rnn:
                if layerwise_rnns:
                    self._back_sub_rnns = [SubRNN() for _ in layer_specs]
                else:
                    self._back_sub_rnns = [SubRNN()] * len(layer_specs)
            else:
                self._back_sub_rnns = self._sub_rnns
        self._output_idx = output_idx



    def _tick(self, sub_rnns, layer_states: List[LayerState], reward: jnp.ndarray,
              inp: jnp.ndarray, reverse=False) -> Tuple[List[LayerState], jnp.ndarray]:

        sub_rnn = sub_rnns[0]
        fwd_msg = jnp.pad(inp[..., None], (*[(0, 0)] * inp.ndim, (0, sub_rnn.msg_size - 1)))
        bwd_msg = jnp.pad(error, ((0, 0), (0, sub_rnn.msg_size - 2)))
        layer_states[0].incoming_fwd_msg = fwd_msg
        layer_states[-1].incoming_bwd_msg = bwd_msg
        output = None

        iterable = list(enumerate(zip(layer_states, self._layer_specs, sub_rnns)))
        if reverse:
            iterable = list(reversed(iterable))
        for i, (ls, lspec, srnn) in iterable:
            lstm_state, fwd_msg, bwd_msg = (ls.lstm_state,
                                            ls.incoming_fwd_msg,
                                            ls.incoming_bwd_msg)
            for _ in range(self._num_micro_ticks):
                args = (srnn, jnp.mean, fwd_msg, bwd_msg, lstm_state)
                if isinstance(lspec, DenseSpec):
                    out = vsml_layers.dense(*args)
                elif isinstance(lspec, ConvSpec):
                    out = vsml_layers.conv2d(*args, stride=lspec.stride)
                else:
                    raise ValueError(f'Invalid layer {lspec}')
                new_fwd_msg, new_bwd_msg, lstm_state = out
            ls.lstm_state = lstm_state
            if i > 0:
                shape = layer_states[i - 1].incoming_bwd_msg.shape
                layer_states[i - 1].incoming_bwd_msg = new_bwd_msg.reshape(shape)
            if i < len(layer_states) - 1:
                shape = layer_states[i + 1].incoming_fwd_msg.shape
                layer_states[i + 1].incoming_fwd_msg = new_fwd_msg.reshape(shape)
            else:
                output = new_fwd_msg[:, self._output_idx]
                if self._tanh_bound:
                    output = jnp.tanh(output / self._tanh_bound) * self._tanh_bound

        return layer_states, output










    def _create_layer_state(self, spec: LayerSpec) -> LayerState:
        sub_rnn = self._sub_rnns[0]
        lstm_state = sub_rnn.initial_state(spec)
        msg_size = sub_rnn.msg_size
        new_msg = functools.partial(jnp.zeros, dtype=lstm_state.hidden.dtype)
        if isinstance(spec, DenseSpec):
            incoming_fwd_msg = new_msg((spec.in_size, msg_size))
            incoming_bwd_msg = new_msg((spec.out_size, msg_size))
        elif isinstance(spec, ConvSpec):
            incoming_fwd_msg = new_msg((spec.in_height, spec.in_width,
                                        spec.in_channels, msg_size))
            incoming_bwd_msg = new_msg((spec.out_height, spec.out_width,
                                        spec.out_channels, msg_size))

        return LayerState(lstm_state=lstm_state,
                          incoming_fwd_msg=incoming_fwd_msg,
                          incoming_bwd_msg=incoming_bwd_msg)

    def _merge_layer_states(self, layer_states: List[LayerState]) -> List[LayerState]:
        def merge(state):
            s1, s2 = jnp.split(state, [state.shape[-1] // 2], axis=-1)
            merged_s1 = jnp.mean(s1, axis=0, keepdims=True)
            new_s1 = jnp.broadcast_to(merged_s1, s1.shape)
            return jnp.concatenate((new_s1, s2), axis=-1)
        for ls in layer_states:
            ls.lstm_state = jax.tree_map(merge, ls.lstm_state)
        return layer_states

    def __call__(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        layer_states = [self._create_layer_state(spec) for spec in self._layer_specs]
        layer_states = jax.tree_map(lambda ls: jnp.stack([ls] * inputs.shape[1]),
                                    layer_states)
        init_error = layer_states[-1].incoming_bwd_msg[..., :2]

        def scan_tick(carry, x):
            layer_states, error = carry
            inp, label = x
            if inp.shape[0] > 1:
                layer_states = self._merge_layer_states(layer_states)

            new_layer_states, out = self._batched_tick(layer_states, error, inp)
            new_error = self._loss_func_grad(out, label)
            label_input = label if self._feed_label else jnp.zeros_like(label)
            new_error = jnp.stack([new_error, label_input], axis=-1)

            if self._backward_pass:
                new_layer_states, _ = self._reverse_batched_tick(new_layer_states, new_error, inp)
                new_error = jnp.zeros_like(new_error)

            return (new_layer_states, new_error), out
        _, outputs = hk.scan(scan_tick, (layer_states, init_error),
                             (inputs, labels))
        return outputs




