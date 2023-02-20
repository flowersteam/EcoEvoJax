# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
from PIL import Image
from PIL import ImageDraw
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask



SIZE_GRID=20
AGENT_VIEW=2
CONVOL_KER=jnp.array([[0,1,0],
                     [1,0,1],
                      [ 0,1,0 ]])




print(CONVOL_KER)


@dataclass
class AgentState(object):
    posx: jnp.int32
    posy: jnp.int32

@dataclass
class State(TaskState):
    obs: jnp.ndarray
    state: jnp.ndarray
    limits: jnp.array
    agent: AgentState
    steps: jnp.int32
    key: jnp.ndarray





def get_obs(state: jnp.ndarray,posx:jnp.int32,posy:jnp.int32) -> jnp.ndarray:
    obs=jnp.ravel(jax.lax.dynamic_slice(jnp.pad(state,((2,2),(2,2),(0,0))),(posx-AGENT_VIEW+2,posy-AGENT_VIEW+2,0),(2*AGENT_VIEW+1,2*AGENT_VIEW+1,2)))
    return obs

def get_init_state_fn(key: jnp.ndarray) -> jnp.ndarray:
    grid=jnp.zeros((SIZE_GRID,SIZE_GRID,3))
    posx,posy=(2,2)
    grid=grid.at[posx,posy,0].set(1)
    posfx,posfy=(5,5)
    grid=grid.at[posfx-1:posfx+2,posfy,1].set(1)
    grid=grid.at[0,:,2].set(1)
    grid=grid.at[:,0,2].set(1)
    grid=grid.at[:,-1,2].set(1)
    grid=grid.at[-1,:,2].set(1)
    return (grid)




class Gridworld(VectorizedTask):
    """gridworld task."""

    def __init__(self,
                 max_steps: int = 1000,
                 test: bool = False,spawn_prob=0.005):

        self.max_steps = max_steps
        self.obs_shape = tuple([(AGENT_VIEW*2+1)*(AGENT_VIEW*2+1)*2+6, ])
        self.act_shape = tuple([5, ])
        self.test = test
        self.spawn_prob=spawn_prob
        self.CONVOL_KER=CONVOL_KER*spawn_prob

        def reset_fn(key):
            next_key, key = random.split(key)
            posx,posy=(2,2)
            agent=AgentState(posx=posx,posy=posy)
            grid=get_init_state_fn(key)


            limits=jnp.zeros((SIZE_GRID,SIZE_GRID))
            limits=limits.at[2:7,2:7].set(1)
            next_key, key = random.split(key)

            rand=jax.random.uniform(key)
            limits=jnp.where(rand>0.5,limits.at[13:18,2:7].set(4),limits.at[2:7,13:18].set(4))
            grid=jnp.where(rand>0.5,grid.at[16:18,5:7,1].set(1),grid.at[5:7,16:18,1].set(1))


            return State(state=grid, obs=jnp.concatenate([get_obs(state=grid,posx=posx,posy=posy),jnp.zeros((5,)),jnp.zeros((1,))]),limits=limits,agent=agent,
                         steps=jnp.zeros((), dtype=int), key=next_key)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))


        def step_fn(state, action):
            #spawn food
            grid=state.state
            fruit=state.state[:,:,1]
            prob=jax.scipy.signal.convolve(fruit,self.CONVOL_KER,mode="same")
            prob=prob*state.limits
            key, subkey = random.split(state.key)
            spawn=jax.random.bernoulli(subkey,prob)
            next_fruit=jnp.clip(fruit+spawn,0,1)
            grid=grid.at[:,:,1].set(next_fruit)
            #move agent
            key, subkey = random.split(key)
            #maybe later make the agent to output the one hot categorical
            action=jax.random.categorical(subkey,action)
            action=jax.nn.one_hot(action,5)

            action=action.astype(jnp.int32)

            posx=state.agent.posx-action[1]+action[3]
            posy=state.agent.posy-action[2]+action[4]
            posx=jnp.clip(posx,0,SIZE_GRID-1)
            posy=jnp.clip(posy,0,SIZE_GRID-1)
            grid=grid.at[state.agent.posx,state.agent.posy,0].set(0)
            grid=grid.at[posx,posy,0].set(1)

            reward=jnp.sum(grid[:,:,0]*grid[:,:,1])

            grid=grid.at[:,:,1].set(jnp.clip(grid[:,:,1]-grid[:,:,0],0,1))



            steps = state.steps + 1
            done = False
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)
            key, sub_key = random.split(key)

            #keep it in case we let agent several trials
            grid = jax.lax.cond(
                done, lambda x: get_init_state_fn(sub_key), lambda x: x, grid)

            return State(state=grid, obs=jnp.concatenate([get_obs(state=grid,posx=posx,posy=posy),action,jnp.ones((1,))*reward]),limits=state.limits,agent=AgentState(posx=posx,posy=posy),
                         steps=steps, key=key), reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)


