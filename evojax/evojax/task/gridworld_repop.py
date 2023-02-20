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



SIZE_GRID=6
AGENT_VIEW=1
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
    agent: AgentState
    steps: jnp.int32
    side:jnp.int32
    repop0:jnp.int32
    repop1:jnp.int32
    key: jnp.ndarray






def get_obs(state: jnp.ndarray,posx:jnp.int32,posy:jnp.int32) -> jnp.ndarray:
    obs=jnp.ravel(jax.lax.dynamic_slice(jnp.pad(state,((AGENT_VIEW,AGENT_VIEW),(AGENT_VIEW,AGENT_VIEW),(0,0))),(posx-AGENT_VIEW+AGENT_VIEW,posy-AGENT_VIEW+AGENT_VIEW,0),(2*AGENT_VIEW+1,2*AGENT_VIEW+1,3)))
    return obs

def get_init_state_fn(key: jnp.ndarray) -> jnp.ndarray:
    grid=jnp.zeros((SIZE_GRID,SIZE_GRID,3))
    posx,posy=(0,0)
    grid=grid.at[posx,posy,0].set(1)

    grid=grid.at[1,1,1].set(1)
    grid=grid.at[1:4,1,2].set(1)
    grid=grid.at[1,1:4,2].set(1)

    return (grid)




class Gridworld(VectorizedTask):
    """gridworld task."""

    def __init__(self,
                 max_steps: int = 200,
                 test: bool = False,spawn_prob=0.005):

        self.max_steps = max_steps
        self.obs_shape = tuple([(AGENT_VIEW*2+1)*(AGENT_VIEW*2+1)*3+6, ])
        self.act_shape = tuple([5, ])
        self.test = test


        def reset_fn(key):
            next_key, key = random.split(key)
            posx,posy=(0,0)
            agent=AgentState(posx=posx,posy=posy)
            grid=get_init_state_fn(key)

            next_key, key = random.split(key)

            rand=jax.random.uniform(key)
            grid=jnp.where(rand>0.5,grid.at[1,4,1].set(1),grid.at[4,1,1].set(1))
            side=jnp.where(rand>0.5,jnp.zeros((), dtype=int),jnp.ones((), dtype=int))
            repop0=jnp.zeros((), dtype=int)
            repop1=jnp.zeros((), dtype=int)


            return State(state=grid, obs=jnp.concatenate([get_obs(state=grid,posx=posx,posy=posy),jnp.zeros((5,)),jnp.zeros((1,))]),agent=agent,
                         steps=jnp.zeros((), dtype=int),side=side,repop0=repop0,repop1=repop1, key=next_key)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))


        def step_fn(state, action):
            side=state.side
            #spawn food
            grid=state.state
            #repop timer
            repop0=state.repop0+(1-grid[1,1,1]).astype(jnp.int32)
            repop1=state.repop1+(1-grid[1,4,1]).astype(jnp.int32)*(1-side)+(1-grid[4,1,1]).astype(jnp.int32)*(side)

            grid=jnp.where(state.repop0>10,grid.at[1,1,1].set(1),grid)
            repop0=jnp.where(state.repop0>10,0,repop0)

            grid=jnp.where(state.repop1>4,grid.at[1,4,1].set(1)*(1-side)+grid.at[4,1,1].set(1)*(side),grid)
            repop1=jnp.where(state.repop1>4,0,repop1)

            #move agent
            key, subkey = random.split(state.key)
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

            return State(state=grid, obs=jnp.concatenate([get_obs(state=grid,posx=posx,posy=posy),action,jnp.ones((1,))*reward]),agent=AgentState(posx=posx,posy=posy),
                         steps=steps,side=state.side,repop0=repop0,repop1=repop1, key=key), reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)


