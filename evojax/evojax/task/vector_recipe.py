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

SIZE_GRID = 3
AGENT_VIEW = 1


@dataclass
class AgentState(object):
    inventory: jnp.int32


@dataclass
class State(TaskState):
    obs: jnp.int32
    last_action: jnp.ndarray
    reward: jnp.ndarray
    state: jnp.ndarray
    agent: AgentState
    steps: jnp.int32
    permutation_recipe: jnp.ndarray
    key: jnp.ndarray


def get_obs(state: jnp.ndarray) -> jnp.ndarray:
    return state


def get_init_state_fn(key: jnp.ndarray,nb_items) -> jnp.ndarray:
    grid = jnp.zeros((nb_items+2,))

    grid = grid.at[:nb_items].set(1)

    return (grid)


def test_recipes(action, inventory, recipes,nb_items):
    recipe_done = jnp.where(jnp.logical_or(jnp.logical_and(recipes[0] == inventory, recipes[1] == action),
                                           jnp.logical_and(recipes[0] == action, recipes[1] == inventory)),
                            jnp.array([recipes[0], recipes[1], nb_items]),
                            jnp.zeros(3, jnp.int32))
    recipe_done = jnp.where(jnp.logical_or(jnp.logical_and(recipes[2] == inventory, action == nb_items),
                                           jnp.logical_and(inventory == nb_items, recipes[2] == action)),
                            jnp.array([recipes[2], nb_items, nb_items+1]), recipe_done)
    product = recipe_done[2]
    reward = jnp.select([product == 0, product == nb_items, product == nb_items+1], [0., 1., 2.])
    return recipe_done, reward


def try_recipe(grid, action, inventory, permutation_recipes):
    in_env = grid[action] > 0
    nb_items=grid.shape[0]-2
    recipe_done, reward = jax.lax.cond(in_env, test_recipes,
                                       lambda x, y, z,a: (jnp.zeros(3, jnp.int32), 0.),
                                       *(action, inventory, permutation_recipes,nb_items))

    # tested so put back
    grid = jnp.where(in_env, grid.at[inventory].set(1), grid)
    grid = jnp.where(recipe_done[2] > 0, grid.at[action].set(0).at[inventory].set(0).at[recipe_done[2]].set(1), grid)
    inventory = jnp.where(in_env, -1, inventory)

    return grid, inventory, reward


def collect(grid, action, inventory, permutation_recipe):
    in_env = (grid[action] > 0)
    grid = jnp.where(in_env, grid.at[action].set(0), grid)
    inventory = jnp.where(in_env, action, -1)

    return grid, inventory, 0.


class Gridworld(VectorizedTask):
    """gridworld task."""

    def __init__(self,
                 max_steps: int = 200,
                 test: bool = False,
                 nb_items=5):
        self.max_steps = max_steps
        self.obs_shape = tuple([(nb_items+2) *2, ])
        self.act_shape = tuple([nb_items+2, ])
        self.test = test
        self.nb_items=nb_items

        def reset_fn(key):
            next_key, key = random.split(key)
            agent = AgentState(inventory=-1)
            grid = get_init_state_fn(key,self.nb_items)

            next_key, key = random.split(next_key)
            permutation_recipe = jax.random.permutation(key, self.nb_items)[:3]
            # rand=jax.random.uniform(key)
            # permutation_recipe=jnp.where(rand>0.5,jnp.array([1,2,3]),jnp.array([1,3,2]))
            # permutation_recipe=jnp.where(rand<0.5,jnp.array([2,3,1]),permutation_recipe)
            return State(state=grid, obs=jnp.concatenate([get_obs(state=grid), jnp.zeros(self.nb_items+2)]),
                         last_action=jnp.zeros((self.nb_items+2,)), reward=jnp.zeros((1,)), agent=agent,
                         steps=jnp.zeros((), dtype=int), permutation_recipe=permutation_recipe, key=next_key)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def rest_keep_recipe(key, recipes, steps,nb_items):
            next_key, key = random.split(key)

            agent = AgentState(inventory=-1)
            grid = get_init_state_fn(key,nb_items)

            return State(state=grid, obs=jnp.concatenate([get_obs(state=grid), jnp.zeros(nb_items+2)]),
                         last_action=jnp.zeros((nb_items+2,)), reward=jnp.zeros((1,)), agent=agent,
                         steps=steps, permutation_recipe=recipes, key=next_key)

        def step_fn(state, action):
            # spawn food
            grid = state.state
            reward = 0

            # move agent
            key, subkey = random.split(state.key)
            # maybe later make the agent to output the one hot categorical
            action = jax.random.categorical(subkey, action)

            # collect or drop
            inventory = state.agent.inventory
            key, subkey = random.split(key)

            # no item in inventory, try to pick else collect
            grid, inventory, reward = jax.lax.cond(inventory < 0, collect, try_recipe,
                                                   *(grid, action, inventory, state.permutation_recipe))

            steps = state.steps + 1
            done = jnp.logical_or(grid[-1] > 0, steps > self.max_steps)

            # key, subkey = random.split(key)
            # rand=jax.random.uniform(subkey)
            # catastrophic=jnp.logical_and(steps>40,rand<1)
            # done=jnp.logical_or(done, catastrophic)
            # a=state.permutation_recipe[1]
            # b=state.permutation_recipe[2]
            # permutation_recipe=jnp.where(catastrophic,state.permutation_recipe.at[1].set(b).at[2].set(a), state.permutation_recipe)
            # steps = jnp.where(catastrophic, jnp.zeros((), jnp.int32), steps)

            action = jax.nn.one_hot(action, self.nb_items+2)

            cur_state = State(state=grid, obs=jnp.concatenate(
                [get_obs(state=grid), jax.nn.one_hot(inventory, self.nb_items+2)]), last_action=action,
                              reward=jnp.ones((1,)) * reward,
                              agent=AgentState(inventory=inventory),
                              steps=steps, permutation_recipe=state.permutation_recipe, key=key)

            # keep it in case we let agent several trials
            state = jax.lax.cond(
                done, lambda x: rest_keep_recipe(key, state.permutation_recipe, steps,self.nb_items), lambda x: x, cur_state)
            done = False
            return state, reward, done

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)




