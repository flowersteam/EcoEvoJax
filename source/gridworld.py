""" This script contains the implementation of the environment.
"""
from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass
import math
from source.agent import MetaRnnPolicy_bcppr
from source.agent import metaRNNPolicyState_bcppr
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask


AGENT_VIEW = 7



@dataclass
class AgentStates(object):
    posx: jnp.uint16
    posy: jnp.uint16
    params: jnp.ndarray
    policy_states: PolicyState
    energy: jnp.ndarray
    time_good_level: jnp.uint16
    time_alive: jnp.uint16
    time_under_level: jnp.uint16
    alive: jnp.int8


@dataclass
class State(TaskState):
    obs: jnp.int8
    last_actions: jnp.int8
    rewards: jnp.int8
    state: jnp.int8
    agents: AgentStates
    steps: jnp.int32
    key: jnp.ndarray


def get_ob(state: jnp.ndarray, pos_x: jnp.int32, pos_y: jnp.int32) -> jnp.ndarray:
    """ Returns an agent's observation.
    """
    obs = (jax.lax.dynamic_slice(jnp.pad(state, ((AGENT_VIEW, AGENT_VIEW), (AGENT_VIEW, AGENT_VIEW), (0, 0))),
                                 (pos_x - AGENT_VIEW + AGENT_VIEW, pos_y - AGENT_VIEW + AGENT_VIEW, 0),
                                 (2 * AGENT_VIEW + 1, 2 * AGENT_VIEW + 1, 3)))
    return obs


def get_init_state_fn(key: jnp.ndarray, SX, SY, posx, posy, pos_food_x, pos_food_y, niches_scale=200) -> jnp.ndarray:
    """ Returns the initial state of the grid.

    The grid has four dimensions: 0 corresponds to agents, 1 to resources, 2 to walls and 3 to the climate function
    """
    grid = jnp.zeros((SX, SY, 4))
    grid = grid.at[posx, posy, 0].add(1) # position agents
    grid = grid.at[posx[:5], posy[:5], 0].set(0) # due a technicality the first 5 positions are not used
    grid = grid.at[pos_food_x, pos_food_y, 1].set(1) # position resources

    # ----- determine climate function based on the niching model -----
    new_array = jnp.clip(
        np.asarray([(math.pow(niches_scale, el) - 1) / (niches_scale - 1) for el in np.arange(0, SX) / SX]), 0,
        1)

    for col in range(SY - 1):
        new_col = jnp.clip(
            np.asarray([(math.pow(niches_scale, el) - 1) / (niches_scale - 1) for el in np.arange(0, SX) / SX]), 0, 1)

        new_array = jnp.append(new_array, new_col)
    new_array = jnp.transpose(jnp.reshape(new_array, (SY, SX)))
    grid = grid.at[:, :, 3].set(new_array)
    # -------------------------------------
    # place the walls
    grid = grid.at[0, :, 2].set(1)
    grid = grid.at[-1, :, 2].set(1)
    grid = grid.at[:, 0, 2].set(1)
    grid = grid.at[:, -1, 2].set(1)
    return (grid)


get_obs_vector = jax.vmap(get_ob, in_axes=(None, 0, 0), out_axes=0)


class Gridworld(VectorizedTask):
    """ gridworld task."""

    def __init__(self,
                 nb_agents: int = 100,
                 SX=300,
                 SY=100,
                 init_food=0,
                 place_agent=False,
                 place_resources=False,
                 reproduction_on=True,
                 params=None,
                 test: bool = False,
                 energy_decay=0.05,
                 max_age: int = 1000,
                 time_reproduce: int = 150,
                 time_death: int = 40,
                 max_ener=3.,
                 regrowth_scale=0.002,
                 niches_scale=200,
                 spontaneous_regrow=1 / 200000,

                 ):
        self.obs_shape = (AGENT_VIEW, AGENT_VIEW, 3)
        self.act_shape = tuple([5, ])
        self.test = test
        self.nb_agents = nb_agents
        self.SX = SX
        self.SY = SY
        self.energy_decay = energy_decay
        self.model = MetaRnnPolicy_bcppr(input_dim=((AGENT_VIEW * 2 + 1), (AGENT_VIEW * 2 + 1), 3), hidden_dim=4,
                                         output_dim=5, encoder_layers=[], hidden_layers=[8])

        self.energy_decay = energy_decay
        self.max_age = max_age
        self.time_reproduce = time_reproduce
        self.time_death = time_death
        self.max_ener = max_ener

        self.regrowth_scale = regrowth_scale
        self.niches_scale = niches_scale
        self.spontaneous_regrow = spontaneous_regrow
        self.place_agent = place_agent
        self.place_resources = place_resources

        def reset_fn(key):
            if self.place_agent:
                next_key, key = random.split(key)
                posx = random.randint(next_key, (nb_agents,), int(2 / 5 * SX), int(3 / 5 * SX))
                next_key, key = random.split(key)
                posy = random.randint(next_key, (nb_agents,), int(2 / 5 * SX), int(3 / 5 * SX))
                next_key, key = random.split(key)
            else:
                next_key, key = random.split(key)
                posx = random.randint(next_key, (nb_agents,), 1, (SX - 1))
                next_key, key = random.split(key)
                posy = random.randint(next_key, (nb_agents,), 1, (SY - 1))
                next_key, key = random.split(key)

            if self.place_resources:
                # lab environments have a custom location of resources
                N = 5 # minimum distance from agents
                N_wall = 5 # minimum distance from wall

                pos_food_x = jnp.concatenate(
                    (random.randint(next_key, (int(init_food / 4),), int(1 / 2 * SX) + N, (SX - 1 - N_wall)),
                     random.randint(next_key, (int(init_food / 4),), N_wall, int(1 / 2 * SX) - N),
                     random.randint(next_key, (int(init_food / 4),), 1 + N_wall, (SX - 1 - N_wall)),
                     random.randint(next_key, (int(init_food / 4),), 1 + N_wall, (SX - 1 - N_wall))))

                next_key, key = random.split(key)
                pos_food_y = jnp.concatenate(
                    (random.randint(next_key, (int(init_food / 4),), 1 + N_wall, SY - 1 - N_wall),
                     random.randint(next_key, (int(init_food / 4),), 1 + N_wall, SY - 1 - N_wall),
                     random.randint(next_key, (int(init_food / 4),), int(1 / 2 * SY) + N,
                                    (SY - 1 - N_wall)),
                     random.randint(next_key, (int(init_food / 4),), N_wall, int(1 / 2 * SY) - N)))
                next_key, key = random.split(key)

            else:
                # in natural environments resources are placed randomly
                pos_food_x = random.randint(next_key, (init_food,), 1, (SX - 1))
                next_key, key = random.split(key)
                pos_food_y = random.randint(next_key, (init_food,), 1, (SY - 1))
                next_key, key = random.split(key)

            grid = get_init_state_fn(key, SX, SY, posx, posy, pos_food_x, pos_food_y, niches_scale)

            next_key, key = random.split(key)

            policy_states = self.model.reset_b(jnp.zeros(self.nb_agents, ))

            agents = AgentStates(posx=posx, posy=posy,
                                 energy=self.max_ener * jnp.ones((self.nb_agents,)).at[0:5].set(0),
                                 time_good_level=jnp.zeros((self.nb_agents,), dtype=jnp.uint16), params=params,
                                 policy_states=policy_states,
                                 time_alive=jnp.zeros((self.nb_agents,), dtype=jnp.uint16),
                                 time_under_level=jnp.zeros((self.nb_agents,), dtype=jnp.uint16),
                                 alive=jnp.ones((self.nb_agents,), dtype=jnp.uint16).at[0:9].set(0))

            return State(state=grid, obs=get_obs_vector(grid, posx, posy), last_actions=jnp.zeros((self.nb_agents, 5)),
                         rewards=jnp.zeros((self.nb_agents, 1)), agents=agents,
                         steps=jnp.zeros((), dtype=int), key=next_key)

        self._reset_fn = jax.jit(reset_fn)

        def reproduce(params, posx, posy, energy, time_good_level, key, policy_states, time_alive, alive):
            """ Implements local reproduction based on a minimal criterion on the energy level of each agent.
            """
            # use agent 0 to 4 as a dump always dead if no dead put in there to be sure not overiding the alive ones
            # but maybe better to just make sure that there are 5 places available by checking if 5 dead (but this way may be better if we augment the 5)
            dead = 1 - alive
            dead = dead.at[0:5].set(0.001)

            next_key, key = random.split(key)
            # empty_spots for new agent are dead ones
            empty_spots = jax.random.choice(next_key, jnp.arange(time_good_level.shape[0]), p=dead, replace=False,
                                            shape=(5,))

            # compute reproducer spot
            next_key, key = random.split(key)
            reproducer = jnp.where(time_good_level > self.time_reproduce, 1, 0)
            reproducer = reproducer.at[0:5].set(0.001)
            reproducer_spots = jax.random.choice(next_key, jnp.arange(time_good_level.shape[0]),
                                                 p=reproducer / (reproducer.sum() + 1e-10), replace=False, shape=(5,))

            next_key, key = random.split(key)
            params = params
            # new agents params with mutate , and also take pos of parents
            if reproduction_on:
                params = params.at[empty_spots].set(
                    params[reproducer_spots] + 0.02 * jax.random.normal(next_key, (5, params.shape[1])))
                posx = posx.at[empty_spots].set(posx[reproducer_spots])
                posy = posy.at[empty_spots].set(posy[reproducer_spots])

            # multiply by reproducer to be sure that the one that got selected by reproducer spot were reproducer indeed,
            # in case nb reproducer <5 but again maybe we can just check that at least 5 reproducer but weird
            energy = energy.at[empty_spots].set(self.max_ener * reproducer[reproducer_spots])
            energy = energy.at[0:5].set(0.)

            # new agents alive and time alive , time_good_alive, and RNN state set at 0
            alive = alive.at[empty_spots].set(1 * reproducer[reproducer_spots])
            time_alive = time_alive.at[empty_spots].set(0)
            time_good_level = time_good_level.at[empty_spots].set(0)
            policy_states = metaRNNPolicyState_bcppr(
                lstm_h=policy_states.lstm_h.at[empty_spots].set(jnp.zeros(policy_states.lstm_h.shape[1])),
                lstm_c=policy_states.lstm_c.at[empty_spots].set(jnp.zeros(policy_states.lstm_c.shape[1])),
                keys=policy_states.keys)

            # put time good level of reproducer back to 0
            # if in the dump don't put to 0 so that they can try reproduce in the next timestep
            time_good_level = time_good_level.at[reproducer_spots].set(
                time_good_level[reproducer_spots] * (empty_spots < 5))

            # kill the dump
            alive = alive.at[0:5].set(0)

            return (params, posx, posy, energy, time_good_level, policy_states, time_alive, alive)

        def step_fn(state):
            key = state.key
            next_key, key = random.split(key)

            # model selection of action
            actions_logit, policy_states = self.model.get_actions(state, state.agents.params,
                                                                  state.agents.policy_states)
            actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit * 50, axis=-1), 5)
            grid = state.state
            energy = state.agents.energy
            alive = state.agents.alive

            # move agent
            action_int = actions.astype(jnp.int32)
            posx = state.agents.posx - action_int[:, 1] + action_int[:, 3]
            posy = state.agents.posy - action_int[:, 2] + action_int[:, 4]

            # wall
            hit_wall = state.state[posx, posy, 2] > 0
            posx = jnp.where(hit_wall, state.agents.posx, posx)
            posy = jnp.where(hit_wall, state.agents.posy, posy)
            posx = jnp.clip(posx, 0, SX - 1)
            posy = jnp.clip(posy, 0, SY - 1)
            grid = grid.at[state.agents.posx, state.agents.posy, 0].set(0)
            # add only the alive
            grid = grid.at[posx, posy, 0].add(1 * (alive > 0))

            ### collect food
            rewards = (alive > 0) * (grid[posx, posy, 1] > 0) * (1 / (grid[posx, posy, 0] + 1e-10))
            grid = grid.at[posx, posy, 1].add(-1 * (alive > 0))
            grid = grid.at[:, :, 1].set(jnp.clip(grid[:, :, 1], 0, 1))

            # regrow resources
            num_neighbs = jax.scipy.signal.convolve2d(grid[:, :, 1], jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
                                                      mode="same")
            scale = grid[:, :, 3]
            scale_constant = regrowth_scale
            next_key, key = random.split(state.key)

            if scale_constant != 0:
                num_neighbs = jnp.where(num_neighbs != 1, 0, num_neighbs)
                num_neighbs = jnp.where(num_neighbs == 1, 0.002, num_neighbs)
                num_neighbs = jnp.multiply(num_neighbs, scale)
                num_neighbs = jnp.where(num_neighbs > 0, num_neighbs, 0)
                num_neighbs = num_neighbs + self.spontaneous_regrow
                num_neighbs = jnp.clip(num_neighbs - grid[:, :, 2], 0, 1)
                grid = grid.at[:, :, 1].add(random.bernoulli(next_key, num_neighbs))

            ####
            steps = state.steps + 1

            # decay of energy and clipping
            energy = energy - self.energy_decay + rewards
            energy = jnp.clip(energy, -1000, self.max_ener)
            time_good_level = jnp.where(energy > 0, (state.agents.time_good_level + 1) * alive, 0)
            time_alive = state.agents.time_alive

            # look if still alive
            time_alive = jnp.where(alive > 0, time_alive + 1, 0)

            # compute reproducer and go through the function only if there is one
            reproducer = jnp.where(state.agents.time_good_level > self.time_reproduce, 1, 0)
            next_key, key = random.split(key)
            params, posx, posy, energy, time_good_level, policy_states, time_alive, alive = jax.lax.cond(
                reproducer.sum() > 0, reproduce, lambda y, z, a, b, c, d, e, f, g: (y, z, a, b, c, e, f, g), *(
                state.agents.params, posx, posy, energy, time_good_level, next_key, state.agents.policy_states,
                time_alive, alive))

            time_under_level = jnp.where(energy < 0, state.agents.time_under_level + 1, 0)
            alive = jnp.where(jnp.logical_or(time_alive > self.max_age, time_under_level > self.time_death), 0, alive)

            done = False
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)
            cur_state = State(state=grid, obs=get_obs_vector(grid, posx, posy), last_actions=actions,
                              rewards=jnp.expand_dims(rewards, -1),
                              agents=AgentStates(posx=posx, posy=posy, energy=energy, time_good_level=time_good_level,
                                                 params=params, policy_states=policy_states,
                                                 time_alive=time_alive, time_under_level=time_under_level, alive=alive),
                              steps=steps, key=key)
            # keep it in case we let agent several trials
            state = jax.lax.cond(
                done, lambda x: reset_fn(state.key), lambda x: x, cur_state)

            return state,rewards, energy

        self._step_fn = jax.jit(step_fn)

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             ) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state)



