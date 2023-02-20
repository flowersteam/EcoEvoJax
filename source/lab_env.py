""" Main script for simulating lab environments.
"""

import os
import sys
import jax
from evojax.util import load_model
import yaml
import numpy as np
import pandas as pd
import random as nj_random
from jax import random
import jax.numpy as jnp
import pickle
from flax.struct import dataclass
from evojax.task.base import TaskState

sys.path.append(os.getcwd())
from source.gridworld import Gridworld

from source.utils import VideoWriter


@dataclass
class AgentStates_noparam(object):
    posx: jnp.uint16
    posy: jnp.uint16
    energy: jnp.ndarray
    time_good_level: jnp.uint16
    time_alive: jnp.uint16
    time_under_level: jnp.uint16
    alive: jnp.int8

@dataclass
class State_noparam(TaskState):
    last_actions: jnp.int8
    rewards: jnp.int8
    agents: AgentStates_noparam
    steps: jnp.int32

# ----- configure the three lab environments -----
test_configs = {"low-resources": {"grid_width": 30,
                                  "grid_length": 30,
                                  "nb_agents": 10,
                                  "hard_coded": 0,
                                  "gen_length": 500,
                                  "init_food": 10,
                                  "place_agent": True,
                                  "place_resources": True,
                                  "regrowth_scale": 0,
                                  "agent_view": 7},

                "medium-resources": {"grid_width": 30,
                                     "grid_length": 30,
                                     "nb_agents": 10,
                                     "hard_coded": 0,
                                     "gen_length": 500,
                                     "init_food": 20,
                                     "place_agent": True,
                                     "place_resources": True,
                                     "regrowth_scale": 0,
                                     "agent_view": 7},

                "high-resources": {"grid_width": 30,
                                   "grid_length": 30,
                                   "nb_agents": 10,
                                   "hard_coded": 0,
                                   "gen_length": 500,
                                   "init_food": 60,
                                   "place_agent": True,
                                   "place_resources": True,
                                   "regrowth_scale": 0,
                                   "agent_view": 7},}


# ----------------------------------------------------

def process_eval(eval_data, project_dir, current_gen, reproduce):
    """ Save evaluation data for all generations up to the current one.

    Parameters
    ----------
    eval_data: list of dataframes
        every element is the dataframe of a generation

    project_dir: str
        name of project's directory

    current_gen: int
        current generation

    reproduce: bool
        whether reproduction is on
    """
    save_dir = project_dir + "/eval/data/reproduce_" + str(reproduce)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + "/gen_" + str(current_gen) + ".pkl", "wb") as f:
        pickle.dump(eval_data, f)


def run_lab_envs(project_dir, params, reproduction_on, key, current_gen):
    """ Evaluate a trained population in the three lab environments.

    Parameters
    ----------
    project_dir: str
        name of project that contains the trained models

    params: bytearray
        the weight values of the trained population

    reproduction_on: bool
        whether reproduction is on during evaluation

    key: int
        for seeding random

    current_gen: int
        index of current generation

    """
    eval_trials = 10
    random_agents = 50
    window = 20
    test_types = ["low-resources",
                  "medium-resources",
                  "high-resources"]

    if reproduction_on:
        time_reproduce = 20
    else:
        time_reproduce = 800

    # metrics to track during evaluation
    eval_columns = ["gen", "test_type", "eval_trial", "agent_idx", "efficiency", "sustainability",
                    "resource_closeness", "resources_sustain", "agent_closeness", "agent_sustain", "noagent_sustain",
                    "energy", "lifetime_consumption"]

    # for metrics "resource_closeness", "resources_sustain", "agent_closeness", "agent_sustain",
    # "noagent_sustain" we observe at fixed windows

    nj_random.seed(1)
    eval_data = []

    for random_agent in range(random_agents):

        # sample a random agent (for technical reasons we need 10 agents to have enough slots for reproduction)
        # agent 9 is the main agent
        with open(project_dir + "/train/data/gen_" + str(current_gen) + "states.pkl","rb") as f:
            state_info = pickle.load(f)
            agent_info = state_info["states"][-1].agents.time_alive
            potential_agents = [idx for idx, el in enumerate(agent_info) if el> 300 ]

        agent_idx = nj_random.choice(potential_agents)
        agent_idxs = [(agent_idx - el)%(params.shape[0]) for el in range(10)]
        params_test = params[agent_idxs, :]

        for test_type in test_types:

            print("Test-bed: ", test_type)
            config = test_configs[test_type]

            test_dir = project_dir + "/eval/" + test_type + "/reproduce_" + str(reproduction_on)
            if not os.path.exists(test_dir + "/media"):
                os.makedirs(test_dir + "/media")

            test_dir = project_dir + "/eval/" + test_type + "/reproduce_" + str(reproduction_on)
            if not os.path.exists(test_dir + "/data"):
                os.makedirs(test_dir + "/data")

            env = Gridworld(
                SX=config["grid_length"],
                SY=config["grid_width"],
                init_food=config["init_food"],
                nb_agents=config["nb_agents"],
                reproduction_on=reproduction_on,
                regrowth_scale=config["regrowth_scale"],
                place_agent=config["place_agent"],
                place_resources=config["place_resources"],
                params=params_test,
                time_death=config["gen_length"] + 1,  # agents never die
                time_reproduce=time_reproduce)

            for trial in range(eval_trials):

                next_key, key = random.split(key)
                state = env.reset(next_key)

                video_dir = test_dir + "/media/agent_" + str(agent_idx) + "/trial_" + str(trial)
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir)
                print("check video at ", video_dir + "/gen_" + str(current_gen) + ".mp4")

                trial_dir = test_dir + "/data/agent_" + str(agent_idx) + "/trial_" + str(trial)
                if not os.path.exists(trial_dir):
                    os.makedirs(trial_dir)

                with VideoWriter(video_dir + "/gen_" + str(current_gen) + ".mp4", 5.0) as vid:

                    group_rewards = []
                    first_rewards = [None for el in range(config["nb_agents"])]
                    within_resources = 0
                    consumed_within_resources = 0
                    agent_rewards = []
                    agent_energy_levels = []
                    within_agents = 0
                    consumed_within_agents = 0
                    consumed_without_agents = 0

                    for i in range(config["gen_length"]):

                        state, reward, energy = env.step(state)

                        for idx_reward in range(len(reward.tolist())):
                            agent_rewards.append(float(reward.tolist()[idx_reward]))
                            agent_energy_levels.append(float(energy.tolist()[idx_reward]))

                        if i % window == 0:
                            # ----- detect whether the agent is close to resources ------
                            resources = state.state[:, :, 1]
                            resources_x, resources_y = np.nonzero(resources)
                            main_agent_idx = 9  # measure metrics only for the main agent
                            posx = state.agents.posx[main_agent_idx]
                            posy = state.agents.posy[main_agent_idx]
                            within_resource = False
                            already_consumed = False
                            for resource_idx in range(len(resources_x)):
                                if ((np.abs(posx - resources_x[resource_idx])) < config["agent_view"]) and (
                                        (np.abs(posy - resources_y[resource_idx])) < config["agent_view"]):
                                    within_resource = True
                                    break

                            if within_resource:
                                within_resources += 1
                            # ---------------------------------------------------------------
                            # ----- detect whether the agent is close to another agent ------
                            other_agents_x = state.agents.posx
                            other_agents_y = state.agents.posy
                            posx = state.agents.posx[main_agent_idx]
                            posy = state.agents.posy[main_agent_idx]

                            within_agent = False
                            already_consumed_agent = False
                            for other_agent_idx in range(len(other_agents_x)):
                                if state.agents.alive[other_agent_idx] and other_agent_idx != main_agent_idx:
                                    if ((np.abs(posx - other_agents_x[other_agent_idx])) < config["agent_view"]) and (
                                            (np.abs(posy - other_agents_y[other_agent_idx])) < config["agent_view"]):
                                        within_agent = True
                                        break

                            if within_agent:
                                within_agents += 1
                            # -------------------------------------------------------------
                        if reward[len(reward) - 1] == 1 and (not already_consumed) and within_resource:
                            consumed_within_resources += 1
                            already_consumed = True

                        if reward[len(reward) - 1] == 1 and (not already_consumed_agent):
                            if within_agent:
                                consumed_within_agents += 1
                                already_consumed = True
                            else:
                                consumed_without_agents += 1
                                already_consumed = True

                        group_rewards.append(float(jnp.mean(reward[config["hard_coded"]:])))

                        # compute sustainability
                        first_times = np.where(reward > 0, i, None)
                        for idx, el in enumerate(first_times):
                            if el != None and first_rewards[idx] == None:
                                first_rewards[idx] = el

                        # save video frame
                        rgb_im = state.state[:, :, :3]
                        rgb_im = jnp.clip(rgb_im, 0, 1)

                        # change color scheme to white green and black
                        rgb_im = jnp.clip(rgb_im + jnp.expand_dims(state.state[:, :, 1], axis=-1), 0, 1)
                        rgb_im = rgb_im.at[:, :, 1].set(0)
                        rgb_im = 1 - rgb_im
                        rgb_im = rgb_im - jnp.expand_dims(state.state[:, :, 0], axis=-1)

                        rgb_im = np.repeat(rgb_im, 5, axis=0)
                        rgb_im = np.repeat(rgb_im, 5, axis=1)

                        vid.add(rgb_im)

                    vid.close()

                    # compute histogram for energy levels
                    hist_energy = {}
                    unique_energies = set(agent_energy_levels)
                    for u in unique_energies:
                        if u > 0:
                            hist_energy[u] = 0
                            for timestep, energy in enumerate(agent_energy_levels):
                                if u == energy:
                                    hist_energy[u] += agent_rewards[timestep]

                    # sustainability is highest when the agent has not consumed
                    sustain = [el for el in first_rewards if el != None]
                    if not len(sustain):
                        sustain = [config["gen_length"]]

                    # save evaluation data
                    for key_energy, value in hist_energy.items():
                        eval_data.append([current_gen, test_type, trial, agent_idx, np.mean(group_rewards),
                                          np.mean(sustain), within_resources / config["gen_length"],
                                          consumed_within_resources / (within_resources + 1),
                                          within_agents / config["gen_length"],
                                          consumed_within_agents / (within_agents + 1),
                                          consumed_without_agents / (config["gen_length"] - within_agents),
                                          key_energy, value])

                    os.rename(video_dir + "/gen_" + str(current_gen) + ".mp4",
                              video_dir + "/gen_" + str(current_gen) + "_sustain_" + str(np.mean(sustain)) + ".mp4")

    eval_data = pd.DataFrame(eval_data, columns=eval_columns)

    process_eval(eval_data, project_dir, current_gen, reproduction_on)


def eval_pretrained(project_dir):
    """ Simulate the lab environments for evaluating trained models.

    Attributes
    ---------
    project_dir: str
        name of directory containing the trained models
    """
    with open(project_dir + "/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    key = jax.random.PRNGKey(np.random.randint(42))

    # choose which generations to evaluate
    gen = 950 # evaluate only the last generation

    reproduction_on_values = [True, False]

    for reproduction_on in reproduction_on_values:

        total_eval_results = []
        params, obs_param = load_model(project_dir + "/train/models", "gen_" + str(gen) + ".npz")

        # run offline evaluation
        run_lab_envs(project_dir, params, reproduction_on, key, gen)


if __name__ == "__main__":
    project_dir = sys.argv[1]

    eval_pretrained(project_dir)
