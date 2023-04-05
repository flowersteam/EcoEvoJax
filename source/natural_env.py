""" Main script for simulating natural environments.
"""

import os
import sys
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
from evojax.util import save_model, load_model
import yaml

sys.path.append(os.getcwd())
from source.gridworld import Gridworld
from source.utils import VideoWriter


def simulate(project_dir):
    """ Simulates the natural environment.

    A new environment and population is created and trained using non-episodic neuroevolution. Trained models are saved
     for later processing.

    Attributes
    ----------
    project_dir: str
        name of project's directory for saving data and models

    """
    with open(project_dir + "/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    key = jax.random.PRNGKey(config["seed"])
    next_key, key = random.split(key)



    env = Gridworld(place_agent=config["place_agent"],
                    init_food=config["init_food"],
                    SX=config["grid_length"],
                    SY=config["grid_width"],
                    nb_agents=config["nb_agents"],
                    regrowth_scale=config["regrowth_scale"],
                    niches_scale=config["niches_scale"],
                    max_age=config["max_age"],
                    time_reproduce=config["time_reproduce"],
                    time_death=config["time_death"],
                    energy_decay=config["energy_decay"],
                    spontaneous_regrow=config["spontaneous_regrow"],
                    wall_kill=config["wall_kill"],
                    )

    state = env.reset(next_key)

    keep_mean_rewards = []
    keep_max_rewards = []

    # ----- main simulation begins -----
    for gen in range(config["num_gens"]):
        accumulated_rewards = jnp.zeros(config["nb_agents"])

        if gen % config["eval_freq"] == 0:
            vid = VideoWriter(project_dir + "/train/media/gen_" + str(gen) + ".mp4", 20.0)


            print("gen ", str(gen), " population size ", str(state.agents.alive.sum()))

        for timestep in range(config["gen_length"]):


            state, reward, _ = env.step(state)
            accumulated_rewards = accumulated_rewards + reward

            if gen % config["eval_freq"] == 0:

                if state.agents.alive.sum() == 0:
                    print("all agents died")
                # every eval_freq generations we save the video of the generation
                rgb_im = state.state[:, :, :3]
                rgb_im = jnp.clip(rgb_im, 0, 1)

                # change color scheme to white green and black
                rgb_im = jnp.clip(rgb_im + jnp.expand_dims(state.state[:, :, 1], axis=-1), 0, 1)
                rgb_im = rgb_im.at[:, :, 1].set(0)
                rgb_im = 1 - rgb_im

                rgb_im = rgb_im - jnp.expand_dims(state.state[:, :, 0], axis=-1)
                rgb_im = np.repeat(rgb_im, 2, axis=0)
                rgb_im = np.repeat(rgb_im, 2, axis=1)

                vid.add(rgb_im)

        keep_mean_rewards.append(np.mean(accumulated_rewards))
        keep_max_rewards.append(np.max(accumulated_rewards))

        if gen % config["eval_freq"] * 10 == 0:
            vid.close()
            # save training data and plots
            with open(project_dir + "/train/data/gen_" + str(gen) + ".pkl", "wb") as f:
                pickle.dump({"mean_rewards": keep_mean_rewards,
                             "max_rewards": keep_max_rewards}, f)

            save_model(model_dir=project_dir + "/train/models", model_name="step_" + str(gen),
                       params=state.agents.params)

            plt.plot(range(len(keep_mean_rewards)), keep_mean_rewards, label="mean")
            plt.plot(range(len(keep_max_rewards)), keep_max_rewards, label="max")
            plt.ylabel("Training rewards")
            plt.legend()
            plt.savefig(project_dir + "/train/media/rewards_" + str(gen) + ".png")
            plt.clf()


if __name__ == "__main__":
    project_dir = sys.argv[1]
    simulate(project_dir)
