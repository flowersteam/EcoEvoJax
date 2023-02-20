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
from source.agent import MetaRnnPolicy_bcppr
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

    # initialize policy
    model = MetaRnnPolicy_bcppr(input_dim=((config["agent_view"] * 2 + 1), (config["agent_view"] * 2 + 1), 3),
                                hidden_dim=4,
                                output_dim=5,
                                encoder_layers=[],
                                hidden_layers=[8])
    key = jax.random.PRNGKey(np.random.randint(42))
    next_key, key = random.split(key)
    params = jax.random.normal(
        next_key,
        (config["nb_agents"], model.num_params,),
    ) / 100

    # initialize environment
    env = Gridworld(SX=config["grid_length"],
                    SY=config["grid_width"],
                    init_food=config["init_food"],
                    nb_agents=config["nb_agents"],
                    params= params,
                    regrowth_scale=config["regrowth_scale"],
                    niches_scale=config["niches_scale"])

    state = env.reset(next_key)

    keep_mean_rewards = []
    keep_max_rewards = []

    timesteps = list(range(config["num_timesteps"]))

    # ----- main simulation begins -----
    for timestep in timesteps:


        accumulated_rewards = jnp.zeros(config["nb_agents"])

        if timestep % config["eval_freq"] == 0:
            vid = VideoWriter(project_dir + "/train/media/gen_" + str(timestep) + ".mp4", 20.0)

        for i in range(config["gen_length"]):
            next_key, key = random.split(key)

            state, reward, _ = env.step(state)
            accumulated_rewards = accumulated_rewards + reward

            if timestep % config["eval_freq"] == 0:
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

        if timestep % config["eval_freq"] == 0:

            # save training data and plots
            vid.close()
            with open(project_dir + "/train/data/gen_" + str(timestep) + ".pkl", "wb") as f:
                pickle.dump({"mean_rewards": keep_mean_rewards,
                             "max_rewards": keep_max_rewards}, f)

            save_model(model_dir=project_dir + "/train/models", model_name="gen_" + str(gen), params=params)

            plt.plot(range(len(keep_mean_rewards)), keep_mean_rewards, label="mean")
            plt.plot(range(len(keep_max_rewards)), keep_max_rewards, label="max")
            plt.ylabel("Training rewards")
            plt.legend()
            plt.savefig(project_dir + "/train/media/rewards_" + str(timestep) + ".png")
            plt.clf()


if __name__ == "__main__":
    project_dir = sys.argv[1]
    simulate(project_dir)
