import sys
import os
import datetime
import copy
import yaml
from evojax.task.base import TaskState
from flax.struct import dataclass
import jax.numpy as jnp
sys.path.append(os.getcwd())
from source.natural_env import simulate
from source.lab_env import eval_pretrained

# these classes are used in the lab environment
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


def setup_project(config, exp_name):
    """ Prepare the directory of the current project, saved under 'projects'.

    Creates subdirectories and saves the project's configuration.

    Attributes
    ------
    config: dict
        contains configuration for current project

    exp_name: str
        a custom name for the directory

    """
    now = datetime.datetime.now()
    today = str(now.day) + "_" + str(now.month) + "_" + str(now.year)

    top_dir = "projects/"
    project_dir = top_dir + today + "/" + exp_name + "/"

    # name the project based on its configuration
    for key, value in config.items():
        if key != "trial" and key != "load_trained":
            project_dir += key + "_" + str(value)

    project_dir += "/trial_" + str(config["trial"])

    if not os.path.exists(project_dir + "/train/data"):
        os.makedirs(project_dir + "/train/data")

    if not os.path.exists(project_dir + "/train/models"):
        os.makedirs(project_dir + "/train/models")

    if not os.path.exists(project_dir + "/train/media"):
        os.makedirs(project_dir + "/train/media")

    if not os.path.exists(project_dir + "/eval/data"):
        os.makedirs(project_dir + "/eval/data")

    if not os.path.exists(project_dir + "/eval/media"):
        os.makedirs(project_dir + "/eval/media")

    print("Saving current simulation under ", project_dir)

    with open(project_dir + "/config.yaml", "w") as f:
        yaml.dump(config, f)

    return project_dir



def train_paper():
    """ Run this to reproduce the natural environment described in the paper.
    """
    config["trial"] = 0
    config["agent_view"] = 7
    config["eval_freq"] = 20
    config["nb_agents"] = 1000
    config["grid_width"] = 200
    config["grid_length"] = config["grid_width"] * 2
    config["init_food"] = 16000

    project_dir = setup_project(config, "train_paper")
    simulate(project_dir)


def process_paper(project_dir):
    """ Run this to evaluate the trained models used in the paper in the lab environments.
    """
    eval_pretrained("projects/" + project_dir)


if __name__ == "__main__":

    mode = str(sys.argv[1])

    # generic config that you need to change for your simulation
    config = {"nb_agents": 0,
              "num_timesteps": 1000,
              "eval_freq": 50,
              "grid_width": 0,
              "init_food": 0,
              "agent_view": 7,
              "regrowth_scale": 0,
              "niches_scale": 2}

    if mode == "natural":
        train_paper() # retrain the models used in the paper

    elif mode == "lab":
        #process_paper("pretrained/seed0") # evaluate existing trained models for seed 0
        process_paper("pretrained/seed3") # evaluate existing trained models for seed 3
