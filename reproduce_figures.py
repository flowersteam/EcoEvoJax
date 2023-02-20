""" This script can be used to reproduce the paper's figures and run statistical significance tests.
"""
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway, tukey_hsd

# ----- global figure config -----
params = {'legend.fontsize': 7,
          "figure.autolayout": True,
          'font.size': 7}
plt.rcParams.update(params)
column_width_pt = 241.14749 # figure fits a single latex columm
column_width_inches = column_width_pt * 0.01384
scale = 1
golden_ratio = 0.707
fig_size = (column_width_inches / scale, column_width_inches * golden_ratio / scale)

metric_labels = {"resources_sustain": "Greediness, $G$",
                 "efficiency": "Efficiency, $E$"}

env_labels = {"low-resources": "Low \n  resources",
              "medium-resources": "Medium \n resources",
              "high-resources": "High \n resources"}
# ---------------------------------


def fig9(project):
    """ Produces figure 9 of the paper.

    Attributes
    ----------
    project: str
        project directory to load data from

    """
    metric = "efficiency"
    gen = 950
    settings = [True, False]
    tests = ["low-resources", "medium-resources", "high-resources"]

    fig, axes = plt.subplots(3, 1, figsize=(fig_size[0], fig_size[1] * 1.5), sharex=True)
    save_dir = project + "/eval/media/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    total_results = {}
    for setting in settings:
        with open(project + "/eval/data/reproduce_" + str(setting) + "/gen_" + str(gen) + ".pkl",
                  "rb") as f:
            results = pickle.load(f)
            agents = list(set(results["agent_idx"]))
            trials = range(9)
            results = results.loc[results["gen"] == gen]
            total_results[setting] = results

    for plot_idx, test in enumerate(tests):

        bar_heights = []
        bar_errors = []
        colors = ["red", "blue", "green"]

        for_significance = []

        for setting in settings:
            results_setting = total_results[setting]
            setting_values = []
            for agent_idx in agents:
                results_agent = results_setting.loc[results_setting["agent_idx"] == agent_idx]
                results_test = results_agent.loc[results_agent["test_type"] == test]
                for trial in trials:
                    results_trial = results_test.loc[results_test["eval_trial"] == trial]
                    # print(results_trial.columns)
                    metric_values = results_trial[metric].tolist()
                    metric_values = list(set(metric_values))
                    setting_values.append(metric_values[-1])
            bar_heights.append(np.mean(setting_values))
            bar_errors.append(np.std(setting_values))
            for_significance.append(setting_values)

        with open(save_dir + "/stat_sign.txt", "a") as f:

            F, p = f_oneway(*for_significance) # run anova
            if p < 0.05:
                f.write("significance for test" + test) # if anova indicates significance run Tukey's range test
                res = tukey_hsd(*for_significance)
                anal = str(res)
                f.write(anal)

        axes[plot_idx].bar(settings, bar_heights, color=colors[plot_idx])
        axes[plot_idx].set_title(env_labels[test])
        axes[plot_idx].set_ylabel(metric_labels[metric])
        axes[plot_idx].errorbar(settings, bar_heights, bar_errors, color=colors[plot_idx])
        axes[plot_idx].set_ylim([0, 0.005])

    print("Saving figure in ", save_dir + '/' + metric + '.pdf')
    print("Saving statistical significance testing results in ", save_dir + "/stat_sign.txt")
    fig.savefig(save_dir + '/' + metric + '.pdf', dpi=800)
    plt.clf()


def fig8(project):
    """ Produces figure 9 of the paper.

    Attributes
    ----------
    project: str
        project directory to load data from

    """
    gen = 950
    settings = [True, False]
    colors = ["red", "blue", "green"]
    tests = ["low-resources", "medium-resources", "high-resources"]
    metrics = [ "efficiency", "resources_sustain"]

    # load results
    total_results = {}
    for setting in settings:
        with open(project + "/eval/data/reproduce_" + str(setting) + "/gen_" + str(gen) + ".pkl",
                  "rb") as f:
            total_results[setting] = pickle.load(f)

    for metric in metrics:

        agents = list(set(total_results[settings[0]]["agent_idx"]))

        for agent_idx in agents:
            fig, axes = plt.subplots(1, len(settings), figsize=(fig_size[0] * len(settings), fig_size[1]), sharey=True)
            if len(settings) == 1:
                axes = [axes]

            for plot_idx, setting in enumerate(settings):
                results = total_results[setting]

                # plot histogram for each agent
                agents = list(set(results["agent_idx"]))
                results_gen = results.loc[results["gen"] == gen]
                signif_result = 0
                results_agent = results_gen.loc[results_gen["agent_idx"] == agent_idx]

                bar_heights = []
                bar_errors = []

                for_significance = []
                for task_idx, test in enumerate(tests):
                    test_agent = results_agent.loc[results_agent["test_type"] == test]

                    trials = list(set(test_agent["eval_trial"].tolist()))
                    sustain = []
                    for trial in trials:
                        trial_agent = test_agent.loc[test_agent["eval_trial"] == trial]

                        trial_sustain = trial_agent[metric].tolist()
                        trial_sustain = list(set(trial_sustain))
                        sustain.append(trial_sustain[-1])

                    bar_heights.append(np.mean(sustain))
                    bar_errors.append(np.std(sustain))
                    for_significance.append(sustain)

                save_dir = project + "/eval/media/reproduce_" + str(setting) + "/" + metric
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(save_dir + "/stat_sign.txt", "a") as f:

                    F, p = f_oneway(*for_significance)
                    if p < 0.05:
                        f.write("significance for agent " + str(agent_idx))
                        res = tukey_hsd(*for_significance)
                        anal = str(res)
                        f.write(anal)
                        signif_result += 1

                        if metric == "resources_sustain":

                            if bar_heights[0] > bar_heights[2]:
                                f.write(str(agent_idx) + " is sustainable in high resources")

                            if bar_heights[0] < bar_heights[2]:
                                f.write(str(agent_idx) + " is sustainable in low resources")

                        if metric == "efficiency":
                            if bar_heights[0] < bar_heights[2]:
                                f.write(str(agent_idx) + " is consuming more in high resources")

                axes[plot_idx].bar(tests, bar_heights, color=colors)
                if setting:
                    plot_title = "Reproduction on"
                else:
                    plot_title = "Reproduction off"
                axes[plot_idx].set_xlabel(plot_title)
                axes[plot_idx].set_ylabel(metric_labels[metric])
                axes[plot_idx].errorbar(tests, bar_heights, bar_errors, ecolor=colors)
                axes[plot_idx].set_xticklabels([env_labels[el] for el in tests])
                #axes[plot_idx].set_ylim([0, 0.7])

            print("Saving figure in ", save_dir)
            fig.tight_layout()
            plt.savefig(save_dir + '/agent_' + str(agent_idx) + '.png', dpi=800)
            plt.savefig(save_dir + '/agent_' + str(agent_idx) + '.pdf', dpi=800)
            plt.clf()

        print(signif_result / len(agents),
              " % of the agents had statistically significant differences across lab environments")


if __name__ == "__main__":
    projects = ["projects/pretrained/seed0"]
    for project in projects:
        fig8(project)
        fig9(project)
