from typing import Pattern
from Utilities.functional import validate_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import os
import sys

work_dir = os.getcwd()
sys.path.append(work_dir)
sns.set(rc = {'figure.figsize':(10,5)})

class Plotter:
    def __init__(self,result_path) -> None:
        self.batch_goals = DataFrame({'x':[0.1,4.9,9.9,0.1,4.9,9.9,4.9],
                                      'y':[0.1,0.1,0.1,9.9,9.9,9.9,4.9],
                                      'type':["goal","goal","goal","goal","goal","goal","initial_position"],
                                      'size':[10,10,10,10,10,10,5]})
        self.result_data = pd.read_csv(result_path)
    
    def plot(self):
        fig, axes = plt.subplots(1,2,sharex=False)
        fig.suptitle("MAML in Navigation Env Training Result (No Additional Info) 0902")

        """visualize the goal first"""
        axes[0].set_title("Point Locations")
        sns.scatterplot(ax=axes[0], data=self.batch_goals, x='x', y='y', hue='type', style='type')

        """visualize the training result"""
        axes[1].set_title("Training Result")
        data = self.result_data.query("type=='episode_distance'")
        sns.lineplot(ax=axes[1], data=data, x='episode', y='distance', hue='type')


        plt.savefig("/home/gamma/wb_alchemy/Customize_env_test/MAML_test_env/tranined_result_090202.png",dpi=400)
    
    def goals(self):
        plt.figure(figsize=(3,3))
        plt.title("Task Overview Navigation Environment 0902")
        sns.scatterplot(data=self.batch_goals, 
                        x='x',
                        y='y', 
                        style='type',
                        hue="type")
        plt.legend(fontsize='xx-small')
        plt.savefig("/home/gamma/wb_alchemy/Customize_env_test/MAML_test_env/goal_090201.png",dpi=400)
    
    def trained_result(self):
        fig, axes = plt.subplots(2,1,sharex=True, sharey=True)
        axes[0].set_title("MAML training result in Navigation Environmrnt (With Additional Info")
        data = self.result_data.query("type=='episode_distance'")
        data["4step_average_distance"] = data["distance"].ewm(span=80).mean()
        fig = sns.lineplot(ax=axes[0], data=data, x='episode', y='distance', hue='type',palette=["gray"], alpha=0.4)
        fig = sns.lineplot(ax=axes[0], data=data, x='episode', y='4step_average_distance', palette=["blue"])

        axes[1].set_title("MAML training result in Navigation Environmrnt (Without Additional Info")
        data = pd.read_csv("experiment_data_0901.csv")
        data = data.query("type=='episode_distance'")
        data["4step_average_distance"] = data["distance"].ewm(span=80).mean()
        fig = sns.lineplot(ax=axes[1], data=data, x='episode', y='distance', hue='type',palette=["gray"], alpha=0.4)
        fig = sns.lineplot(ax=axes[1], data=data, x='episode', y='4step_average_distance', palette=["blue"])

        lineplot_fig = fig.get_figure()
        lineplot_fig.savefig("/home/gamma/wb_alchemy/Customize_env_test/MAML_test_env/result_090202.png",dpi=400)


def validation_plot(validate_result_path):
    seed_list = [128,258,852,364]
    df = pd.read_csv(validate_result_path)
    fig, axes = plt.subplots(2,2,sharex=True, sharey=True)
    fig.suptitle('Validation Result')
    palette = {"MAML":'red',"random":"gray", "pretrained":"blue"}
    for i in range(4):
        axes[i%2,i//2].set_title('seed {}'.format(seed_list[i]))
        data = df.query("seed_index=={}".format(i))
        if i != 3:
            fig=sns.lineplot(ax =axes[i%2,i//2],  data=data, x='episode', y='distance', hue='type', palette=palette, style='type', legend=False)
        else:
            fig=sns.lineplot(ax =axes[i%2,i//2],  data=data, x='episode', y='distance', hue='type', palette=palette, style='type')
            plt.legend(fontsize='xx-small',bbox_to_anchor=(1.25,1.3))
        print("finish {} plot".format(i))
    plt.tight_layout()
    lineplot_fig = fig.get_figure()
    lineplot_fig.savefig("/home/gamma/wb_alchemy/Customize_env_test/MAML_test_env/validate_090203.png",dpi=400)


if __name__ == "__main__":
    # validate_result_path = "validation.csv"
    # validation_plot(validate_result_path)
    path = "experiment_data_0902_04.csv"
    Plotter(path).trained_result()
