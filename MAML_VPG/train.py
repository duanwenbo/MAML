import torch
from Networks.networks import CategoricalPolicy, StateValueNet
from functional import sample_alchemy_envs, clone_module
import yaml
from copy import deepcopy
import wandb
import csv
from meta_learner import MetaLeaner
from copy import deepcopy
from Algorithms.VPG import VPG


# load configs
with open("config.yml", "r") as f:
    configs = yaml.load(f.read())

# initialization
"The main policy networks for meta learner"
policy_net = CategoricalPolicy(input_size=39, hidden_size=128, output_size=40)
baseline_net = StateValueNet(input_size=39, hidden_size=128, output_size=1)
meta_policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=configs["learning_rate"])
device = "cuda:0" if configs["GPU"] else "cpu"
wandb.init("MAML_VPG_ALCHEMY_0830")

# for fixed environment
# envs = sample_alchemy_envs(envs_num=2)  # TODO: using a fixed environmrnt first
# environment =envs[0]

for i in range(configs["episode_num"]):
    cumulative_loss = 0.  # used for optimizing the meta_learner
    cumulative_rewards = 0.  # used for evaluating the mete_learner
    envs = sample_alchemy_envs(envs_num=5) # initialize 5 empty chemistry with different seed value
    # for j in range(10):  # using 10 tasks for each meta optimization
    for j, environment in enumerate(envs):
        meta_learner = MetaLeaner(env=environment,
                                  policy_net=clone_module(policy_net),
                                  baseline_net=deepcopy(baseline_net),
                                  learning_rate=configs["meta_baseline_lr"],
                                  device=device,
                                  inner_algo= VPG
                                  )
        trajectory = meta_learner.sample_trajectory()
        task_rewards = meta_learner.learn(trajectory)
        # collect the policy loss for each task after one step optimization, 
        new_trajectory = meta_learner.sample_trajectory()
        policy_loss = meta_learner.policy_loss(new_trajectory)
        cumulative_loss += policy_loss
        cumulative_rewards += task_rewards
        print("task {}, reward: {}".format(j, task_rewards))
        with open("experiment_data.csv", "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["task_reward", j, task_rewards])
    
    meta_policy_optimizer.zero_grad()
    cumulative_loss.backward()
    meta_policy_optimizer.step()
    
    episode_reward = cumulative_rewards / 5  # 5 tasks for each episode
    print("##########################################")
    print("current epoch:{}, average reward:{}".format(i, episode_reward))
    print("##########################################")
    wandb.log({"episode": i, "average rewards": episode_reward})
    with open("experiment_data.csv", "a+") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["episode_reward", i, episode_reward])
