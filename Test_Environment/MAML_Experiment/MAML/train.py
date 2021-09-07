import torch
from Networks.networks import CategoricalPolicy, StateValueNet
from Utilities.functional import sample_alchemy_envs, clone_module, sample_navigation_env, sample_navigation_env_fixed
import yaml
from copy import deepcopy
import wandb
import csv
from Utilities.meta_learner import MetaLeaner
from copy import deepcopy
from Algorithms.VPG import VPG


# load configs
with open("/home/gamma/wb_alchemy/Customize_env_test/MAML_test_env/MAML/Utilities/config.yml", "r") as f:
    configs = yaml.load(f.read())

# initialization
"The main policy networks for meta learner"
policy_net = CategoricalPolicy(input_size=4, hidden_size=128, output_size=8)
baseline_net = StateValueNet(input_size=4, hidden_size=128, output_size=1)
meta_policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=configs["learning_rate"])
device = "cuda:0" if configs["GPU"] else "cpu"
# wandb.init("MAML_VPG_ALCHEMY_0830")

# for fixed environment
# envs = sample_alchemy_envs(envs_num=2)  # TODO: using a fixed environmrnt first
# environment =envs[0]
try:
    for i in range(configs["episode_num"]):
        cumulative_loss = 0.  # used for optimizing the meta_learner
        cumulative_rewards, cumulative_distance = 0., 0.  # used for evaluating the mete_learner
        envs = sample_navigation_env_fixed(envs_num=6) # initialize 5 empty chemistry with different seed value
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
            task_rewards, last_distance = meta_learner.learn(trajectory)
            # collect the policy loss for each task after one step optimization, 
            new_trajectory = meta_learner.sample_trajectory()
            policy_loss = meta_learner.policy_loss(new_trajectory)
            cumulative_loss += policy_loss
            cumulative_rewards += task_rewards
            cumulative_distance += round(last_distance,3)
            print("task {}, distance: {}".format(j, round(last_distance,3)))
            with open("experiment_data_0902_03.csv", "a+") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["task_distance", j, last_distance])
        
        meta_policy_optimizer.zero_grad()
        cumulative_loss.backward()
        meta_policy_optimizer.step()
        
        episode_distance = round(cumulative_distance / 6 , 3) # 5 tasks for each episode
        print("##########################################")
        print("current epoch:{}, average distance:{}".format(i, episode_distance))
        print("##########################################")
        # wandb.log({"episode": i, "average distance": episode_distance})
        with open("experiment_data_0902_04.csv", "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["episode_distance", i, episode_distance])
        
        if episode_distance <=0.05:
            break
        
    torch.save(policy_net, 'policy_net_0902_04.pkl')
    print("save model successfully !!")

except BaseException:
    torch.save(policy_net, 'policy_net_0902_04.pkl')
    print("programe terminate, save model successfully !!")
