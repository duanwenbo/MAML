"""
validate the performance difference between meta_learned model and others
plot the result
"""
import sys, os
from warnings import formatwarning
work_dir = os.getcwd()
sys.path.append(work_dir)
from MAML.Algorithms.VPG import VPG
from MAML.Networks.networks import StateValueNet, CategoricalPolicy
import torch
from MAML.Utilities.functional import validate_env
from MAML.Utilities.RL_agent import RL_agent
import csv
import random
from torch.distributions import Categorical

class Trained_agent:
    def __init__(self, step_num, model_path, env, seed_index, repeat_time) -> None:
        self.step_num = step_num
        self.model_path = model_path
        self.env = env
        self.seed_index = seed_index
        self.repeat_time = repeat_time

    def validate(self):
        agent = RL_agent(env= self.env,
                        policy_net=torch.load(self.model_path),
                        baseline_net=StateValueNet(input_size=4, hidden_size=64, output_size=1),
                        learning_rate=0.00005,
                        device="cpu",
                        inner_algo=VPG)
        for i in range(self.step_num):           
            trajectory = agent.sample_trajectory()
            distance = agent.learn(trajectory)
            with open("validation.csv", "a+") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([self.seed_index, self.repeat_time,"MAML", i, distance])
        print("model test finised")



class Random_agent:
    def __init__(self, step_num, env, seed_index, repeat_time) -> None:
        self.step_num = step_num
        self.env = env
        self.seed_index = seed_index
        self.repeat_time =repeat_time
    
    def validate(self):
        for i in range(self.step_num):
            done = False
            while not done:
                action = random.randint(0, self.env.action_space.n - 1)
                next_state, reward, done, distance = self.env.step(action)
            with open("validation.csv", "a+") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([self.seed_index, self.repeat_time, "random", i, distance])   
        print("random test finished") 


class Pretained_agent:
    def __init__(self, step_num, model_path, env, seed_index, repeat_time) -> None:
        self.step_num = step_num
        self.policy_net = torch.load(model_path)
        self.env = env
        self.seed_index = seed_index
        self.repeat_time = repeat_time

    def validate(self):
        for i in range(self.step_num):
            state = self.env.reset()
            done = False
            while not done:
                state = torch.as_tensor(state, dtype=torch.float32)
                action = Categorical(self.policy_net(state)).sample().item()
                next_state, reward, done, distance = self.env.step(action)
                state = next_state
            with open("validation.csv", "a+") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([self.seed_index, self.repeat_time, "pretrained", i, distance])   
        print("random test finished") 



def validate():
    seed_list = [128,258,852,364]
    repeat_time = 3
    model_path = "policy_net_0902_04.pkl"
    for seed_index, seed in enumerate(seed_list):
        for repeat_index in  range(repeat_time):
            Trained_agent(step_num=4,
                        model_path=model_path,
                        env=validate_env(seed),
                        seed_index=seed_index,
                        repeat_time=repeat_index).validate()
            Random_agent(step_num=4,
                        env=validate_env(seed),
                        seed_index=seed_index,
                        repeat_time=repeat_index).validate()
            Pretained_agent(step_num=4,
                            model_path="policy_net_{}_0902.pkl".format(seed),
                            env=validate_env(seed),
                            seed_index=seed_index,
                            repeat_time=repeat_index).validate()



def pretrained(episode_num, seed):
    env = validate_env(seed)
    agent = RL_agent(env=env,
                    policy_net=CategoricalPolicy(input_size=env.observation_space.shape[0],
                                                    hidden_size=64, 
                                                    output_size=env.action_space.n),
                    baseline_net=StateValueNet(input_size=env.observation_space.shape[0],
                                                hidden_size=64,
                                                output_size=1),
                    learning_rate=0.000015,
                    device="cpu",
                    inner_algo=VPG)
    for i in range(episode_num):
        trajectory = agent.sample_trajectory()
        if trajectory["distance"] <= 0.02:
            print("finished")
            break
        else:
            distance = agent.learn(trajectory)    
            print("episode: {}   distance: {}".format(i, round(distance,3)))
    torch.save(agent.policy_net, "policy_net_{}_0902.pkl".format(seed))
                        




if __name__ == "__main__":
    validate()
