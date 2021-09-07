import torch
from copy import deepcopy
from Algorithms.Differentiable_SGD import DifferentiableSGD
from torch.distributions import Categorical


class MetaLeaner():
    def __init__(self, env, policy_net, baseline_net, learning_rate, device, inner_algo) -> None:
        self.env = env
        self.device = device
        self.inner_algo = inner_algo

        self.policy_net = policy_net.to(self.device)
        self.baseline_net = baseline_net.to(self.device)

        self.policy_opt = DifferentiableSGD(module=self.policy_net)
        self.baseline_opt = torch.optim.Adam(self.baseline_net.parameters(), lr=learning_rate)
    
    def _choose_action(self, state):
        """
        state: array, from the env
        return: action (init), log_prob 
        """
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        distribution = Categorical(self.policy_net(state))
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob
    
    def _get_state_value(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        state_value = self.baseline_net(state)
        return state_value

    def sample_trajectory(self):
        states = []
        rewards = []
        actions = []
        next_states = []
        log_probs =[]
        state_values =[]
        timestep = self.env.reset()
        # step_num = 0
        while not timestep.last():
            # step_num += 1
            # print(step_num)
            state = timestep.observation["symbolic_obs"]
            action, log_prob = self._choose_action(state)
            state_value = self._get_state_value(state)
            timestep = self.env.step(action)
            next_state = timestep.observation["symbolic_obs"]

            states.append(state)
            rewards.append(timestep.reward)
            actions.append(action)
            next_states.append(next_state)
            log_probs.append(log_prob)
            state_values.append(state_value)
            state = next_state
        return {"states":states, "rewards":rewards, "actions":actions, 
                "next_states":next_states, "log_probs":log_probs, "state_values":state_values
                }

    
    def learn(self, trajectory):
        # see the current scores
        current_rewards = sum(trajectory["rewards"])

        # fit baseline first
        self.baseline_opt.zero_grad()
        baseline_loss = self.inner_algo(trajectory, self.device).baseline_loss()
        baseline_loss.backward()
        self.baseline_opt.step()

        # fit policy net
        policy_loss = self.policy_loss(trajectory)
        self.policy_opt.set_grads_none()
        policy_loss.backward(retain_graph=True)
        with torch.set_grad_enabled(True): 
            self.policy_opt.step()
            
        return current_rewards
    
    def policy_loss(self, trajectory):
        baseline_loss = self.inner_algo(trajectory, self.device).policy_loss()
        return baseline_loss






if __name__ == "__main__":
    class A:
        def __init__(self) -> None:
            self.name = "Bobby"
    a = A
    print(a().name)