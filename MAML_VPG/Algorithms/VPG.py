import numpy as np
import torch


class VPG:
    def __init__(self, trajectory, device) -> None:
        self.states = trajectory["states"]
        self.rewards = trajectory["rewards"]
        self.actions = trajectory["actions"]
        self.log_probs = trajectory["log_probs"]
        self.state_values = trajectory["state_values"]
        self.device = device
    

    def _advantage_function(self):
        # Generalize Advantage Estimator
        # delta R(st,at) + gamma * V(st') - V(st)
        # extract state value list, which contains grad fn before
        self.state_values = [state_value.item() for state_value in self.state_values]
        rewards, state_values = np.array(self.rewards), np.array(self.state_values)
        delta = rewards[:-1] + 0.98 * state_values[1:] - state_values[:-1]

        # compute discounted accumulation
        for i in reversed(range(len(delta))):
            delta[i] = delta[i] + 0.98 * 0.97 * delta[i + 1] if i + 1 < len(delta) else delta[i]
        # gae_advantage = torch.as_tensor(delta, dtype=torch.float32)
        return delta 
    
    def _reward2go(self):
        # discounted return
        rtg = np.zeros_like(self.rewards)
        addition = 0
        for i in reversed(range(len(self.rewards))):
            rtg[i] = self.rewards[i] + addition
            addition = rtg[i] * 0.98
        return rtg[:-1]


    def policy_loss(self):
        """
        policy loss: E[log probability * advantage function]
        """
        policy_loss = 0.
        advantages = self._advantage_function()
        assert len(advantages) == len(self.log_probs) - 1, "length between advantage and log_probs won't matched"
        for log_prob, advantage in zip(self.log_probs[:-1], advantages):
            policy_loss +=  log_prob * advantage
        policy_loss =  - policy_loss.mean()
        assert type(policy_loss) == torch.Tensor, "data type error, suppose getting a tensor"
        return policy_loss
    
    def baseline_loss(self):
        rtgs = self._reward2go()
        assert len(rtgs) == len(self.state_values[:-1]), "length between reward to go and state values won't matched"
        baseline_loss = 0.
        for rtg, state_value in zip(rtgs, self.state_values):
            baseline_loss += (rtg - state_value) ** 2
        baseline_loss = baseline_loss.mean()
        return baseline_loss



        