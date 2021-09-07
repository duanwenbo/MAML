from os import PRIO_PGRP
from dm_alchemy import symbolic_alchemy, EnvironmentSettings
import random
import numpy as np
import torch
import os, sys
work_dir = os.getcwd()
sys.path.append(work_dir)
from Environment.navigation import Navigation, NavigationPro, NavigationVal


def sample_alchemy_envs(envs_num):
    envs = []
    seed_list = [10, 20, 30, 40, 50]
    assert len(seed_list) == envs_num, "check you seed list length!"
    level_name = 'alchemy/perceptual_mapping_randomized_with_rotation_and_random_bottleneck'
    for s in seed_list:
        env = symbolic_alchemy.get_symbolic_alchemy_level(level_name, seed=s)
        envs.append(env)
    return envs

def sample_navigation_env(envs_num):
    envs = []
    seed_list = [10, 100, 2, 20, 30,300,40,90,500,55,67,600,79,700,34,678,890,43,67,97]
    assert len(seed_list) == envs_num, "check you seed list length!"
    for s in seed_list:
        initial_state = np.array([9.0,9.0])
        random.seed(s)
        goal = (round(random.uniform(0.,10.,), 1), round(random.uniform(0.,10.), 1))
        envs.append(Navigation(goal))
    return envs


def sample_navigation_env_fixed(envs_num):
    envs = []
    initial_state = np.array([9.0,9.0])
    goals = [(0.1,0.1),(4.9,0.1),(9.9,0.1),
             (0.1,9.9),(4.9,9.9),(9.9,8.9)]
    # goals = [(0.1,0.1),(9.9,0.1),(4.9,4.9)]
    for goal in goals:
        envs.append(NavigationPro(goal))
    return envs
    

def validate_env(seed):
    # initial_state = np.array([9.0,9.0])
    random.seed(seed)
    goal = (round(random.uniform(0.,10.,), 1), round(random.uniform(0.,10.), 1))
    # goal = (9.9,4.9)
    return NavigationPro(goal)

def clone_module(module, memo=None):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().

    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.

    **Arguments**

    * **module** (Module) - Module to be cloned.

    **Return**

    * (Module) - The cloned module.

    **Example**

    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone


if __name__ == "__main__":
    sample_navigation_env(5)

