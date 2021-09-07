from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import os
from scipy.interpolate import interp1d



class Navigation_old(Env):
    metadata = {"render.modes":["console"]}
    def __init__(self) -> None:
        super(Navigation_old, self).__init__()
        self.action_space = Discrete(8)
        self.observation_space = Box(low=np.array([0,0]), high=np.array([7.07,7.07]))
        self.state = np.array([7.07,7.07])
        self.episode_length = 50
        self.action_dict = {"0":(0,0.2),"1":(0.2,0.2),"2":(0.2,0),"3":(0.2,-0.2),"4":(0,-0.2),"5":(-0.2,-0.2),"6":(-0.2,0),"7":(-0.2,0.2)}
        # self.goal = (random.uniform(0,10), random.uniform(0,10))
        self.goal = (0., 0.) # for model test
        self.reward = lambda x: 0 if x >10.5 or x<=0 else ((10)**(-x/10))*4.976

    def step(self, action):
        self.episode_length -= 1
        # action = self.map(np.tanh(action))
        # x_increment, y_increment = 0.3 * math.cos(math.radians(action)), 0.3 * math.sin(math.radians(action))
        self.state = (self.state[0]+ self.action_dict[str(action)][0], self.state[1]+self.action_dict[str(action)][1])
        distance = ((self.state[0] - self.goal[0])**2 + (self.state[1] - self.goal[1])**2)**0.5
        reward = self.reward(distance)
        if self.episode_length <=0:
            done = True
        else:
            done = False
    
        return np.array(self.state), reward, done, distance

    def reset(self):
        self.state =  np.array([7.07, 7.07])
        self.episode_length = 50
        return self.state

    def render(self):
        pass


class Navigation(Env):
    metadata = {"render.modes":["console"]}
    def __init__(self, goal) -> None:
        super(Navigation, self).__init__()
        self.action_space = Discrete(8)
        self.observation_space = Box(low=np.array([0,0]), high=np.array([10,10]))
        self.state = np.array([9.,9.])
        self.episode_length = 100
        self.action_dict = {"0":(0,0.1),"1":(0.1,0.1),"2":(0.1,0),"3":(0.1,-0.1),"4":(0,-0.1),"5":(-0.1,-0.1),"6":(-0.1,0),"7":(-0.1,0.1)}
        # self.goal = (random.uniform(0,10), random.uniform(0,10))
        self.goal = goal # for model test

    def step(self, action):
        self.episode_length -= 1
        x_increment, y_increment = self.action_dict[str(action)][0], self.action_dict[str(action)][1]
        # x_increment, y_increment = 0.3 * math.cos(math.radians(action)), 0.3 * math.sin(math.radians(action))
        self.state = (self.state[0]+ x_increment, self.state[1]+y_increment)
        distance = ((self.state[0] - self.goal[0])**2 + (self.state[1] - self.goal[1])**2)
        # amplify_factor= max([0.008, self.episode_length / 100]) # normalized amplified factor
        reward = (-distance) 
        if self.episode_length <=0:
            done = True
        elif distance <= 0.05:
            done = True
        else:
            done = False
        return np.array(self.state), reward, done, distance

    def reset(self):
        self.state =  np.array([9.,9.])
        self.episode_length = 100
        return self.state

    def render(self):
        pass


class NavigationPro(Env):
    metadata = {"render.modes":["console"]}
    def __init__(self, goal) -> None:
        super(NavigationPro, self).__init__()
        self.action_space = Discrete(8)
        self.observation_space = Box(low=np.array([0,0,0,0]), high=np.array([10,10,10,10]))
        self.goal = goal
        self.state = np.array([4.9,4.9,self.goal[0], self.goal[1]])
        self.episode_length = 100
        self.action_dict = {"0":(0,0.1),"1":(0.1,0.1),"2":(0.1,0),"3":(0.1,-0.1),"4":(0,-0.1),"5":(-0.1,-0.1),"6":(-0.1,0),"7":(-0.1,0.1)}
        # self.goal = (random.uniform(0,10), random.uniform(0,10))

    def step(self, action):
        self.episode_length -= 1
        x_increment, y_increment = self.action_dict[str(action)][0], self.action_dict[str(action)][1]
        # x_increment, y_increment = 0.3 * math.cos(math.radians(action)), 0.3 * math.sin(math.radians(action))
        self.state = (self.state[0]+ x_increment, self.state[1]+y_increment,self.goal[0], self.goal[1])
        distance = ((self.state[0] - self.goal[0])**2 + (self.state[1] - self.goal[1])**2)
        # amplify_factor= max([0.008, self.episode_length / 100]) # normalized amplified factor
        reward = (-distance) 
        if self.episode_length <=0:
            done = True
        elif distance <= 0.05:
            done = True
        else:
            done = False
        return np.array(self.state), reward, done, distance

    def reset(self):
        self.state =  np.array([4.9,4.9,self.goal[0], self.goal[1]])
        self.episode_length = 100
        return self.state

    def render(self):
        pass


class NavigationVal(Env):
    metadata = {"render.modes":["console"]}
    def __init__(self, goal) -> None:
        super(NavigationVal, self).__init__()
        self.action_space = Discrete(8)
        self.observation_space = Box(low=np.array([0,0,0,0]), high=np.array([20,20,20,20]))
        self.goal = goal
        self.state = np.array([9.,9.,self.goal[0], self.goal[1]])
        self.episode_length = 100
        self.action_dict = {"0":(0,0.2),"1":(0.2,0.2),"2":(0.2,0),"3":(0.2,-0.2),"4":(0,-0.2),"5":(-0.2,-0.2),"6":(-0.2,0),"7":(-0.2,0.2)}
        # self.goal = (random.uniform(0,10), random.uniform(0,10))

    def step(self, action):
        self.episode_length -= 1
        x_increment, y_increment = self.action_dict[str(action)][0], self.action_dict[str(action)][1]
        # x_increment, y_increment = 0.3 * math.cos(math.radians(action)), 0.3 * math.sin(math.radians(action))
        self.state = (self.state[0]+ x_increment, self.state[1]+y_increment,self.goal[0], self.goal[1])
        distance = ((self.state[0] - self.goal[0])**2 + (self.state[1] - self.goal[1])**2)
        # amplify_factor= max([0.008, self.episode_length / 100]) # normalized amplified factor
        reward = (-distance) 
        if self.episode_length <=0:
            done = True
        elif distance <= 0.05:
            done = True
        else:
            done = False
        return np.array(self.state), reward, done, distance

    def reset(self):
        self.state =  np.array([9.,9.,self.goal[0], self.goal[1]])
        self.episode_length = 100
        return self.state

    def render(self):
        pass






if __name__ == "__main__":
    a = np.array([1,2])
    b = np.array([4,4,a[0]])
    print(b)