from hashlib import md5
from multiprocessing.dummy import Array
from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

import gym
from gym import spaces
from gym.utils import seeding

from models.battery_simulator import battery


class Actions(Enum):
    Charge = 0
    Discharge = 1
    Standby = 2

class HomerEnv(gym.Env):
    """Home Energy Management Environment that follows gym interface"""
    metadata = {'render_modes': ['human'], "render_fps": 4}

    def __init__(self, capacity=10, start_soc='full', render_mode=None, 
    data=None) -> None:

        """
        TODO: Update docstring.
        """     
        # Set data and devices
        self.df=data   
        self._process_data()

        # Battery Things
        self.start_capacity=capacity
        self.start_soc=start_soc
        self.exportable = True
        self.importable = True
        self.e_flux = None 
        self.net = None 

        # Episode 
        self.start_tick = 0
        self._current_tick = None
        self._end_tick = None
        self.reward=None
        self.episode_reward = None
        self.action = None

        # Rendering
        self._first_rendering = None

        # Data Structure for Logging
        self.history = None
           
        ## Action Spaces
        self.action_space = spaces.Discrete(len(Actions))

        ## Observation Spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.data_arr.shape[1],), 
                                            dtype=np.float32)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None):# -> Tuple[Any, Dict]:
        """
        Resets the episode and performs the first action, which is always 
        putting the battery in standby mode. Assumed to always run before 
        .step()
        
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #Reset reward
        self.episode_reward = 0

        #Reset battery in line with initial operating conditions.
        self.battery = battery(capacity=self.start_capacity, 
                               start_soc=self.start_soc, 
                               importable=self.importable, 
                               exportable=self.exportable, 
                               charge_rate=12, discharge_rate=12)

        # Set first and last tick
        self._current_tick = self.start_tick
        self._end_tick = len(self.df) - 1

        #Get Initial observation and do first step:
        first_obs = self._get_obs() 
        self._do_first_step(first_obs)

        # Get observation and info
        obs = self._get_obs()  
        info = self._get_info()
        self._log_step(info)

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, action):# -> Tuple[Any, float, Any, Dict]:
        """
        Accepts an action, computes the state of the environment after applying
        that action and then returns the 4-tuple (observation, reward, done,
        info). 
        """
        # Take a step, update observation index. 
        self._current_tick += 1
        self.action = action
        obs = self._get_obs()

        # Update Battery and calculate reward
        self.net, self.e_flux = self._update_battery(obs[self.idx['solar']],
                                                    obs[self.idx['loads']], 
                                                    obs[self.idx['max_d']], 
                                                    obs[self.idx['max_c']], 
                                                    action)
        self.reward = self._calculate_reward(self.net,
                                            obs[self.idx['export_tariff']], 
                                            obs[self.idx['import_tariff']])
        self.episode_reward += self.reward 

        # Calculate whether terminated
        done = self._current_tick == self._end_tick

        # Get observation and info
        info = self._get_info()
        self._log_step(info)

        if self.render_mode == "human":
            self._render_frame()

        return obs, self.reward, done, False, info
    
    def _do_first_step(self, first_obs)-> None:
        """
        first obervation is returned from get_obs. 
        array slice of dimension
        """
        
        self.action = Actions.Standby.value
        # Calculate net
        self.net, self.e_flux = self._update_battery(
            first_obs[self.idx['solar']], 
            first_obs[self.idx['loads']], 
            first_obs[self.idx['max_d']],
            first_obs[self.idx['max_c']], 
            self.action)
        # Caclulate Reward        
        self.reward = self._calculate_reward(
            self.net, 
            first_obs[self.idx['export_tariff']], 
            first_obs[self.idx['import_tariff']])
        self.episode_reward += self.reward 

    def _get_obs(self) -> Any:
        """
        returns the state observations. 
        """
        # Get observation indexes. 
        c = self._current_tick
        s = self.idx['solar']
        l = self.idx['loads']
        mc = self.idx['max_c']
        md = self.idx['max_d']
        sc = self.idx['soc']
        # get limits
        solar = self.data_arr[c:c+1, s:s+1]
        loads = self.data_arr[c:c+1, l:l+1]
        max_d, max_c = self.battery.get_limits(solar, loads)
        #Update observations
        self.data_arr[c:c+1, md:md+1] = max_d
        self.data_arr[c:c+1, mc:mc+1] = max_c
        self.data_arr[c:c+1, sc:sc+1] = self.battery.soc

        # Need to flatten the array so that it matches observation dim. 
        return self.data_arr[c:c+1,:].squeeze(0)

    def _get_info(self) -> Dict:
        """ translates the environments state into an observation"""
        return {"reward": self.reward, "net": self.net, "action":self.action,
                "bat_output": self.e_flux}
    
    def _log_step(self, info) -> None:
        """
        logs all steps
        """
        if not self.history:
            self.history = {key:[] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _update_battery(self, solar, loads, max_d, max_c, action
    ) -> tuple[float,float]:
        # Calculate Grid State, Update Battery state
        if action == Actions.Charge.value:
            net = loads + solar + max_c
            self.battery.charge_(max_c)
            e_flux = max_c

        elif action == Actions.Discharge.value:
            net = loads + solar + max_d
            self.battery.discharge_(abs(max_d))
            e_flux = max_d

        elif action == Actions.Standby.value:
            net = loads + solar
            e_flux = 0
        else:
            raise Exception('Action not recognised!')

        return net, e_flux

    def _calculate_reward(self, net, export_tariff, import_tariff):# -> float:
        # Calculate reward
        if net < 0:
            reward = net * export_tariff * -1
        elif net > 0:
            reward = net * import_tariff * -1
        else:
            reward = 0

        return float(reward)

    def render(self, mode='human') -> None:       
        """
        TODO: 
        """
        print(f"Step Reward: {self.reward}\nAction : {self.history['action']}")

    def close(self) -> None:
        #plt.close()
        print(f'Episode Reward {self.episode_reward}')
        print("done")

    def save_rendering(self, filepath) -> None:
        pass #plt.savefig(filepath)
    
    def _process_data(self) -> None:
        """
        Import data
        """
        # Enumerate column indx 
        self.idx = {k: v for v, k in enumerate(self.df.columns)}
        # Full data array
        self.data_arr = self.df.to_numpy(copy=True, dtype=np.float32)