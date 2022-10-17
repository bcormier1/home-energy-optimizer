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

    def __init__(self, capacity=10, start_soc='empty', devices=['test_device'], 
                render_mode=None, data=None, unbalanced_tariff=True) -> None:

        """
        TODO: Update docstring.
        """     
        # Set data and devices
        self.devices=devices
        self.df=data   
        self._process_data()

        # Battery Things
        self.start_capacity=capacity
        self.start_soc=start_soc
        self.exportable = True
        self.importable = True
        self.e_flux = None 
        self.net = None 

        # Set tariff settings 
        self.unbalanced_tariff = unbalanced_tariff

        # Episode 
        self.start_tick = 0
        self._current_tick = None
        self._end_tick = None
        self.reward=None
        self.episode_reward = None
        self.action = None

        # Rendering
        self._first_rendering = None
        self.window = None
        self.clock = None

        # Data Structure for Logging
        self.history = None
           
        ## Action Spaces
        self.action_space = spaces.Discrete(len(Actions))
        
        ## Observation space - needs updating?
        n_devices = len(self.devices) 
        n_regions = 1

        self.observation_space = spaces.Dict({
            "solar": spaces.Box(low=-np.inf, high=0, shape=(n_devices,),
                                dtype=np.float32),
            "loads": spaces.Box(low=0, high=np.inf, shape=(n_devices,), 
                                dtype=np.float32),
            "soc": spaces.Box(low=0, high=np.inf, shape=(n_devices,), 
                                        dtype=np.float32),
            "max_discharge": spaces.Box(low=-np.inf, high=0, dtype=np.float32,
                                        shape=(n_devices,)), 
            "max_charge": spaces.Box(low=0, high=np.inf, dtype=np.float32,
                                    shape=(n_devices,)),
            "region": spaces.MultiBinary(3),
            "import_tariff": spaces.Box(low=0, high=np.inf, shape=(n_regions,),
                                        dtype=np.float32),
            "export_tariff": spaces.Box(low=0, high=np.inf, shape=(n_regions,),
                                        dtype=np.float32),
            "time": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "weekend": spaces.MultiBinary(1),
            "month":spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        })
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
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

        #Get Initial observation:
        first_observation = self._get_obs() 
        # Do first action.:
        self._do_first_step(first_observation)

        # Get observation and info
        observation = self._get_obs()  
        info = self._get_info()
        self._log_step(info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, info 

    def step(self, action) -> Tuple[Dict, float, Any, Any, Dict]:
        """
        Accepts an action, computes the state of the environment after applying
        that action and then returns the 4-tuple (observation, reward, done,
        info). 
        """

        # Take a step, update observation index. 
        self._current_tick += 1
        observation = self._get_obs()

        # Update Battery and calculate reward
        self.net, self.eflux = self._update_battery(
            observation['solar'], observation['loads'], 
            observation['max_discharge'], observation['max_charge'], action)

        self.reward = self._calculate_reward(
            self.net, observation['export_tariff'], 
            observation['import_tariff'])

        self.episode_reward += self.reward 

        # Calculate whether terminated
        terminated = self._current_tick == self._end_tick

        # Get observation and info
        info = self._get_info()
        self._log_step(info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, self.reward, terminated, False, info
    
    def _do_first_step(self, first_observation)-> None:
        
        self.action = Actions.Standby.value
        # Calculate net
        self.net, self.e_flux = self._update_battery(
            first_observation['solar'], first_observation['loads'], 
            first_observation['max_discharge'],first_observation['max_charge'], 
            self.action)
        # Caclulate Reward        
        self.reward = self._calculate_reward(
            self.net, first_observation['export_tariff'], 
            first_observation['import_tariff'])
        self.episode_reward += self.reward 


    def _get_obs(self) -> Dict:
        """
        returns the state observations. 
        NOTE: NOT VECTORISED, supports only one device ID!
        """
        idx = self._current_tick

        solar = self.solar_data[idx]
        loads = self.loads_data[idx]
        max_d, max_c = self.battery.get_limits(solar, loads)

        return {
            "solar": self.solar_data[idx:idx+1], 
            "loads": self.loads_data[idx:idx+1], 
            "soc": np.array([self.battery.soc], dtype=np.float32), 
            "max_discharge":np.array([max_d], dtype=np.float32),
            "max_charge":np.array([max_c], dtype=np.float32),
            "region":self.region_data[idx],
            "import_tariff":self.in_tariff_data[idx:idx+1],
            "export_tariff":self.out_tariff_data[idx:idx+1],
            "time":self.time_data[idx],
            "weekend":self.weekend_data[idx:idx+1],
            "month": self.month_data[idx]
        }

    def _get_info(self) -> Dict:
        """ translates the environments state into an observation"""
        return {
            "reward": self.reward, "net": self.net, "action":self.action,
            "bat_output": self.e_flux
        }
    
    def _log_step(self, info) -> None:
        """
        logs all steps
        """
        if not self.history:
            self.history = {key:[] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def _update_battery(self, solar, loads, max_d, max_c, action
    ) -> Tuple[float,float]:
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

    def _calculate_reward(self, net, export_tariff, import_tariff) -> float:

        # Calculate reward
        if net < 0:
            reward = net * export_tariff
        elif net > 0:
            reward = net * import_tariff
        else:
            reward = np.array([0])

        return float(reward)

    def render(self, mode='human'):

        ## FIX UP RENDERS
        #if self._first_render:
        #    self._first_render = False
        #    plt.cla()
        #    plt.plot()
        
        print(
            f"Total Reward: {self.reward}\n"
            f"Action : {self.state_description}kW"
        )

    def close(self):
        #plt.close()
        print("done")

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def _process_data(self):
        
        # Get data from the dataframe
        self.solar_data = self.df.loc[:, 'solar']\
            .to_numpy(copy=True, dtype=np.float32)
        self.solar_data = self.solar_data * -1 if np.max(self.solar_data) > 0 else self.solar_data
        self.loads_data = self.df.loc[:, 'home']\
            .to_numpy(copy=True, dtype=np.float32)
        self.in_tariff_data = self.df.loc[:, 'import_tariff']\
            .to_numpy(copy=True, dtype=np.float32)
        self.out_tariff_data = self.df.loc[:, 'export_tariff']\
            .to_numpy(copy=True, dtype=np.float32)

        self.time_data = self.df.loc[:, 'time'].to_numpy(copy=True)
        self.weekend_data = self.df.loc[:, 'weekend'].to_numpy(copy=True)
        self.month_data = self.df.loc[:, 'month'].to_numpy(copy=True)
        self.region_data = self.df.loc[:, 'region'].to_numpy(copy=True)