from selectors import EpollSelector
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

import gym
from gym import spaces
from gym.utils import seeding

from models.battery_simulator import battery

"""
To Do:
- Work out the state variables to check.
    - Temporal: Month, Day of Week, Time
    - Regional: The state, one hot encoded vector
    - Energy: Solar, Loads, Net balance (with Battery)
    - Price: Import tariff, export tariff
    - Battery: Available Energy, discharge_limit, charge_limit
    - Forecasts: NOT IMPLEMENTED 
- Update class attributes required
- Update data handler class.
- Update observation state
- Update get_obs and get_info
- Update _calculate_rewards, _update_battery_
- check render, check reset


"""

class Actions(Enum):
    Charge = 0
    Discharge = 1
    Standby = 2

class HomerEnv(gym.Env):
    """Home Energy Management Environment that follows gym interface"""
    metadata = {'render_modes': ['human'], "render_fps": 4}

    def __init__(self, capacity=10, start_soc='full', devices=['test_device'], 
                render_mode=None, data = main_df) -> None:

        """
        avl_energy (str): "full", "empty", 'random'
        """

        # TODO: DataFrame, Reward Configuration
        self.start_capacity=capacity
        self.start_soc=start_soc
        self.devices=devices
        self.df=main_df
        self.regions=self.df.region.unique()
        
        # Instantiate a battery
        self.battery = battery(capacity=self.start_capacity, 
                               start_soc=self.start_soc, 
                               importable=True, dischargeable=True, 
                               charge_rate=12, discharge_rate=12)
        
        # Action Spaces
        self.action_space = spaces.Discrete(len(Actions))

        # TODO: Observation space
        self.observation_space = spaces.Dict({
            "solar": spaces.Box(low=0, high=np.inf, shape=len(self.devices,), 
                                dtype=np.float32),
            "loads": spaces.Box(low=0, high=np.inf, shape=len(self.devices,), 
                                dtype=np.float32),
            "net": spaces.Box(low=0, high=np.inf, shape=len(self.devices,),
                            dtype=np.float32),
            "battery": spaces.Dict({
                "state_of_charge": spaces.Box(low=0, high=np.inf, 
                                            shape=len(self.devices,), 
                                            dtype=np.float32),
                "current_discharge_limit": spaces.Box(low=-np.inf, high=0, 
                                                    shape=len(self.devices,), 
                                                    dtype=np.float32),
                "current_charge_limit": spaces.Box(low=0, high=np.inf, 
                                                    shape=len(self.devices,), 
                                                    dtype=np.float32),
                "battery_output": spaces.Box(low=0, high=np.inf, 
                                            shape=len(self.devices,), 
                                            dtype=np.float32),
                "agent_action": spaces.MultiBinary([len(self.devices),
                                                    len(Actions)])}),
            "region": spaces.MultiBinary([len(self.devices),
                                          len(self.regions)-1]),
            "price": spaces.Dict({
                "import_tariff": spaces.Box(low=0, high=np.inf, 
                                            shape=len(self.regions,), 
                                            dtype=np.float32),
                "export_tariff": spaces.Box(low=0, high=np.inf, 
                                            shape=len(self.regions,), 
                                            dtype=np.float32)})
        })
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        #Episode TODO: Resolve list and counters. 
        self.start_tick = 0
        self._end_tick = len(self.df) - 1
        self._current_tick = 0
        self._battery_state = None
        self._first_rendering = None
        self.total_reward = None
        self.window = None
        self.clock = None

    def _get_obs(self):

        # TODO: FIALISE data strcture for observations
        return {
            "solar": self._current_solar, "grid": self._current_grid,
            "load": self._current_loads, "battery": self._current_battery_state,
            "price": self._current_price
            }

    def _get_info(self):
        """ translates the environments state into an observation"""
        #TODO: FINISH OFF THE INFO!
        return {"forecasts":{"price":[], "loads":[], "weather":[]}  }

    def _update_battery(self, action):
        # TODO: UPDATE VARIABLES for CUrrenT STATE

        # Update Battery and calculate reward
        max_d, max_c = self.battery.get_limits(- SOLAR, LOADS)

        # Calculate Grid State, Update Battery state
        if action == Actions.Charge.value:
            net = HOME + SOLAR + max_c
            self.batttery.charge_(max_c)

        elif action == Actions.Discharge.value:
            net = HOME + SOLAR + max_d
            self.batttery.discharge_(max_d)

        elif action == Actions.Standby.value:
            net = HOME + SOLAR

        return net

    def _calculate_reward(self, net):
        # TODO Update variables
        if net < 0:
            reward = grid * export_tariff
        elif net > 0:
            reward = grid * import_tariff
        else:
            reward = 0

        self.total_reward += reward

        return reward

    def step(self, action):
        """
        Accepts an action, computes the state of the environment after applying
        that action and then returns the 4-tuple (observation, reward, done,
        info). 
        """

        # Take a step, update observation index. 
        self._current_tick += 1

        # Update Battery and calculate reward
        net_energy = self._update_battery(action)
        reward = self._calculate_reward(net_energy)

        # Calculate whether terminated
        terminated = self._current_tick == self._end_tick

        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()


        return observation, reward, terminated, info

    def reset(self, seed=None, options=None):
        # TODO RESET ALL THINGS
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #Reset battery in line with initial operating conditions.
        self.battery.reset(self.start_capcity, self.start_soc)

        # An episode is done iff the end of the series is reached.  

        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info 

    def render(self, mode='human'):

        ## FIX UP RENDERS
        
        if self._first_render:
            self._first_render = False
            plt.cla()
            plt.plot()
        
        plt.suptitle(
            f"Total Reward: {self.reward}\n"
            f"Action : {self.state_description}kW"
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def _process_data(self):

        ### FINALISE THIS!
        
        # Get data from the dataframe
        solar = self.data.loc[:, 'Solar Consumption'].to_numpy()
        home = self.data.loc[:, 'Home Consumption'].to_numpy()
        import_tariff = self.data.loc[:, 'tariff'].to_numpy()
        export_tariff = self.data.loc[:, 'tariff'].to_numpy()
        net = home - solar

        return {
            "solar": solar, "home": home, "import": import_tariff,
            "export": export_tariff, "net": net 
        }