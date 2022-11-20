"""
Date: 17th November 2022

Gym Environment for Home Energy Management

Authors: Alexander To and Brody Cormier
Version: v1
"""


from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import os
import gym
from gym import spaces
import pandas as pd

from models.battery_simulator import battery

class Actions(Enum):
    Charge = 2
    Discharge = 1
    Standby = 0

class HomerEnv(gym.Env):
    """Home Energy Management Environment that follows gym interface"""
    metadata = {'render_modes': ['human'], "render_fps": 4}

    def __init__(self, capacity=14, start_soc='full', render_mode=None, 
        discrete=True, data=None, charge_rate=5, action_intervals=10, 
        save_history=False, save_path="", device_id=None) -> None:

        """
        Initialises a HOMER Env.
        """     
        # Set data and devices
        self.discrete=discrete
        self.df=data   
        self._process_data()
        self.save_history=save_history
        self.save_path=save_path
        self.device_id=device_id

        self.action_intervals = action_intervals

        # Battery Things
        self.start_capacity=capacity
        self.start_soc=start_soc
        self.exportable = True
        self.importable = True
        self.e_flux = None 
        self.net = None 
        self.charge_rate = charge_rate
        self.discharge_rate = charge_rate

        # Episode 
        self.start_tick = 0
        self._current_tick = None
        self._end_tick = None
        self.reward=None
        self.cumulative_reward = None
        self.action = None

        # Rendering
        self._first_rendering = None
           
        ## Action Spaces
        if self.discrete:
            self.action_space = spaces.Discrete(3, start=0)
        # Pseudo continuous with n=action_intervals fractional increments for
        # charge/discharge amounts. 
        else:
            self.action_space = spaces.Discrete((2*self.action_intervals)+1, start=0)

        ## Observation Spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.data_arr.shape[1],), 
                                            dtype=np.float32)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None) -> Tuple[Any, Dict]:
        """
        Resets the episode and performs the first action, which is always 
        putting the battery in standby mode. Assumed to always run before 
        .step()
        
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #Reset reward and history
        # AT - do we need to overwrite the data_arr?
        self.cumulative_reward = 0
        self.history = None
        self.updated_action=None

        #Reset battery in line with initial operating conditions.
        self.battery = battery(capacity=self.start_capacity, 
                               start_soc=self.start_soc, 
                               importable=self.importable, 
                               exportable=self.exportable, 
                               charge_rate=self.charge_rate, 
                               discharge_rate=self.discharge_rate)

        # Set first and last tick
        self._current_tick = self.start_tick
        self._end_tick = len(self.df) - 1

        #Get Initial observation and do first step:
        first_obs = self._get_obs() 
        self._do_first_step(first_obs)

        # Get observation and info
        info = self._get_info()
        self._log_step(info)

        if self.render_mode == "human":
            self._render_frame()

        return first_obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
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
        if self.discrete:
            self.net, self.e_flux = self._update_battery(
                obs[self.idx['solar']],
                obs[self.idx['loads']], 
                obs[self.idx['max_d']], 
                obs[self.idx['max_c']], 
                action
            )
        else:
            
            self.updated_action = (action - self.action_intervals) / self.action_intervals
            self.net, self.e_flux = self._apply_action(
                self.updated_action, 
                obs[self.idx['loads']] + obs[self.idx['solar']]
            )

        self.reward = self._calculate_reward(
            self.net,
            obs[self.idx['export_tariff']],
            obs[self.idx['import_tariff']]
        )
        self.cumulative_reward += self.reward 

        # Calculate whether terminated
        done = self._current_tick == self._end_tick

        # Get observation and info
        info = self._get_info()
        self._log_step(info)

        # Save output dataframe.
        if done and self.save_history:
            self.save_results()
        
        if self.render_mode == "human":
            self._render_frame()

        return obs, self.reward, done, False, info
    
    def _do_first_step(self, obs)-> None:
        """
        first obervation is returned from get_obs. 
        array slice of dimension
        """
        
        self.action = Actions.Standby.value
        # Calculate net
        self.net, self.e_flux = self._update_battery(
            obs[self.idx['solar']], 
            obs[self.idx['loads']], 
            obs[self.idx['max_d']],
            obs[self.idx['max_c']], 
            self.action)
        # Caclulate Reward        
        self.reward = self._calculate_reward(
            self.net, 
            obs[self.idx['export_tariff']], 
            obs[self.idx['import_tariff']])
        self.cumulative_reward += self.reward 

    def _get_obs(self) -> np.ndarray:
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
        """ 
        translates the environments state into an observation
        """
        info_dict = {
            "tick": self._current_tick,
            "reward": self.reward, 
            "net": self.net, 
            "action":self.action,
            "updated_action":self.updated_action,
            "bat_output": self.e_flux, 
            "cumulative_reward":self.cumulative_reward
        }
        return info_dict
    
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
            net = loads + solar + (max_c)
            _, e_flux, _ = self.battery.charge(max_c)
            if e_flux < 0:
                raise Exception
            #e_flux = max_c
        elif action == Actions.Discharge.value:
            net = loads + solar + (max_d)
            _, e_flux, _ = self.battery.discharge(max_d)
            if e_flux > 0:
                raise Exception
            #e_flux = max_d
        elif action == Actions.Standby.value:
            net = loads + solar
            e_flux = 0
        else:
            raise Exception(f'Action not recognised! a{action}, d{max_d}, c{max_c}')
        return net, e_flux

    def _apply_action(self,action, home):
        if action > 0:
            charge_request = action * self.battery.max_input
            _, e_flux, _ = self.battery.charge(charge_request)
        else:
            discharge_request = action * self.battery.max_output
            _, e_flux, _ = self.battery.discharge(discharge_request)
        
        net = home + e_flux
        return net, e_flux

    def _calculate_reward(self, net, export_tariff, import_tariff) -> float:
        # Calculate reward
        if net < 0:
            reward = net * export_tariff
            reward = reward if reward > 0 else reward * -1
        elif net > 0:
            reward = net * import_tariff
            reward = reward if reward < 0 else reward * -1
        else:
            reward = 0
        return float(reward)

    def render(self, mode='human') -> None:
        if mode == 'human':

            fig, axs = plt.subplots(1, 2, layout="constrained")

            # Set Colour
            c = 'red'
            if self.battery.soc > 0.7:
                c = 'green'
            elif 0.3 < self.battery.soc <= 0.7:
                c = 'orange'

            bar_1 = axs[0].bar(['Battery Capacity'], self.battery.soc * 100, 
                color=c, alpha = 0.8)
            axs[0].set_ylim(0,100)
            axs[0].bar_label(bar_1, label_type='center', fmt='%.2f')
            axs[0].set_ylabel('Percent %')

            s = 'black'
            if self.e_flux > 0:
                s = 'blue'
            elif self.e_flux < 0:
                s = 'yellow' 
            bar_2 = axs[1].bar(['Battery Output'], self.e_flux, 
                color=s, alpha=0.8)
            axs[1].set_ylim(self.battery.max_output * -1,   
                            self.battery.max_input)
            axs[1].bar_label(bar_2, label_type='center', fmt='%.2f')
            axs[1].set_ylabel('Kilowatts kW')

            if self.action == Actions.Charge.value:
                ac = 'Charging!'
            elif self.action == Actions.Discharge.value:
                ac = 'Discharging!'
            else:
                ac = 'Standby'
                
            title = str(f'Step: {self._current_tick}'
                    f'Action: {ac} Net Energy: {self.net:.4f} kW'
                    f"\nStep Reward: \${self.reward:.2f},"
                    f"Cumulative Reward: \${self.cumulative_reward:.2f}")
            fig.suptitle(
                f'Step: {self._current_tick}'
                f'Action: {ac} Net Energy: {self.net:.4f} kW'
                f"\nStep Reward: \${self.reward:.2f},"
                f"Cumulative Reward: \${self.cumulative_reward:.2f}",
                y=1.1)
            self.save_rendering(f'{os.getcwd()+"/renders/"}plot')

        else:
            pass
            print(f"Step Reward: {self.reward}\n"
                f"Action : {self.history['action'][self._current_tick]}")

    def close(self) -> None:
        
        plt.close()

    def save_rendering(self, filepath) -> None:
        plt.savefig(f'{filepath}_{str(self._current_tick)}.png')
        
    def save_results(self):
        
        #Save
        obs = pd.DataFrame(data=self.data_arr, columns=self.df.columns)
        info = pd.DataFrame(self.history)
        results = pd.merge(obs, info, left_index=True, right_index=True)
        # Note for repetitions this will overwrite previous repetitions.
        device = self.device_id if self.device_id != None else "dummy"
        results.to_csv(
            self.save_path+f"/{device}_results_array.csv", index=False
        )
    
    def _process_data(self) -> None:
        """
        Import data from the passed dataframe into a numpy array. 
        all operations performed inplace for speed. 
        """
        # Enumerate column indx - useful for indexing later 
        self.idx = {k: v for v, k in enumerate(self.df.columns)}
        # Full data array
        self.data_arr = self.df.to_numpy(copy=True, dtype=np.float32)
        
    def seed(self, seed):
        np.random.seed(seed)
