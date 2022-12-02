"""
Date: 17th November 2022

Gym Environment for Home Energy Management

Authors: Andi Morey Peterson, Alexander To and Brody Cormier
Version: v2
"""


from datetime import datetime
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
                 discrete=True, data=None, charge_rate=5, action_intervals=1,
                 save_history=False, save_path="", device_id=None, 
                 benchmarks=True, exportable=True, importable=True,
                 episode_length=0) -> None:

        """
        Initialises a HOMER Env.
        """
        self.standard = False
        self.print_save = False  # Whether to print and save unique data summary and actions.
        self.benchmarks = benchmarks
        self.episode_length = episode_length
        # Set data and devices
        self.discrete = discrete
        self.df = data
        self._process_data()
        self.save_history = save_history
        self.save_path = save_path
        self.device_id = device_id
        self.env_best = -np.inf

        # Battery Things
        self.start_capacity = capacity
        self.start_soc = start_soc
        self.exportable = exportable
        self.importable = importable
        self.e_flux = None
        self.net = None
        self.charge_rate = charge_rate
        self.discharge_rate = charge_rate

        # Episode
        self.episode = False if self.episode_length == 0 else True
        self.max_episode_steps = len(self.data_arr)
        self.n_env_epochs = 0
        self.start_tick = 0
        self._current_tick = None
        self._end_tick = None
        self.reward = None
        self.cumulative_reward = None
        self.action = None
        self.global_tick = 0
        
        if self.benchmarks:
            self.sq_net = None
            self.no_solar_reward = None
            self.no_solar_cumulative_reward = None
            self.no_battery_reward = None
            self.no_battery_cumulative_reward = None
            self.sq_cumulative_reward = None

        # Rendering
        self._first_rendering = None

        ## Define action spaces
        self.action_intervals = action_intervals
        if self.discrete:
            self.action_space = spaces.Discrete((2*self.action_intervals)+1,
                                                start=0)
        else:
            raise NotImplementedError

        ## Observation Spaces
        self.observation_space = spaces.Box(low=-np.inf, 
                                            high=np.inf,
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

        # Reset reward and history, set Episode length
        if self.episode:
            # Get start and end  indexes for each episode.
            self.episode_index_list = self._get_intervals()
            # Internal counter for number of runs through the complete data.
            self.ep_idx = self.n_env_epochs % self.n_episodes
            # Set current and end ticks
            self._current_tick = self.episode_index_list[self.ep_idx]
            self._end_tick = self.episode_index_list[self.ep_idx + 1] - 1
            if self.ep_idx == 0:
                self.history = None
                self.n_env_epochs = 0 if self.n_env_epochs == 0 else self.n_env_epochs +=1
        else:
            self.history = None
            self._current_tick = self.start_tick
            self._end_tick = len(self.df) - 1
            self.n_env_epochs = 0 if self.n_env_epochs == 0 else self.n_env_epochs +=1
        
        self.global_tick = 0 if self.global_tick == 0 else self.global_tick + 1
        self.episode_tick = 0
        self.cumulative_reward = 0
        self.updated_action = None        


        # Reset battery in line with initial operating conditions.
        self.battery = battery(capacity=self.start_capacity,
                               start_soc=self.start_soc,
                               importable=self.importable,
                               exportable=self.exportable,
                               charge_rate=self.charge_rate,
                               discharge_rate=self.discharge_rate)
        sq_start_soc = self.battery.soc
        # Instantiate benchmark things
        if self.benchmarks:
            self.no_solar_cumulative_reward = 0
            self.no_battery_cumulative_reward = 0
            self.sq_cumulative_reward = 0
            self.sq_updated_action = None
            # Reset battery in line with initial operating conditions.
            self.sq_battery = battery(capacity=self.start_capacity,
                                      start_soc=sq_start_soc,
                                      importable=False,
                                      exportable=False,
                                      charge_rate=self.charge_rate,
                                      discharge_rate=self.discharge_rate)

        # Get Initial observation and do first step:
        first_obs = self._get_obs()
        self._do_first_step(first_obs)
        # Get observation and info
        info, extra_info = self._get_info()
        self._log_step(info, extra_info)

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
        self._update_ticks()
        self.action = action
        obs = self._get_obs()

        # Update Battery and calculate reward
        if self.discrete:
            self.updated_action = (action - self.action_intervals) / self.action_intervals
            self.net, self.e_flux = self._apply_action(
                self.updated_action,
                obs[self.idx['loads']] + obs[self.idx['solar']]
            )
        else:  # Continuous action space
            raise NotImplementedError
            
        self.reward = self._calculate_reward(self.net,
                                     obs[self.idx['export_tariff']],
                                     obs[self.idx['import_tariff']])

        self.cumulative_reward += self.reward
        
        ### StatusQuo Operation:
        if self.benchmarks:
            # Get max discharge/charge limits
            sq_max_d, sq_max_c = self.sq_battery.get_limits(obs[self.idx['solar']],
                                                            obs[self.idx['loads']])

            self.sq_net, self.sq_eflux, self.sq_updated_action = self._apply_sq_action(
                obs[self.idx['loads']]+obs[self.idx['solar']],
                sq_max_d,
                sq_max_c
            )

            self.calc_benchmark_rewards(obs[self.idx['loads']],
                            obs[self.idx['solar']],
                            obs[self.idx['export_tariff']],
                            obs[self.idx['import_tariff']])

        # Calculate whether terminated
        done = self._current_tick == self._end_tick

        # Get observation and info
        info, extra_info = self._get_info()
        self._log_step(info, extra_info)

        # Save output dataframe.
        if done and self.save_history and not self.episode:
            if self.env_best < self.cumulative_reward:
                self.save_results()
                self.env_best = self.cumulative_reward
        elif done and self.save_history:
            self.save_results()

        if self.render_mode == "human":
            self._render_frame()

        return obs, self.reward, done, False, info

    def _do_first_step(self, obs) -> None:
        """
        first observation is returned from get_obs.
        array slice of dimension
        """
        self.action = self.action_intervals # First step 'was' always standby
        self.updated_action = 0
        # Calculate net
        self.net, self.e_flux = self._apply_action(
            self.updated_action,
            obs[self.idx['loads']] + obs[self.idx['solar']]
        )

        # Calculate Reward
        self.reward = self._calculate_reward(self.net,
                                             obs[self.idx['export_tariff']],
                                             obs[self.idx['import_tariff']])
        self.cumulative_reward += self.reward

        ### Benchmark Operation:
        if self.benchmarks:
            self.sq_net = obs[self.idx['loads']] + obs[self.idx['solar']]
            self.sq_updated_action = 0  # do nothing
            self.sq_eflux=0
            
            self.calc_benchmark_rewards(obs[self.idx['loads']],
                                        obs[self.idx['solar']],
                                        obs[self.idx['export_tariff']],
                                        obs[self.idx['import_tariff']])

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
        solar = self.data_arr[c:c + 1, s:s + 1]
        loads = self.data_arr[c:c + 1, l:l + 1]

        max_d, max_c = self.battery.get_limits(solar, loads)
        # Update observations
        self.data_arr[c:c+1, md:md+1] = max_d
        self.data_arr[c:c+1, mc:mc+1] = max_c
        self.data_arr[c:c+1, sc:sc+1] = self.battery.soc
        # Need to flatten the array so that it matches observation dim.
        return self.data_arr[c:c+1,:].squeeze(0)

    def _get_info(self) -> Tuple[Dict, Dict]:
        """
        translates the environments state into an observation
        """
        # Info for the agent
        info_dict = {
            "reward": self.reward / 100, 
            "net": self.net, 
            "action":self.action,
            "updated_action":self.updated_action,
            "bat_output": self.e_flux, 
        }
        # Extra info for logging/debug
        extra_info = {
            "tick": self._current_tick,
            "episode_tick": self.episode_tick,
            "global_tick": self.global_tick,
            "env_epochs": self.n_env_epochs,
            "cumulative_reward": self.cumulative_reward,
        }
        if self.benchmarks:
            benchmark_info = {
                "no_solar_cumulative_reward": self.no_solar_cumulative_reward,
                "no_battery_cumulative_reward": self.no_battery_cumulative_reward,
                "sq_cumulative_reward": self.sq_cumulative_reward,
                "sq_soc": self.sq_battery.soc,
                "sq_net": self.sq_net,
                "sq_updated_action": self.sq_updated_action,
                "sq_bat_output":self.sq_eflux
            }
            # Add in the extra info.
            extra_info = extra_info | benchmark_info

        return info_dict, extra_info

    def _update_ticks(self) -> None:
        self.episode_tick+= 1
        self.global_tick += 1

    def _get_intervals(self):
        
        # Get number of complete episodes of length episode_length
        if self.max_episode_steps < self.episode_length:
            raise Exception(f'Episode length {self.max_episode_steps} '
                            f'cannot be less than the total data lenghth {self.episode_length}!')
        self.n_episodes = self.max_episode_steps // self.episode_length
        r = self.max_episode_steps % self.episode_length 
        # Create a list of indexes, add remainder to last interval endpoint. 
        index_list = np.arange(0, self.episode_length * self.n_episodes + 1, self.episode_length)
        #index_list[-1] = index_list[-1] + r # Uncomment to make uneven intervals
        
        return index_list

    def _log_step(self, info, extra_info) -> None:
        """
        logs all steps
        """
        all_info = info | extra_info

        if not self.history:
            self.history = {key:[] for key in all_info.keys()}

        for key, value in all_info.items():
            self.history[key].append(value)

    def _apply_action(self, action, home) -> Tuple[float, float]:
        """
        Applies action to battery taking into account the system configuration
        regarding export/import. 
        if importable/exportable, then the the action translates to the 
        fraction of the max output/input of the battery. Else, the action 
        translates to the fraction of the max output/input of the battery
        which does not push the home energy system into export/import. 
        """
        # requires battery._get_limits() having been called in .step().
        max_d_, max_c_ = self.battery.battery_limits
        if action > 0: # Charge
            if self.importable:
                charge_request = action * self.battery.max_input
            else:
                charge_request = action * float(max_c_) 
            _, e_flux, _ = self.battery.charge(charge_request)
        else: # Discharge 
            if self.exportable:
                discharge_request = action * self.battery.max_output
            else:
                discharge_request = action * float(max_d_) * -1
            _, e_flux, _ = self.battery.discharge(discharge_request)
            
        net = home + e_flux
        return net, e_flux

    def _apply_sq_action(self, home, max_d, max_c
                        ) -> Tuple[float, float, float]:
        if home > 0: # Must discharge        
            updated_action = max_d / self.sq_battery.max_output
            _, e_flux, _ = self.sq_battery.discharge(max_d)
        else: # Must charge
            updated_action = max_c / self.sq_battery.max_input
            _, e_flux, _ = self.sq_battery.charge(max_c)

        net = home + e_flux
        return net, e_flux, updated_action

    def _calculate_reward(self, net, export_tariff, import_tariff) -> float:
        # Calculate reward
        if net < 0:
            tariff = export_tariff if export_tariff < 0 else export_tariff * -1
            reward = net * tariff
        elif net > 0:
            tariff = import_tariff if import_tariff < 0 else import_tariff * -1
            reward = net * tariff
        else:
            reward = 0
        
        return float(reward)

    def save_results(self) -> None:

        # Save
        obs = pd.DataFrame(data=self.data_arr, columns=self.df.columns)
        info = pd.DataFrame(self.history)
        # Add back in time data. 
        if self.df_index is not None:
            df2 = pd.merge(self.df_index, obs, left_index=True, right_index=True)
            results = pd.merge(df2, info, left_index=True, right_index=True)
        else:
            results = pd.merge(obs, info, left_index=True, right_index=True)
        device = self.device_id if self.device_id != None else "dummy"
        # Add Device Column
        results.loc[:, 'device_id'] = device
        if self.print_save:
            print(results['updated_action'].value_counts())
            results.to_csv(
                self.save_path+f"/{device}_results_array_"
                f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv",
                index=False
            )
        else:
            results.to_csv(self.save_path + f"/{device}_results_array.csv", index=False)

    def _process_data(self) -> None:
        """
        Import data from the passed dataframe into a numpy array.
        all operations performed inplace for speed.
        """
        self.df_index = None
        
        # Enumerate column indx - useful for indexing later
        remove_list = ['Datetime', 'Timestamp']
        c_list = self.df.columns.tolist()
        self.obs_shape = len(c_list)
        if 'Datetime' in c_list and 'Timestamp' in c_list:
            _ = [c_list.remove(x) for x in remove_list]
            self.df_index = self.df[remove_list].reset_index()
            self.df = self.df[c_list]
        self.idx = {k: v for v, k in enumerate(self.df.columns)}
        self.data_arr = self.df[c_list].to_numpy(dtype=np.float32)

    def seed(self, seed) -> None:
        np.random.seed(seed)
        
    def calc_benchmark_rewards(self, loads, solar, export_tariff,
                               import_tariff) -> None:
            
        # Calculate step rewards
        self.no_solar_reward = self._calculate_reward(loads,
                                                      export_tariff,
                                                      import_tariff)

        self.no_battery_reward = self._calculate_reward(loads + solar,
                                                        export_tariff,
                                                        import_tariff)

        self.sq_reward = self._calculate_reward(self.sq_net,
                                                export_tariff,
                                                import_tariff)

        # Update Cumulative 
        self.no_solar_cumulative_reward += self.no_solar_reward
        self.no_battery_cumulative_reward += self.no_battery_reward
        self.sq_cumulative_reward += self.sq_reward
    
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
                               color=c, alpha=0.8)
            axs[0].set_ylim(0, 100)
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
                        f"Cumulative Reward: \${self.cumulative_reward:.2f}"
                        f"No Solar Cumulative Reward: \${self.no_solar_cumulative_reward:.2f}"
                        f"No Battery Cumulative Reward: \${self.no_battery_cumulative_reward:.2f}"
                        f"StatusQuo Cumulative Reward: \${self.statusquo_cumulative_reward:.2f}")
            fig.suptitle(
                f'Step: {self._current_tick}'
                f'Action: {ac} Net Energy: {self.net:.4f} kW'
                f"\nStep Reward: \${self.reward:.2f},"
                f"Cumulative Reward: \${self.cumulative_reward:.2f}",
                f"No Solar Cumulative Reward: \${self.no_solar_cumulative_reward:.2f}",
                f"No Battery Cumulative Reward: \${self.no_battery_cumulative_reward:.2f}",
                f"StatusQuo Cumulative Reward: \${self.statusquo_cumulative_reward:.2f}",
                y=1.1)
            self.save_rendering(f'{os.getcwd() + "/renders/"}plot')

        else:
            pass
            print(f"Step Reward: {self.reward}\n"
                  f"Action : {self.history['action'][self._current_tick]}")

    def close(self) -> None:

        plt.close()

    def save_rendering(self, filepath) -> None:
        plt.savefig(f'{filepath}_{str(self._current_tick)}.png')