import numpy as np
import gym
from gym import spaces
import battery_sim
from battery_sim import BatteryOverdrawError, BatteryOverflowError


class HomerEnv(gym.Env):
    def __init__(self):
        self.battery = battery_sim.battery()
        self.timestamp_iter = 0
        self.total_energy = 0
        
        
        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-1, 1, shape=(3,), dtype=int)
        
    def calc_total_energy(self, obs):
        print(f'obs[0]: {obs[0]}, obs[1]: {obs[1]}')
        self.total_energy  = (obs[0] + self.battery.max_output) - obs[1] # calcualate total energy for the current timestamp here later
        print(f'total_energy: {self.total_energy}')
        
    def _get_obs(self):
        """
        Return state of the world.  This is weather and market forcasts, and battery and energy info.
        For now return zeros
        """
        obs = np.array([0,0,0]) # this will be real data here
        self.calc_total_energy(obs)
        return obs
    
    def _get_info(self):
        return {'info': None}
    
    def sell_energy(self,energy):
        price = 1 # this will get actual price later
        print(f'sell: {price*energy}')
        return float(price*energy)
    
    def buy_energy(self,energy):
        price = 1 # this will get actual price later
        print(f'buy: {-1*price*energy}')
        return float(-1*price*energy)
    
    def _apply_action_to_battery(self, energy):
        if energy > self.battery.max_output:
            try:
                self.battery.charge(energy)
            except BatteryOverflowError:
                    pass
        else:
            try:
                self.battery.discharge(self.battery.max_output - energy)
            except BatteryOverdrawError:
                pass
    
    def _apply_action(self,action):
        if self.total_energy < 0:
            return self.buy_energy(self.total_energy)
        else:
            self._apply_action_to_battery(action*self.total_energy)
            print(f'energy prior to sell: {self.total_energy}')
            return self.sell_energy((1-action)*self.total_energy)
        
    
    def _check_final_timestamp(self):
        if self.timestamp_iter >= 100: # this will check for the last timestamp in the series later
            return True
        return False
    
    def reset(self):
        self.battery = battery_sim.battery()
        self.timestamp_iter = 0
        return self._get_obs()
    
    def step(self, action):
        """
        Return (observation, reward, done, info)
        """        
        self.timestamp_iter += 1
        
        observation = self._get_obs() # this needs to run before _apply_action so that the total energy is properly calculated
        
        reward = self._apply_action(action)
        print(f'reward: {reward}')
        done = self._check_final_timestamp()
        
        
        info = self._get_info()
        
        return observation, reward, done, info
    
    def close(self):
        pass

        
        
        