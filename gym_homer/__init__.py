from gym.envs.registration import register

register(
    id='gym_homer/HomerEnv-v0',
    entry_point='gym_homer.envs:HomerEnv'
)