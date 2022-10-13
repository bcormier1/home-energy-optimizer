from gym.envs.registration import register

register(
    id='gym_homer/test_env-v0',
    entry_point='gym_homer.envs:test_env'
)