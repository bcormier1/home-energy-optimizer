import sys
import os
import pandas as pd
import numpy as np
import json
import argparse
import torch
import datetime
import wandb

#set working dir and path. 
parent_dir = os.getcwd()
path = os.path.dirname(parent_dir)
sys.path.append(path)

from data.data_utils import (
    DataLoader, 
    ConfigParser
)

import gym
from gym import spaces, wrappers
from gym_homer.envs.homer_env import HomerEnv

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import WandbLogger
from tianshou.data import (
    Collector, 
    VectorReplayBuffer, 
    AsyncCollector
)
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

import warnings
warnings.filterwarnings('ignore')

def main(args):
    
    # Load settings and create config class. 
    settings = load_settings(args.file)
    config = ConfigParser(settings)
    for k,v in settings.items():
        setattr(config, k, v)
    
    # Set up logs
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    config.algo_name = "ppo"
    log_name = os.path.join(config.task, config.algo_name, now)
    log_path = os.path.join(config.log_path, log_name) 
    
    # Set up directoriesfor logging
    exists = os.path.exists(log_path)
    if not exists:
        os.makedirs(log_path)
        print(f"New directory was created at '{log_path}'")
    print(f"Logging to '{log_path}'")
    setattr(config, "result_path", log_path)
    
    # Initialise the wandb logger
    logger = WandbLogger(
        save_interval=1000,
        #run_id=settings.get('run_id',None),
        #name=settings.get('run_name',None),
        project="RL_project", 
        entity="w266_wra",
        config=settings
        )
    writer = SummaryWriter(config.log_path)
    writer.add_text("args", str(config))
    logger.load(writer)
    
    if args.sweep:
        print('Running sweep!')
        # Hacky wandb sweep integration. 
        wandb_config = logger.wandb_run.config
        for item in wandb_config.items():
            wandb_key = item[0]
            wandb_val = item[1]
            if wandb_key in settings.keys():
                settings[wandb_key] = wandb_val
            else:
                settings.update({wandb_key: wandb_val})
            print(f"Sweep Argument Check: {wandb_key}: {settings[wandb_key]}")
    
    policy = None
    test_collector = None
    
    if config.do_train:
        print('Running Taining')
        policy, test_collector = train_agent(config, logger, log_path)

    # Close wandb logging
    logger.wandb_run.finish()

    if config.do_eval:
        # Do eval
        if policy is None or test_collector is None:
            print('No Policy or Collector loaded! Skipping evaluation')
        else:
            print('Running evaluation')
            do_eval(config, policy, test_collector)
    else:
        print('!! Skipping evaluation loop !!')

    print('Run completed successfully')
    
def load_settings(file):
    print(file)
    with open(file, 'r') as f:
        settings = json.load(f)
    return settings

def train_agent(config, logger, log_path):
    
    # Check hardware device:
    log_path = log_path
    device = args.device
    print(f'Using {device}!')
    
    # Make environments:
    print(f"Loading environments.")
    train_envs, _ = load_homer_env(config, 'train')  
    print("Loaded train Enironments")
    test_envs, _ = load_homer_env(config, 'validation')
    print("Loaded validation environments")

    
    # Define Network
    env, _ = load_homer_env(config, 'train', True) 
    if config.parameterised_mlp:
        print(f"Loading parameterised network with {config.n_hidden_layers}"
              f" hidden layers, each with {config.hidden_layer_dims} neurons")
        hidden_sizes = [
            config.hidden_layer_dims for dim in range(config.n_hidden_layers)
        ]
    else:
        print(f"Loading network with dimension{config.hidden_sizes}")
        hidden_sizes = config.hidden_sizes
    
    obs_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    
    net = Net(
        obs_shape, 
        hidden_sizes=hidden_sizes, 
        device=device
    )
    
    print(f"Environment Action space: {env.action_space}")
    print(f"Environment Action shape: {action_shape}")
    print(f"Environment Observation shape: {env.observation_space.shape}")
    
    actor = Actor(net, action_shape, device=device).to(device)
    critic = Critic(net, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)
    
    # optimizer of the actor and the critic
    lr_optimizer = config.lr_optimizer
    optim = torch.optim.Adam(actor_critic.parameters(), lr=lr_optimizer)
    
    lr_scheduler = None
    if config.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            config.steps_per_epoch / config.steps_per_collect
            ) * config.n_max_epochs
        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    # Since environment action space is discrete 
    def dist(p):
        return torch.distributions.Categorical(logits=p)
    
    policy = PPOPolicy(
        actor, 
        critic, 
        optim, 
        dist,
        discount_factor=config.gamma,
        gae_lambda=config.gae_lambda,
        max_grad_norm=config.max_grad_norm,
        vf_coef=config.vf_coef,
        ent_coef=config.ent_coef,
        reward_normalization=config.rew_norm,
        action_scaling=config.action_scaling,
        lr_scheduler=lr_scheduler, 
        action_space=env.action_space, 
        deterministic_eval=True,
        eps_clip=config.eps_clip,
        value_clip=config.value_clip,
        dual_clip=config.dual_clip,
        advantage_normalization=config.norm_adv,
        recompute_advantage=config.recompute_adv,
    ).to(device)

    if config.resume_path:
        print(f"Resuming from path {config.resume_path}")
        policy.load_state_dict(
            torch.load(config.resume_path, map_location=args.device))
        print("Loaded agent from: ", config.resume_path)
    else:
        print("Commencing new run with untrained agent.")
    
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        config.replay_buffer_collector,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
    #    save_only_last_obs=True,
    #    stack_num=config.frames_stack,
    )
    print('Buffer Loaded')    
    train_collector = Collector(
        policy, 
        train_envs, 
        buffer, 
        exploration_noise=True
    )
    test_collector = Collector(
        policy, 
        test_envs,
        exploration_noise=True
    )
    
    print(f"Testing train_collector, filling replay buffer")
    train_collector.collect(
        n_step=config.batch_size * config.training_num
    )
    print("Done")
    def save_best_fn(policy):
        best_pth = os.path.join(log_path, "best_policy.pth")
        torch.save(policy.state_dict(), best_pth)
        print(f"Best policy saved to {best_pth}")

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save({"model": policy.state_dict()}, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
        return ckpt_path
    
    print("Running training")
    # see https://tianshou.readthedocs.io/en/master/api/tianshou.trainer.html
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=config.n_max_epochs,
        step_per_epoch=config.steps_per_epoch,
        repeat_per_collect=config.repeat_per_collector,
        episode_per_test=config.test_num,
        batch_size=config.batch_size,
        step_per_collect= config.steps_per_collect,
        stop_fn=lambda mean_reward: mean_reward >= config.reward_stop,
        logger=logger,
        verbose=True,
        save_best_fn=save_best_fn,
        resume_from_log=config.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
    )
    print("Training complete.\nSummary:\n")
    print(result)
    return policy, test_collector

def do_eval(config, policy, test_collector):
    
    # Create test envs
    test_envs, device_list = load_homer_env(config, 'test')
    print("Loaded test environments")
    
    # Load test collector with new test envs. 
    
    if config.save_test_data:
        print(f"Logging Output to {config.result_path}")
    else:
        print("Not logging.")
    
    df_list = []
    for repeat in range(config.eval_n_repeats):
        # Load test collector with new test envs. 
        test_collector = Collector(
            policy, 
            test_envs,
            exploration_noise=True
        )
        policy.eval()
        result = test_collector.collect(
            n_episode=test_collector.env_num, 
            render=False
        )
        print(
            f"Repeat: {repeat} "
            f"Final reward: {result['rews'].mean():.4f},"
            f" length: {result['lens'].mean():.4f}\n"
            f"{result}"
        )
        
        # Build dict:
        if config.save_test_data:
            result_dict = {
                "repeat": np.ones(result["n/ep"])*repeat,
                "deviceid": np.array(device_list)[result['idxs']],
                "steps": result["lens"],
                "reward": result["rews"]
            }
            df_list.append(pd.DataFrame(result_dict))
 
    if config.save_test_data:
        summary_dict = pd.concat(df_list)
        pth = config.result_path+"/aggregated_result_summary.csv"
        print(f"saving to {pth}")
        summary_dict.to_csv(pth, index=False)

def load_homer_env(config, data_subset, example=False):

    file_loader = DataLoader(config, data_subset)
    
    # If test, we need to supply the save data
    if data_subset == 'test' and config.save_test_data:
        history = config.save_test_data
        result_path = config.result_path 
    else: #Don't save. 
        history = False
        result_path = ""

    if config.pricing_env == 'dummy':
        device_list = ['dummy' for _ in range(config.n_dummy_envs)]
        if example:
            envs =  HomerEnv(
                data=file_loader.load_dummy_data(), 
                start_soc=config.start_soc, 
                discrete=config.discrete_env,
                capacity=10,
                charge_rate=12,
                action_intervals=config.action_intervals
            )   
        else:
            envs = SubprocVectorEnv(
                [lambda: HomerEnv(
                    data=file_loader.load_dummy_data(), 
                    start_soc=config.start_soc, 
                    discrete=config.discrete_env,
                    capacity=10,
                    charge_rate=12,
                    action_intervals=config.action_intervals,
                    save_history=history,
                    save_path=result_path) 
                for i in range(config.n_dummy_envs)]
            )
    elif config.pricing_env == 'simple':
        device_list = file_loader.device_list
        if example:
            # Load a single example to get env dimensions.
            envs =  HomerEnv(
                data=file_loader.load_device(device_list[0]), 
                start_soc=config.start_soc, 
                discrete=config.discrete_env,
                charge_rate=config.charge_rate,
                action_intervals=config.action_intervals
            ) 
        else:
            n_devices = file_loader.n_devices
            env_list = [
                lambda i=i: HomerEnv(
                    data=file_loader.load_device(device_list[i], 
                                                 truncate=config.truncate, 
                                                 max_days=config.n_days,
                                                 val_offset=config.val_offset), 
                    start_soc=config.start_soc, 
                    discrete=config.discrete_env,
                    charge_rate=config.charge_rate,
                    action_intervals=config.action_intervals,
                    save_history=history,
                    save_path=result_path,
                    device_id=device_list[i]
                ) for i in range(n_devices)
            ]            
            envs = SubprocVectorEnv(env_list)
            print(f"Sucessfully loaded {n_devices} environments")
    
    elif config.pricing_env == 'complex':
        raise NotImplementedError    
    else:
        print(f"Invalid value for pricing_env: {config.pricing_env}")
        raise Exception

    return envs, device_list

if __name__ == "__main__":
    
    # Parse Arguments. 
    parser = argparse.ArgumentParser(description="Train an Agent")
    
    parser.add_argument(
        '--file', help='json file with all arguments', type=str)
    parser.add_argument(
        '--debug', help='Ture/False for debug mode', 
        action=argparse.BooleanOptionalAction)
    parser.add_argument(
        '--sweep', help= 'Ture/False for WandB sweep', 
        action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--device", type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    main(args)