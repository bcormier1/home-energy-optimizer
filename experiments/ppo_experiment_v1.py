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
from torch import nn

from tianshou.utils import WandbLogger
from tianshou.data import (
    Collector, 
    VectorReplayBuffer, 
    PrioritizedVectorReplayBuffer
)
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy, ICMPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import (
    Actor, 
    Critic, 
    IntrinsicCuriosityModule)

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
        save_interval=1,
        project="homer_dev", 
        entity="w266_wra",
        train_interval=1,
        update_interval=1,
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
        # Update config. 
        for k,v in settings.items():
            setattr(config, k, v)
    
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
    
    # Seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_envs.seed(config.seed)
    test_envs.seed(config.seed)
    
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
        action_shape,
        hidden_sizes=hidden_sizes, 
        device=device
    )
    
    print(f"Environment Action space: {env.action_space}")
    print(f"Environment Action shape: {action_shape}")
    print(f"Environment Observation shape: {env.observation_space.shape}")
    
    if torch.cuda.is_available() and config.data_parallel:
        actor = DataParallelNet(
            Actor(net, action_shape, device=None).to(device)
        )
        critic = DataParallelNet(Critic(net, device=None).to(device))
    else:
        actor = Actor(net, action_shape, device=device).to(device)
        critic = Critic(net, device=device).to(device)
    
    actor_critic = ActorCritic(actor, critic)
    
    # orthogonal initialization required for PPO 
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    # optimizer of the actor and the critic
    lr_optimizer = config.lr_optimizer
    optim = torch.optim.Adam(actor_critic.parameters(), lr=lr_optimizer)
    
    lr_scheduler = None
    if config.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            config.steps_per_epoch / config.episode_length#config.steps_per_collect
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
        eps_clip=config.eps_clip,
        dual_clip=config.dual_clip,
        value_clip=config.value_clip,
        advantage_normalization=config.norm_adv,
        recompute_advantage=config.recompute_adv,
        vf_coef=config.vf_coef,
        ent_coef=config.ent_coef,
        max_grad_norm=config.max_grad_norm,
        gae_lambda=config.gae_lambda,
        reward_normalization=config.rew_norm,
        max_batchsize = config.max_batchsize_policy,
        action_scaling=config.action_scaling,
        action_bound_method='clip',
        action_space=env.action_space,
        lr_scheduler=lr_scheduler,  
        deterministic_eval=False
    ).to(device)
    
    if config.icm: # Use the intrinsic Curiosity module
        print("Loading Intrinsic Curiousity Module")
        feature_net = Net(obs_shape,
                          action_shape,
                          hidden_sizes=hidden_sizes, 
                          device=device)
        output_dim = int(np.prod(action_shape)) * 1 
        feature_net.net = nn.Sequential(
                feature_net.model, 
                nn.Linear(output_dim, output_dim),
                nn.ReLU(inplace=True)
            )
        action_dim = np.prod(action_shape)
        feature_dim = output_dim
        icm_net = IntrinsicCuriosityModule(
            feature_net.net,
            feature_dim,
            action_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        )
        icm_optim = torch.optim.Adam(icm_net.parameters(), lr=lr_optimizer)
        policy = ICMPolicy(
            policy, 
            icm_net, 
            icm_optim,  
            config.icm_lr_scale, 
            config.icm_reward_scale,
            config.icm_fwd_loss
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
    if config.prioritized_replay_buffer:
        buffer = PrioritizedVectorReplayBuffer(
            config.replay_buffer_collector,
            buffer_num=len(train_envs),
            alpha=config.alpha,
            beta=config.beta,
        )
    else:
        buffer = VectorReplayBuffer(
            total_size=config.replay_buffer_collector,
            buffer_num=len(train_envs)
        )
    print(f'Buffer Loaded with length {len(buffer)}')    
    train_collector = Collector(
        policy, 
        train_envs, 
        buffer
    )
    test_collector = Collector(
        policy, 
        test_envs
    )
    
    print(f"Running train_collector, filling replay buffer")
    collector_output = train_collector.collect(
        n_step=config.batch_size * config.training_num
    )
    print(f'{len(train_envs)} vectorised buffers loaded, each '
          f'with {len(buffer)} steps\nSampled action summary:\n')    

    unique, counts = np.unique(buffer.act, return_counts=True)
    for i in range(len(unique)):
        print(f"Action: {unique[i]:.4f}, Counts: {counts[i]}")
    
    c_keys=['n/ep', 'n/st', 'rews', 'lens', 'rew', 'len', 'rew_std', 'len_std']
    print('Collector Stats')
    for key in c_keys:
        print(f'{key}: {collector_output[key]}')

    print("\nDone")
    def save_best_fn(policy):
        best_pth = os.path.join(log_path, "best_policy.pth")
        torch.save(policy.state_dict(), best_pth)
        print(f"Best policy saved to {best_pth}")

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save({"model": policy.state_dict(), "optim": optim.state_dict()}, 
                    ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
        return ckpt_path
    
    b1 = config.steps_per_collect == None
    b2 = config.episode_per_collect == None
    if b1 != b2:
        pass
    else:
        raise Exception (
            "Both steps_per_collect or episode_per_collect may not be set, set one to 'null'. "
        )

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
        episode_per_collect=config.episode_per_collect,
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
    
    history = config.save_test_data
    result_path = config.result_path 

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
    
    elif config.pricing_env == 'debug' or 'simple':
        device_list = file_loader.device_list
        if example:
            # Load a single example to get env dimensions.
            envs =  HomerEnv(
                data=file_loader._load_device(device_list[0]), 
                start_soc=config.start_soc, 
                discrete=config.discrete_env,
                charge_rate=config.charge_rate,
                action_intervals=config.action_intervals,
                episode_length=config.episode_length,
            ) 
        else:
            n_devices = file_loader.n_devices
            env_list = [
                lambda i=i: HomerEnv(
                    data=file_loader._load_device(device_list[i], 
                                                 val_offset=config.val_offset,
                                                 n_days_train=config.n_days_train,
                                                 n_days_val=config.n_days_val,
                                                 n_days_test=config.n_days_val), 
                    start_soc=config.start_soc, 
                    discrete=config.discrete_env,
                    charge_rate=config.charge_rate,
                    action_intervals=config.action_intervals,
                    exportable=config.exportable,
                    importable=config.importable,
                    benchmarks=config.benchmarks,
                    episode_length=config.episode_length,
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