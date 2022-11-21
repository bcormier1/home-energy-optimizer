# How to Run an Experiment

## Quick Start

To install the environments create a new virtual enviroment, run

```bash
pip install -r requirements.txt
```

for cuda_enabled environments run
```bash
pip install -r requirements_cuda.txt
```

To then run an experiment, ensure the directory tree is set up properly as below. Then cd into the experiment directory and run the following command:

```bash
python ppo_experiment_v1.py --file config/settings_template.json 
```

## Run a sweep 

Perform the following steps from the terminal within the `experiments` directory.

1. Log into weights and biases:

```bash
wandb login
```
2. Intialise the sweep agent.

Make sure to provide the correct project name (below we use `RL_project`)corresponding to your project name in W&B, and config .yaml file. 
```bash
wandb sweep --project RL_project config/sweep.yaml
```

3. Run the sweep. 

The previous command will set up the sweep directroy and agent, and provide a commmand in the terminal output to run the sweep. It will look something like below:

```bash
wandb agent w266_wra/RL_project/<sweep_ID>
```
where the `sweep_ID` is the generated sweep ID. 

## Directory Set up

The data for the simulations should be stored as parquet files for the simple and complex pricing simulations. For the dummy data, the same text file as a csv is stores in both directories. 

```bash
home-energy-optimizer
├── data
│   ├── data_utils.py
│   ├── complex_pricing
│   │   ├── test
│   │   └── train
│   ├── dummy_data
│   │   ├── test
│   │   └── train
│   └── simple_pricing
│       ├── test
│       └── train
├── experiments
│   ├── config
│   │   ├── settings_template.json
│   │   └── templates
│   ├── logs
│   │   ├── Homer_DummyEnv
│   │   └── Homer_SimpleEnv
│   ├── results
│   │   └── benchmarks
│   ├── wandb
│   └── ...
├── gym_homer
│   ├── envs
│   │   ├── homer_env.py
│   │   └── ...
│   └── setup.py
...
```