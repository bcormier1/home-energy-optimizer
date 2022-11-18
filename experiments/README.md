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

## Directory Set up

The data for the simulations should be stored as parquet files for the simple and complex pricing simulations. For the dummy data, the same text file as a csv is stores in both directories. 

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