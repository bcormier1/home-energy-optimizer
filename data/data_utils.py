import pandas as pd
import os
import numpy as np

class DataLoader():
    """
    Class to handle the loading of the dataset into the environment.
    """

    def __init__(self, config, data_subset):
        """
        config (dict): the parsed settings.py file.
        data_subset (str): string indicating whether loading the train/
                            validation or train dataset.
        """

        self.device_list = []
        self.n_devices = 0
        self.subset = data_subset
        self.dataset_type = config.pricing_env
        self.path = os.path.dirname(os.getcwd())
        self.episodes = False if config.episode_length == 0 else True
        self.train_is_val = config.train_is_val
        self.load_data(config)
        self.debug = True if self.dataset_type == 'debug' else False

    def load_data(self, config):

        if config.pricing_env == 'dummy':
            return self.load_dummy_data()
            
        elif config.pricing_env in ['debug','simple','complex']:   
            self.device_list = self.load_device_list(self)
            if (config.max_devices <= len(self.device_list) and 
                config.max_devices > 0):
                self.n_devices = config.max_devices
            else:
                print('Invalid input, defaulting to number of found devices')
                self.n_devices = len(self.device_list)
            print(f"Found {self.n_devices} devices")

    def load_device_list(self, config):
        if self.dataset_type == 'debug':
            device_dir = self.path+f"/data/debug/device_list.csv"
        else:
            device_dir = self.path+f"/data/{self.dataset_type}_pricing/device_list.csv"
        try:
            print(device_dir)
            device_list = pd.read_csv(device_dir)['device_id'].tolist()
        except:
            print("Could not load device list")
        return device_list
        
    
    def load_device(self, device, truncate=False, max_days=None, 
                    val_offset=10):
        """
        Loads the data for a particular device assuming that device has it's own
        file according to format:
        <device_name>_simple_<subset>.parquet
        returns a data frame 
        """

        # Handle the validation data, which can be truncated but uses the same
        # Data directory as the train set. 
        was_valid = False
        offset_validation = None
        if self.subset == "validation" and not self.debug:
            was_valid = True
            offset_validation = val_offset * 12 * 24
            self.subset = "train"

        # Load file
        if self.debug:
            data_dir = self.path+f"/data/{self.dataset_type}/"+self.subset+"/"
        else:
            data_dir = self.path+f"/data/{self.dataset_type}_pricing/"+self.subset+"/"
        fname = f"{device}_{self.dataset_type}_{self.subset}.parquet"
        data = pd.read_parquet(data_dir+fname).fillna(0)
        
        # Truncate the dataset 
        if truncate and max_days is not None:
            steps = max_days * 24 * 12
            if was_valid == False:
                if len(data) <= steps:
                    print("Warning: not enough data! Defaulting to 7 days")
                    steps = 24 * 12 * 7
                data = data.head(steps)
        
        # Offset the validation set. 
        if offset_validation is not None:
            # offset the first few values to step the dataset forward in time.
            data = data.loc[offset_validation:offset_validation*2,].copy()
        
        print(f"loaded {len(data)} steps from {fname}")
        return data
    
    def _load_device(self, device, val_offset=0,
                    n_days_train=-1, n_days_val=-1, n_days_test=-1):
        """
        Loads the data for a particular device assuming that device has it's own
        file according to format:
        <device_name>_simple_<subset>.parquet
        returns a data frame 
        """
        # Load file

        if self.debug:
            data_dir = self.path+f"/data/{self.dataset_type}/"+self.subset+"/"
            fname = f"{device}_{self.dataset_type}_{self.subset}.parquet"
        else:
            if self.train_is_val and self.subset == 'validation':
                data_dir = self.path+f"/data/{self.dataset_type}_pricing/train/"
                fname = f"{device}_{self.dataset_type}_train.parquet"
            else:
                data_dir = self.path+f"/data/{self.dataset_type}_pricing/"+self.subset+"/"
                fname = f"{device}_{self.dataset_type}_{self.subset}.parquet"
        
        data = pd.read_parquet(data_dir+fname).fillna(0)
                
        if self.subset == 'train':
            n = n_days_train * 288
        elif self.subset == 'validation':
            n = (n_days_val + val_offset) * 288
            n = abs(n) * -1 if n_days_val == -1 else abs(n)
        elif self.subset == 'test':
            n = n_days_test * 288
        # Truncate the dataset       
        if n <= len(data) and n > 0:
            if self.subset == 'train' or self.subset == 'test':
                data = data.head(n)
            else:
                data = data.tail(n).loc[val_offset:,].copy()
        else:
            if self.subset == 'validation':
                data = data.loc[val_offset:,].copy()
        
        print(f"loaded {len(data)} steps from {fname}")
        return data

    def load_dummy_data(self):
        
        if self.subset == "validation":
            self.subset = "test"
        
        data_dir = self.path+"/data/dummy_data/"+self.subset
        fname = "/test_env_data.csv"

        return pd.read_csv(data_dir+fname, index_col=False).fillna(0)

    
class ConfigParser():
    
    def __init__(self, config_dict):
        """
        Simple class to mimic the argparse package, whereby the keys
        are accessed as attributes.
        """
        self.config_dict = config_dict    
    
     

            
