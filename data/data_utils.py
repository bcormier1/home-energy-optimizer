import pandas as pd
import os

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
        self.load_data(config)

    def load_data(self, config):

        if config.pricing_env == 'dummy':
            return self.load_dummy_data()
            
        elif config.pricing_env == 'simple':   
            self.device_list = self.load_device_list(self)
            if config.max_devices <= len(self.device_list):
                self.n_devices = config.max_devices
            else:
                self.n_devices = len(self.device_list)
            print(f"Found {self.n_devices} devices")
        
        elif config.pricing_env == 'complex':
            print("Complex Environment not implemented yet.")
            raise NotImplementedError

    def load_device_list(self, config):
        device_dir = self.path+f"/data/{self.dataset_type}_pricing/device_list.csv"
        try:
            device_list = pd.read_csv(device_dir)['device_id'].tolist()
        except:
            print("Could not load device list")
        
        return device_list
        
    
    def load_device(self, device):
        
        if self.subset == "validation":
            self.subset = "train"
        data_dir = self.path+f"/data/{self.dataset_type}_pricing/"+self.subset
        fname = f"/{device}_simple_{self.subset}.parquet"
        data = pd.read_parquet(data_dir+fname).fillna(0)
        print(f"loaded {fname}")
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
            
    
     

            
