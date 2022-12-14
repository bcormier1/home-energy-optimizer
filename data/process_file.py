import pandas as pd
import os

def main():
    i = 0
    device_list = pd.read_csv("../../device_list.csv")['device_id'].tolist()
    print(len(device_list))
    
    for device in device_list:

        df = pd.read_parquet(f'{device}_complex_test.parquet')
        df.loc[df['Home Consumption'] < 0, 'Home Consumption'] = 0
        df.loc[:, 'loads'] = df['Home Consumption']
        df.loc[:, 'import_tariff'] = df['rrp_x'] * -1 / 1000 / 1000 * 100
        df.loc[:, 'export_tariff'] = df['rrp_x'] / 1000 / 1000 * 100
        df.loc[:,"max_d"] = 0
        df.loc[:,"max_c"] = 0
        df.loc[:,"soc"] = 0
        filter_cols = [
            'Datetime',
            'Timestamp',
            'time_x',
            'time_y',
            'weekday',
            'month_x',
            'month_y',
            'region_1',
            'region_2',
            'region_3',
            'solar',
            'loads',
            'import_tariff',
            'export_tariff',
            'max_d',
            'max_c',
            'soc'
        ]
        df[filter_cols].to_parquet(f'../{device}_complex_test.parquet')
        print(i)
        i +=1


if __name__ == "__main__":
    
    main() 