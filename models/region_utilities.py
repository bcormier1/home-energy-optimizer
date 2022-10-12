import numpy as np

def encode_region(region):

    if region == 'NSW':
        return np.array([0,0,0])
    elif region == ['VIC']:
        return np.array([1,0,0])
    elif region == 'QLD':
        return np.array([0,1,0])
    elif region == 'SA':
        return np.array([0,0,1])
    else:
        raise Exception(f'Region {region} not recognised!')
