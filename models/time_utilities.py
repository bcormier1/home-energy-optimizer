"""
Need to capture the seasonality of the data which occurs 
Time functions;
- Annual Scale: Monthly seasonality, related primarily to the seasons.
- Weekly Scale: The day of the week, as a weekend or not.
- Daily Scale: The time of day, corresponding to the peak,. 

"""

import numpy as np
from datetime import datetime, time


def polar_time(self, ts, interval=5):
    """
    Takes a timestamp as input and returns the time in time
    """
    t = datetime.fromtimestamp(ts)
    current_h = t.hour
    current_m = t.minute

    n_intervals = 24 * 60 / interval

    g_min = int((current_h * 60 / interval) + (current_m // interval))
    t_sin = np.sin(2 * np.pi * g_min / n_intervals)
    t_cos = np.cos(2 * np.pi * g_min / n_intervals)
    
    return t_cos, t_sin

def encode_weekend(self, ts):
    """
    takes a timestamp as input
    returns the day of week as:
    0 weekday
    1 weekend
    """
    return 0 if ts.weekday() < 5 else 1

def encode_month(self, ts):
    """
    takes a unix timestamp as input
    returns the month as:

    """
    m_sin = np.sin(2 * np.pi * ts.month / 12)
    m_cos = np.cos(2 * np.pi * ts.month / 12)

    return m_cos, m_sin