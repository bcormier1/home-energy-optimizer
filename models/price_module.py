from datetime import datetime, time
import random

class reward:
    
    
    def __init__(self):
        """
        TODO: Complete Docs 
        Initialize the reward module. 
        
        args: 
        
        """
        #self.region = region
        #self.tariff_type = tariff_type
        #self.export_tariff = export_tariff
        
        self.TOU = {
            "VIC":{
                "peak": 0.3021, 
                "shoulder": 0.1735, 
                "off-peak": 0.1735, 
                'solar':0.052, 
                'connection': 1.1623
                },
            "NSW":{
                "peak": 0.6234, 
                "shoulder": 0.2839, 
                "off-peak": 0.1642, 
                'solar':0.050, 
                'connection': 1.0480
                },
            "QLD":{
                "peak": 0.3566, 
                "shoulder": 0.2500, 
                "off-peak": 0.2032, 
                'solar':0.050, 
                'connection': 1.2180
                },
            "SA":{
                "peak": 0.4435, 
                "shoulder": 0.2178, 
                "off-peak": 0.2633, 
                'solar':0.050, 
                'connection': 0.9514
                }
        }
        
    def get_tariff_rate(self, current_timestamp, region):
        """
        Returns the Tariff Rate for a given region and timestamp, based on the 
        ToU dict.
        
        inputs:
            current_timestamp (datetime.datetime): The time stamp including 
                date and time.
            region (str)
            
        return (float): The rate in $/kWh
        
        """
    
        current_time = current_timestamp.time()
        
        if region == 'VIC':
            # Only peak or shoulder/off-peak
            if self.time_between(time(15,0), time(21,0), current_time):
                return self.TOU['VIC']['peak']
            else: #shoulder/off-peak - either doesn't matter.
                return self.TOU['VIC']['shoulder']
            
        elif region == 'SA':
            # Determine whether peak, shoulder or off-peak
            if (
                self.time_between(time(6,0), time(10,0), current_time) or 
                self.time_between(time(15,0), time(1,0), current_time)
                ):                
                return self.TOU['SA']['peak']
            elif self.time_between(time(1,0), time(6,0), current_time):
                return self.TOU['SA']['off-peak']
            else: #shoulder
                return self.TOU['SA']['shoulder']
        
        else:
            # Either NSW or QLD which have weekday dependent peaks
            dow = current_timestamp.weekday()
            if region == 'QLD':
                if self.time_between(time(22,0), time(7,0), current_time):
                    return self.TOU['QLD']['off-peak']
                elif (
                    dow < 5 and 
                    self.time_between(time(16,0), time(20,0), current_time)
                    ):
                    return self.TOU['QLD']['peak']
                else:
                    return self.TOU['QLD']['shoulder']
            
            elif region == 'NSW':
                # NSW only has a peak in winter and autum
                month = current_timestamp.month
                if self.time_between(time(22,0), time(7,0), current_time):
                    return self.TOU['NSW']['off-peak']
                # If it's a weekday, in peak time during the months of Summer 
                # or Winter. 
                elif (
                    dow < 5 and 
                    self.time_between(time(14,0), time(20,0), current_time) and 
                    month in [1,2,3,6,7,8,12]
                    ):
                    return self.TOU['NSW']['peak']
                else:
                    return self.TOU['NSW']['shoulder']             
            else:
                print('unknown region!')
                
            
    def get_solar_fit(self, region):
        """
        Returns the rate of reward for solar power exported in $/kWh by region
        region (str): The state as string
        """
        return self.TOU[f'{region}']['solar']
    
    def get_connection_fee(self, region):
        """
        Returns the daily connection fee in $/day
        region (str): The state as string
        
        """
        return self.TOU[f'{region}']['connection']           
    
    def time_between(self, begin_time, end_time, check_time):
        """
        Checks if a time is between two times. 
        from https://stackoverflow.com/questions/10048249/how-do-i-determine-if-current-time-is-within-a-specified-range-using-pythons-da
        """

        check_time = check_time
        if begin_time < end_time:
            return check_time >= begin_time and check_time <= end_time
        else: # crosses midnight
            return check_time >= begin_time or check_time <= end_time