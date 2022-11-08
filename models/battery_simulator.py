from typing import Tuple
import numpy as np

class BatteryFullError(Exception):
    pass

class BatteryEmptyError(Exception):
    pass

class BatteryOverflowError(Exception):
    pass

class BatteryOverdrawError(Exception):
    pass

class BatteryNotexportableError(Exception):
    pass

class BatterySourceMissingError(Exception):
    pass

class battery:
    capacity: float
    exportable: bool
    charge_rate: float
    discharge_rate: float
    avl_energy: float
    round_trip_efficiency: float # Percentage of energy recoverable from each unit of input energy
    total_energy_flow: float # the highest allowable charge/dischare rate in kW
    trip_count: int # The total number of full cycles (Charge and discharge) of the battery. 
    capacity_degredation_rate: float # The rate at which the capacity is decrememnted per trip in kWh/trip
    previous_trip_count: int # The starting counter for the trip count.
    interval: int # The action interval in minutes. 
    battery_limits: tuple # Upper and lower bounds for battery operation in kWh. 

    def __init__(self, capacity=14, start_soc = 'full', importable=False, 
                 exportable = False, charge_rate=5, discharge_rate=5, 
                 round_trip_efficiency=0.9, solar_present=True) -> None:
        """
        Initialize battery specs.
        `capacity` is the total current energy capacity of the battery in kWh        
        `exportable` represents if the battery can send it's power back to 
            the optimizer for use in another purpose.  
            For example a car cannot do this while a wall battery can.
            
        `importable` represents if the battery can pull power from the grid in order to charge. If not,
            it can only be charged by excecss solar. Must be True if solar_present is false. 
        `charge_rate` is the rate of battery charging in kW
        `discharge_rate` is the rate of battery discharge (if applicable) in kW
        
        `round_trip_efficiency` is the Percentage of energy recoverable from each unit of input energy
        
        `solar_present` indicates whether a solar system is connected (True) or not (False)
               
        """
        
        self.capacity = capacity
        self.exportable = exportable
        self.importable = importable
        self.max_charge_rate = charge_rate 
        self.max_discharge_rate = discharge_rate
        self.battery_limits = ()
        self.start_soc = start_soc
        
        # Connected to solar (True) or not (False)
        self.solar = solar_present
        
        # Check battery can be charged somehow
        if not self.importable and not self.solar:
            raise BatterySourceMissingError         
       
        # Action interval length in minutes
        self.interval = 5
        self.eps = 1e-6 # Threshold for floating point error 
        
        # Define the max input/output energy for the given interval
        self.max_input = self.max_charge_rate / 60 * self.interval
        self.max_output = self.max_discharge_rate / 60 * self.interval
        
        # Variables used for capacity decrementing
        self.total_energy_flow = 0
        self.trip_count = 0
        # Set capacity degredation in kWh/trip for Tesla Powerwall2
        self.capacity_degredation_rate = 0.00025 
        self.previous_trip_count = 0

        # Set starting capacity.
        if self.start_soc == 'full':
            self.avl_energy=capacity * 1
            self.soc = self.avl_energy / self.capacity
        elif self.start_soc == 'empty':
            self.avl_energy=capacity * 0
            self.soc = self.avl_energy / self.capacity
        elif self.start_soc == 'random':
            self.avl_energy=capacity * np.random.random_sample()
            self.soc = self.avl_energy / self.capacity
        elif self.start_soc == 'half':
            self.avl_energy=capacity * 0.5
            self.soc = self.avl_energy / self.capacity
        
        # Assume same rate for charge and discharge. 
        self.discharge_loss = round_trip_efficiency ** 0.5
        self.charge_loss = round_trip_efficiency ** 0.5
        
        # Operation History
        self.status_dict = {
            "timestamp": [],
            "Total Capacity": [],
            "Available Capcity": [],
            "Total Energy Flow": [],
            "Trip Count": [],
            "Max Discharge": [],
            "Max Charge": [],
            "Discharge Limit": [],
            "Charge Limit": [],
            "Setpoint": []
        }
    
  
    def charge(self, input_energy) -> Tuple[float, float]:        
        """
        Charge the battery with the given input energy amount in kWh.
        The actual amount the capacity changes is less than the input
        energy since there are efficiency losses
        
        input:
            input_energy (float): The energy to be input into the battery in 
                kWh, before charge losses.
        
        return:
            net_input (float): The amount of energy added to the battery 
                capacity via charging, taking into account charging losses.
            avl_energy (float): The updated current capacity of the battery 
                available for use in kWh after  charging is completed.
        """       
        if input_energy < 0:
            raise ValueError(f'Non-positive input! {input_energy}')
        elif input_energy == 0: #Skip
            return 0, self.avl_energy
        
        available_capacity = self.get_charge_potential()
        
        if abs(input_energy - available_capacity) < self.eps:
            input_energy = available_capacity        

        if input_energy * self.charge_loss > available_capacity:
            # Need to clip the energy input to max allowable.
            net_input = available_capacity
        else:
            # Calculate the actual amount decremented
            net_input = input_energy * self.charge_loss
        # Update capacity and decrement capacity
        self.avl_energy += net_input
        self.update_capacity_degredation(net_input)
        self.soc = self.avl_energy / self.capacity
            
        return net_input, self.avl_energy
           
    def discharge(self, output_energy) -> Tuple[float, float]:
        """
        Discharges energy = output_energy from the battery in kWh. 
        Decrements the capacity taking into account the discharge losses,
        which means that the actual capacity decrement will be larger than the
        
        input:
            output_energy (float): The energy to be extracted from the battery in kWh, 
                                    taking into account discharge losses.
        
        return:
            full_decrement (float): The total amount of energy taken from the battery capacity 
                                    during discharge, taking into account losses.
            avl_energy (float): The updated current capacity of the battery available for use in kWh after
                                discharging is completed.
        """
        if output_energy > 0:
            raise ValueError(f'Non-negative input! {output_energy}')
        elif output_energy == 0: #Skip 
            return 0, self.avl_energy 

        available_capacity = self.get_discharge_potential()      
        # avoid fp error
        if abs(output_energy - available_capacity) < self.eps:
            output_energy = available_capacity


        if output_energy / self.discharge_loss > available_capacity:
            full_decrement = available_capacity
        else:
            # Calculate the full amout required to be extracted including losses
            full_decrement = output_energy / self.discharge_loss
        
        # Discharge battery, update capacity decrement
        self.avl_energy += full_decrement
        self.update_capacity_degredation(full_decrement)
        self.soc = self.avl_energy / self.capacity
            
        return full_decrement, self.avl_energy 
       
    def update_capacity_degredation(self, energy):        
        """
        Simulates the reduction in maximum battery capacity as a result of use.
        In this case degredation is a function of the number of  'round trips' 
        (discharge/charge flow equivalent to 2x the maximum capacity.
        Models a linear decrease vs number of trips.
        
        inputs:
            energy(float): the energy either charged or discharged by the battery in kWh
        """
        
        # Update total energy flow aggregator
        self.total_energy_flow += abs(energy)
        
        # Convert total energy flow into a count of the number of round trips
        self.trip_count = self.total_energy_flow // (self.capacity * 2)
        
        if self.trip_count > self.previous_trip_count:
            
            # Decrement total capacity
            self.capacity -= self.capacity_degredation_rate
        
        # Update previous_trip_count
        self.previous_trip_count = self.trip_count
        
    def get_charge_potential(self) -> float:        
        """
        returns the maximum allowable input into the battery in kWh
        over a given time interval, allowing for charge losses.
        Value returned is in absolute magnitude, not taking into account 
        direction of flow.
        
        inputs:
            capacity (float): total current storage potential of the battery 
                in kWh.
            avl_energy (float): total current energy present in the battery 
                in kWh.
            charge_loss (float): the percentage loss in energy due to charging 
                efficiency losses.
        
        return:
            charge_max (float): the maximum allowable input into the battery 
                over the given time interval in kWh. 
        """
        
        capacity=self.capacity 
        avl_energy=self.avl_energy 
        charge_loss=self.charge_loss
             
        charge_potential = (capacity - avl_energy) / charge_loss
        
        # Now we return the lesser of the two to prevent 'over' charging
        if self.max_input >= charge_potential:
            charge_max =  charge_potential #limited by space available
        else:
            charge_max = self.max_input #Limited by discharge rate/time
            
        if charge_max < self.eps:
            charge_max = 0
            
        return charge_max
    
    def get_discharge_potential(self) -> float:        
        """
        returns the maximum available energy output from the battery for some 
        time interval, allowing for discharge losses.
        Energy returned is in absolute magnitude, not taking into account
        direction. 
        
        inputs:
            avl_energy (float): total current energy present in the battery in 
                kWh.
            discharge_loss (float): the percentage loss in energy due to 
                discharging efficiency losses.
        
        return:
            discharge_max (float): the maximum available output from the 
            battery over the given time interval in kWh. 
        """
        avl_energy = self.avl_energy 
        discharge_loss = self.discharge_loss
        
        discharge_potential = avl_energy * discharge_loss
        
        # Now we return the lesser of the two to prevent over discharging
        if self.max_output >= discharge_potential:
            discharge_max = discharge_potential # limited by available energy
        else:
            discharge_max = self.max_output #Limited by discharge rate/time
            
        if discharge_max < self.eps:
            discharge_max = 0
        
        return discharge_max
        

    def snapshot(self, timestamp, setpoint = None):
        """
        WIP
        snapshots the current state of the battery 
        """
       
        update_dict = {
            "timestamp": timestamp,
            "Total Capacity": self.capacity,
            "Available Capcity": self.avl_energy,
            "Total Energy Flow": self.total_energy_flow,
            "Trip Count": self.trip_count,
            "Max Discharge": self.get_discharge_potential(),
            "Max Charge": self.get_charge_potential(),
            "Discharge Limit": self.battery_limits[0],
            "Charge Limit": self.battery_limits[1],
            "Setpoint": setpoint
        }            
            
        for key, val in update_dict.items():
            self.status_dict[f"{key}"].update(val)
        
    def get_limits(self, solar, load) -> Tuple[float,float]:
        """
        Calculates the operation limits of the battery, given some inputs for 
        the system solar outputs and household loads. It calculates the net 
        energy given some solar and house hold, and then depending on whether 
        the battery is allowed to push HOMER into a state of grid export, will 
        calculate a discharge limit. Similarly, it also checks whether the 
        battery is able to charge from the grid, and sets an appropriate charge
        limit. These limits take into account the maximum charge/discharge rate 
        and the available/empty capacity by calling the .get_charge_potential()
         and the .get_discharge_potential() methods of the `battery` class. 
        
        The discharge limit is negative or zero, and the charge limit is 
        positive or zero. Together these form a continuous range of allowable 
        values for the battery.
        inputs:
            solar (float): The solar system output in kWh
            load (float): The total household loads in kWh 
        return:
            battery_limit (tuple): (discharge_limit, charge_limit)
                Interval endpoints bounding the allowable the battery 
                operation settings where:
            discharge_limit (float): the max amount of energy the battery can 
                discharge for the current system state in kWh. 
            charge_limit (float): the max amount of ebergy the battery could 
                receive to charge within the current system state in kWh.
        """

        # Has solar and battery.    
        if self.solar: 
            solar = solar
        # No solar, battery only.
        else:
            solar = 0

        # Calculate net energy:
        # -ve is excess solar, +ve is excess load
        net_energy = solar + load

        # Get max output from battery
        max_batt_supply = self.get_discharge_potential()
        max_batt_consumption = self.get_charge_potential() 

        ## Solar exceeds load
        if net_energy < 0: 

            ## Get Charge limits: check if battery can charge from grid
            if self.importable:
                # If we can charge from grid, we are limited by the available battery capacity/
                # and max charge rate, whichever is lesser. 
                charge_limit = max_batt_consumption
            else:
                # We cannot charge from the grid, we are limited by the excess solar,
                # and the max charge rate, whichever is lesser. 
                charge_limit = min(max_batt_consumption, abs(net_energy))

            ## Get Discharge limits: check if battery can export to grid
            if self.exportable: 
                # We can export as much as we can, Load/Solar irrelevant.
                discharge_limit = - max_batt_supply
            else: 
                # There is an excess of energy in the system already, we cannot discharge
                discharge_limit = 0

        ## Load exactly matched by solar, or solar deficit/no solar.
        elif net_energy >= 0:              

            ## Get Charge limits: check if battery can charge from grid
            if self.importable:
                # If we can charge from grid, we are limited by the available free capacity
                # And max charge rate, whichever is lesser. 
                charge_limit = max_batt_consumption
            else:
                # We cannot charge from the grid, there is no excess solar, so we cannot charge at all. 
                charge_limit = 0

            ## Get Discharge limits: check if battery can export to grid
            if self.exportable: 
                # We can export as much as we can, Load/Solar irrelevant,
                # limited only by max discharge/ current capacity
                discharge_limit = - max_batt_supply
            else: 
                # We cannot discharge more than the solar deficit, or
                # the max discharge/current capacity of our battery
                discharge_limit = - min(max_batt_supply, abs(net_energy))
        else:
            print(net_energy)
            raise Exception
        # Return discharge and charge bounds. 
        self.battery_limits = (discharge_limit, charge_limit)
        return self.battery_limits

    def reset(self, capacity, starting_capacity) -> None:
        
        # Reset capacty and state of charge
        self.capacity = capacity
        self.avl_energy = self.capacity * starting_capacity 

        # Reset Decrement Counters
        self.total_energy_flow = 0
        self.trip_count = 0
        self.previous_trip_count = 0

        # Reset Operational Limits and History
        self.battery_limits = ()
        self.status_dict = {
            "timestamp": [],
            "Total Capacity": [],
            "Available Capcity": [],
            "Total Energy Flow": [],
            "Trip Count": [],
            "Max Discharge": [],
            "Max Charge": [],
            "Discharge Limit": [],
            "Charge Limit": [],
            "Setpoint": []
        }