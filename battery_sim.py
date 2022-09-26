class BatteryFullError(Exception):
	pass

class BatteryEmptyError(Exception):
	pass

class BatteryNotDischargeableError(Exception):
	pass

class battery:
	capacity: float
	dischargeable: bool
	charge_rate: float
	discharge_rate: float
	avl_energy: float
## AT: not sure if these two now need to be listed here seeing as they are not attributes and not inputs. 
    discharge_loss: float # percentage of energy lost in the discharge process
	charge_loss: float # percentage of energy lost in the charge process
##
	round_trip_efficiency: float # Percentage of energy recoverable from each unit of input energy

	def __init__(self, capacity=14, dischargeable = False, charge_rate=5, discharge_rate=5, 
		round_trip_efficiency=0.9):
		"""
		Initialize battery specs.

		`capacity` is the total current energy capacity of the battery in kWh        

		`dischargeable` represents if th battery can send it's power back to the optimizer for use in another
		purpose.  For example a car cannot do this while a wall battery can.

		`charge_rate` is the rate of battery charging in kW

		`discharge_rate` is the rate of battery discharge (if applicable) in kW
        
        `total_energy_flow` is the highest allowable charge/dischare rate in kW
        
        `trip_count` is the total number of full cycles (Charge and discharge) of the battery. 
        
        `capacity_degredation_rate` is the rate at which the capacity is decrememnted per trip in kWh/trip
        
        `previous_trip_count` starting counter for the trip count.
        
        `interval` is the action interval in minutes. calc
        
		"""
        
		self.capacity = capacity
		self.dischargeable = dischargeable
		self.charge_rate = charge_rate
		self.discharge_rate = discharge_rate
        
        # AT: defined seperately here so as to not confuse with your current usage above. 
        # depending how we set it up these may not always be the same.
		self.max_charge_rate = 5 # in kW from the tesla datasheet
		self.max_discharge_rate = 5 # in kW from the tesla datasheet

        # Variables used for capacity decrementing
        self.total_energy_flow = 0
        self.trip_count = 0
        self.capacity_degredation_rate = 0.00025 # kWh/trip for Tesla Powerwall 2
        self.previous_trip_count = 0
        
        # Potential variable for simultion with fixed time intervals
        self.interval = 5 # Action interval length in minutes
        
        # Define the max input/output energy for the given interval
        self.max_input = self.max_charge_rate / 60 * self.interval
        self.max_output = self.max_discharge_rate / 60 * self.interval

		self.avl_energy = 0 # current available energy
        
        # Assume same rate for charge and discharge. 
		self.discharge_loss = round_trip_efficiency ** 0.5
		self.charge_loss = round_trip_efficiency ** 0.5

	def charge(self, time=5):
		"""
		Charge the battery for the given time in minutes. return energy consumed
		"""
		potential_energy_change = (self.charge_rate/60)*time
		# have to update current state of charge with less than was sent to the battery to compensate for loss
		self.avl_energy += potential_energy_change - (self.charge_loss*potential_energy_change)

		if self.avl_energy > self.capacity:
			self.avl_energy = self.capacity
			raise BatteryFullError

		return potential_energy_change

	def discharge(self, time=5):
		"""		
		Discharge the battery for the given time in minutes. return energy discharged
		"""
		if not self.dischargeable:
			raise BatteryNotDischargeableError
		else:
			potential_energy_change = (self.dischargec_rate/60)*time
			# have to pull more energy than is in the battery to compensate for loss
			self.avl_energy -= potential_energy_change + (self.discharge_loss*potential_energy_change)

			if self.avl_energy < self.capacity:
				self.avl_energy = 0
				raise BatteryEmptyError

		return potential_energy_change
    
    def update_capacity_degredation(self, energy):
        
        """
        Simulates the reduction in maximum battery capacity as a result of use.
        In this case a degredation is a function of the number of 
        'round trips' (discharge/charge flow equivalent to 2x the maximum capacity.
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
        self.trip_count = self.previous_trip_count
        
    def get_charge_potential(self):
        
        """
        returns the maximum allowable input into the battery in kWh
        over a given time interval.
        """
        
        charge_potential = (self.capacity - self.avl_energy) / self.charge_loss
        
        # Now we return the lesser of the two to prevent 'over' charging
        if self.max_input >= charge_potential:
            return charge_potential #limited by space available
        else:
            return self.max_input #Limited by discharge rate/time
    
    def get_discharge_potential(self):
        
        """
        returns the maximum available energy output from the battery for some
        time interval
        """
        
        discharge_potential = self.avl_energy * self.discharge_loss
        
        if self.max_output >= discharge_potential:
            return discharge_potential # limited by available energy
        else:
            return self.max_output #Limited by discharge rate/time
       





