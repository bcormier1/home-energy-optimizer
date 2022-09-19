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
	discharge_loss: float # percentage of energy lost in the discharge process
	charge_loss: float # percentage of energy lost in the charge process


	def __init__(self, capacity=14, dischargeable = False, charge_rate=5, discharge_rate=5, 
		discharge_loss=0, charge_loss=0):
		"""
		Initialize battery specs.

		`capacity` is the total energy of the battery in kWh

		`dischargeable` represents if th battery can send it's power back to the optimizer for use in another
		purpose.  For example a car cannot do this while a wall battery can.

		`charge_rate` is the rate of battery charging in kW

		`discharge_rate` is the rate of battery discharge (if applicable) in kW
		"""
		self.capacity = capacity
		self.dischargeable = dischargeable
		self.charge_rate = charge_rate
		self.discharge_rate = discharge_rate

		self.avl_energy = 0 # current available energy

		self.discharge_loss = discharge_loss
		self.charge_loss = charge_loss

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





