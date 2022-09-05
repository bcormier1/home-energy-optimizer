class BatteryFullError(Exception):
	pass

class BatteryEmptyError(Exception):
	pass

class battery:
	size: float
	dischargeable: bool
	charge_rate: float
	discharge_rate: float
	capacity: float


	def __init__(size=14, dischargeable = False, charge_rate=5, discharge_rate=5):
		"""
		Initialize battery specs.

		`size` is the total energy of the battery in kWh

		`dischargeable` represents if th battery can send it's power back to the optimizer for use in another
		purpose.  For example a car cannot do this while a wall battery can.

		`charge_rate` is the rate of battery charging in kW

		`discharge_rate` is the rate of battery discharge (if applicable) in kW
		"""
		self.size = size
		self.dischargeable = dischargeable
		self.charge_rate = charge_rate
		self.discharge_rate = discharge_rate

		self.capacity = 0 # current capicity

	def charge(time=1):
		"""
		Charge the battery for the given time in minutes
		"""
		for minute in time:
			self.capacity += self.charge_rate/60

			if self.capacity > self.size:
				self.capacity = self.size
				raise BatteryFullError

	def discharge(time=1):
		"""		
		Discharge the battery for the given time in minutes
		"""
		for minute in time:
			self.capacity -= self.charge_rate/60

			if self.capacity < self.size:
				self.capacity = 0
				raise BatteryEmptyError






