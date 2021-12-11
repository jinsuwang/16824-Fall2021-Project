class BaseEvaluationStrategy:

	def evaluate(self, driver_state):
		return DriverEvaluationResult()


class StopSignEvaluationStrategy(BaseEvaluationStrategy):

	def evaluate(self, driver_state):
		final_label = None
		has_stop_sign = False 
		looked_left = False
		looked_right = False 

		for idx, state in enumerate(driver_state.states):

			if state.has_stop_sign:
				has_stop_sign = True

			if has_stop_sign and state.eye_direction == "left":
				looked_left = True

			if has_stop_sign and looked_left and state.eye_direction == "right":
				looked_right = True

			if has_stop_sign and looked_left and looked_right:
				final_label = "pass"


		result = DriverEvaluationResult()
		result.final_label = final_label
		result.has_stop_sign = has_stop_sign
		result.looked_left = looked_left
		result.looked_right = looked_right

		return result

class DriverEvaluationResult:

	def __init__(self):
		self.final_label = None
		self.has_stop_sign = False 
		self.looked_left = False
		self.looked_right = False 
