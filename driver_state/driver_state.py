class State:

    def __init__(self):
        self.speed = 0
        self.eye_direction = None
        self.has_stop_sign = False 

    def __repr__(self):
        return "speed: {}, eye_direction: {}, has_stop_sign: {}".format(self.speed, self.eye_direction, self.has_stop_sign)


class DriverState:
    def __init__(self):
        self.states = []
        self.label = None

    def append_state(self, state):
        self.states.append(state)

