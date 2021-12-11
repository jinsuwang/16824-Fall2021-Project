from driver_state.driver_state import DriverState, State

class LabelReader():

    def __init__(self, frames, fps):
        self.fps = fps
        self.frames = frames


    def generate_driver_state(self, label_path):

        driver_state = DriverState()
        for i in range (self.frames):
            driver_state.append_state(State())

        with open(label_path) as f:
            
            line = f.readline()
            assert line.strip() == "# overall"
            line = f.readline()
            driver_state.label = line
            line = f.readline()
            assert line.strip() == "# eye direction"
            
            while line:
                line = f.readline()
                if line.strip() == "# stop sign":
                    break
                print("Processing... ", line)
                start = float(line.split(",")[0])
                end = float(line.split(",")[1])
                label = line.split(",")[2]
                self.update_driver_state(driver_state, start, end, label, "eye_direction")

            
            line = f.readline()
            while line:
                print(line.split(","))
                start = float(line.split(",")[0])
                end = float(line.split(",")[1])
                label = bool(line.split(",")[2])          
                self.update_driver_state(driver_state, start, end, label, "has_stop_sign")
                line = f.readline()


        print("=========== generated states from file ==============")
        for s in driver_state.states:
            print(s)

        return driver_state 


    def update_driver_state(self, driver_state, start, end, label, field):
        start_index = int(float(start) * self.fps)
        end_index = int(float(end) * self.fps)
        for i in range(start_index, end_index):
            setattr(driver_state.states[i], field, label)



label_reader = LabelReader(50, 5)
label_reader.generate_driver_state("label_generation/video_1_driver_state.txt")

