from enum import Enum
import time


class RecorderState(Enum):

    IDLE = 0

    PREPARING = 1

    RECORDING = 2

    POST_EVENT = 3


class IncidentRecorder:

    def __init__(self):

        self.state = RecorderState.IDLE

        self.start_time = 0

        self.post_start = 0

        self.trigger_time = 1.0

        self.post_duration = 10

    def update(self, high_threat):

        now = time.time()

        if self.state == RecorderState.IDLE:

            if high_threat:

                self.start_time = now

                self.state = RecorderState.PREPARING

        elif self.state == RecorderState.PREPARING:

            if not high_threat:

                self.state = RecorderState.IDLE

            elif now - self.start_time >= self.trigger_time:

                self.state = RecorderState.RECORDING

                print("Recording Started")

        elif self.state == RecorderState.RECORDING:

            if not high_threat:

                self.post_start = now

                self.state = RecorderState.POST_EVENT

        elif self.state == RecorderState.POST_EVENT:

            if high_threat:

                self.state = RecorderState.RECORDING

            elif now - self.post_start >= self.post_duration:

                self.state = RecorderState.IDLE

                print("Incident Finished")

        return self.state