from collections import deque
import time
import cv2


class VideoBuffer:

    def __init__(

        self,

        fps=30,

        seconds=10,

        frame_size=(640,480)

    ):

        self.fps = fps

        self.seconds = seconds

        self.frame_size = frame_size

        self.max_frames = fps * seconds

        self.buffer = deque(maxlen=self.max_frames)

    def add_frame(self, frame):

        self.buffer.append(frame.copy())

    def save(self, filename):

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(

            filename,

            fourcc,

            self.fps,

            self.frame_size

        )

        for frame in self.buffer:

            writer.write(frame)

        writer.release()