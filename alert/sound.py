try:
    import winsound
except ImportError:
    winsound = None


class AlertSound:

    def __init__(self):
        self.played = False

    def reset(self):
        self.played = False

    def play(self):

        if self.played:
            return

        self.played = True

        if winsound is not None:
            winsound.Beep(1400, 300)