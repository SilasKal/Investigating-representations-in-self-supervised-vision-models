

class BaseTransform:
    def __init__(self):
        pass

    def get(self):
        raise NotImplementedError()

    def set(self, value):
        raise NotImplementedError()

    def __str__(self):
        pass
