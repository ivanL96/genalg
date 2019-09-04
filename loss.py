
class GeneticBaseLoss(object):
    def __init__(self, target=None, **kwargs):
        self.target = target

    def compute(self):
        raise NotImplementedError


class F:
    def __init__(self):
        pass
     