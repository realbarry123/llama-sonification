import random
import torch

class ModelWrapper():

    def __init__(self):
        self.vocab = (
            " ",
            "ear",
            "gly",
            "tion",
            "loab",
            "so",
            "uh",
            "um",
            "ac",
            "tual",
            "ly"
        )

    def next(self):
        states = torch.linspace(-1, 1, 17).view(1, 17, 1).expand(3, 17, 1048)
        return random.choice(self.vocab), states
    
if __name__ == "__main__":
    bob = ModelWrapper()
    print(bob.next())