from AiManager import *


#TODO: should inititialize a set of models, heuristic fn for testing, or a model manager
class Ai:
    def __init__(self, aiManager):
        self.aiManager = aiManager
        self.aiManager.addAi(self)

    def update(self):
        pass


