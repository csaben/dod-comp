from PlannerProto_pb2 import ScenarioConcludedNotificationPb, \
    ScenarioInitializedNotificationPb  # Scenario start/end notifications
from PlannerProto_pb2 import ErrorPb  # Error messsage if scenario fails
from PlannerProto_pb2 import StatePb, AssetPb, TrackPb  # Simulation state information
from PlannerProto_pb2 import OutputPb, ShipActionPb, WeaponPb
from publisher import Publisher
from AiManager import AiManager
from google.protobuf.json_format import MessageToDict, ParseDict
import PlannerProto_pb2

import os
import sys
import time
import pprint
from Strategy import Strategy


class LocalGameGenerator(AiManager):
    # Constructor
    def __init__(self, publisher: Publisher, strategy: str):
        print("Constructing AI Manager")
        self.ai_pub = publisher
        self.folder = f"./sample_games/{strategy}"
        #make a folder if it doesn't exist
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.strategy = strategy

    def receiveScenarioConcludedNotificationPb(self, msg: ScenarioConcludedNotificationPb):
        print("Ended Run: " + str(msg.sessionId) + " with score: " + str(msg.score))

    # Is passed StatePb from Planner
    def receiveStatePb(self, msg: StatePb):
        output_message: OutputPb = OutputPb()

        # collect strategy based on cli call
        self.collectStrategy(Strategy)

        # use strategy, if one present
        if self.strategy:
            output_message = getattr(Strategy, self.strategy)(msg)

        # write to file
        self.writePbToFile(msg)

        # publish to planner
        self.ai_pub.publish(output_message)

    def writePbToFile(self, msg: StatePb):
        filepath = utils.get_next_filepath(self.folder, "state_pb", "bin")
        with open(filepath, "wb") as f:
            f.write(msg.SerializeToString())

    def returnDictionary(self, msg: StatePb):
        # StatePb
        message = PlannerProto_pb2.StatePb()
        # serialize the message
        msg = msg.SerializeToString()
        message.ParseFromString(msg)
        message_dict = MessageToDict(message)
        # return as dictionary
        return message_dict

    def collectStrategy(self, strategy_manager: Strategy) -> callable:
        if self.strategy in dir(strategy_manager) and callable(getattr(strategy_manager, \
                    self.strategy)) and self.strategy.__contains__(f"{self.strategy}"): 
            print(f"{self.strategy} strategy found!")
        else:
            self.strategy = None
            print(f"No strategy, {self.strategy} found!")

class utils():

    @staticmethod
    def get_next_filepath(directory, base_filename, extension):
        import pathlib
        from pathlib import Path
        index = 1
        while True:
            filename = f"{base_filename}_{index}.{extension}"
            directory = Path(directory)
            filepath = directory / filename
            if not filepath.exists():
                return filepath
            index += 1
