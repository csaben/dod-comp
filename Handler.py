# Imports
from PlannerProto_pb2 import ScenarioConcludedNotificationPb, \
    ScenarioInitializedNotificationPb  # Scenario start/end notifications
from PlannerProto_pb2 import ErrorPb  # Error messsage if scenario fails
from PlannerProto_pb2 import StatePb, AssetPb, TrackPb  # Simulation state information
from PlannerProto_pb2 import OutputPb, ShipActionPb, WeaponPb
from publisher import Publisher
from AiManager import AiManager
from google.protobuf.json_format import MessageToDict, ParseDict
import json
from pathlib import Path
import PlannerProto_pb2
import numpy as np
import re
import time
import math
import sys
from State import *

class Handler(AiManager):
    # Constructor
    def __init__(self, publisher: Publisher):
        print("Constructing AI Manager")
        self.ai_pub = publisher
        self.memory = []
        self.directory = Path("./output4/")
        self.base_file = "ttd_state.json"
        self.filepath = self.get_next_filepath(self.directory, self.base_file)
        self.ifYouShootShutUp = []
        self.targeted_track_ids=[]
        self.default_behavior = "shoot_all" #"protect HUV" "shoot closest with closest"

    # Is passed StatePb from Planner
    def receiveStatePb(self, msg: StatePb):
        # self.printStateAsDict(msg)
        output_message: OutputPb = OutputPb()
        output_message = self.generateState(msg)
        self.ai_pub.publish(output_message)

    # This method/message is used to notify of new scenarios/runs
    def receiveScenarioInitializedNotificationPb(self, msg: ScenarioInitializedNotificationPb):
        print("Scenario run: " + str(msg.sessionId))

    # This method/message is used to nofify that a scenario/run has ended
    def receiveScenarioConcludedNotificationPb(self, msg: ScenarioConcludedNotificationPb):
        print("Ended Run: " + str(msg.sessionId) + " with score: " + str(msg.score))


    # Generate a state
    def generateState(self, msg: StatePb):
        situation = self.parseSituation(msg)
        if situation[0] == "idle":
            return situation[1]
        elif situation[0] == "combat":
            missle_info = self.missleInformaton(msg)
            routine = self.routine_selector(missle_info)
            self.default_behavior = routine

            #grab relevant combat solution
            combat_solution = next((method_name for method_name in dir(ai_manager)
                        if callable(getattr(ai_manager, method_name)) #EDIT AIMAN
                        and method_name.__contains__(self.default_behavior)), None)

            #You can add more combat solutions in a separate file and class
            #and replace the ai_manager with the name of your class and then
            #you can access all your methods in that class cleanly with this
            #string notation

            #error check
            if combat_solution is None:
                print("error in combat solution")
                sys.exit()

            #generate execution order based on combat solution
            execution_order = combat_solution(msg)


    def parseSituation(self, msg: StatePb):
        # idle
        if 'Tracks' not in json_dict:
            print("idle")
            output_message: OutputPb = OutputPb()
            return "idle", output_message

        elif 'Tracks' in json_dict:
            print("combat")
            return "combat"

        else:
            print("error in parseSituation method")
            sys.exit()

    def routine_selector(self, msg: StatePb):
        pass

    def missleInformaton(self, msg: StatePb):
        pass

    def combatRoutine_shoot_all(self, msg: StatePb):
        output_message: OutputPb = OutputPb()
        pass






