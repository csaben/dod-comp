#Imports
import math

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
import time


class State2(AiManager):
    # Constructor
    def __init__(self, publisher: Publisher):
        print("Constructing AI Manager")
        self.ai_pub = publisher
        self.memory = []
        # depends on model
        self.directory = Path("./output2/")
        self.base_file = "ttd_state.json"
        self.filepath = self.get_next_filepath(self.directory, self.base_file)
        self.ifYouShootShutUp = []
        self.future_sight = []

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

    # Example function for building OutputPbs, returns OutputPb

    # function to get the next available file path
    def get_next_filepath(self, directory, base_filename):
        index = 1
        while True:
            filename = f"{base_filename}_{index}.json"
            filepath = directory / filename
            if not filepath.exists():
                return filepath
            index += 1

    def generateState(self, msg: StatePb):
        json_dict = self.cleanState(msg)
        # (Shane) Append state as dictionary to output file

        # needs to make new file if file already tyhere

        with open(self.filepath, "a") as file:
            file.write(json.dumps(json_dict) + "\n")

        # idle
        if 'Tracks' not in json_dict:
            print("idle")
            output_message: OutputPb = OutputPb()
            return output_message
        else:
            origin = self.calculateOrigin(json_dict['assets'])
            # execution order is a list of tuples (time, id)
            execution_order = self.calculateExecutionOrder(json_dict['Tracks'], origin)
            output_message: OutputPb = OutputPb()

            execution_order = [enemy for enemy in execution_order if enemy[1] not in self.memory]
            # print(execution_order)
            # make sure its still sorted (it should be lol)
            execution_order = sorted(execution_order, key=lambda x: x[0])

            # This should fix your memory issue
            if len(self.memory) == 30:
                self.memory = []
                self.ifYouShootShutUp = []

            for missle in execution_order:
                ship_action: ShipActionPb = ShipActionPb()
                # set the target id to the missle id
                ship_action.TargetId = missle[1]
                ship_action.weapon, ship_action.AssetName \
                    = self.whoShootsFirst(json_dict['assets'], missle[1])

                if ship_action.weapon != "":
                    output_message.actions.append(ship_action)

                else:
                    self.ifYouShootShutUp = []
                    return output_message
            print("printing execution order")
            print(execution_order)
            print("printing memory")
            print(self.memory)
            self.ifYouShootShutUp = []
            print("printing output message")
            print(output_message.actions)

            self.calculateWhoTargeted(json_dict['Tracks'], json_dict['assets'])
            return output_message


    def calculateWhoTargeted(self, missile_list, assets):
        for asset in assets:
            if asset['health'] == -1:
                pass
            else:
                for missile in missile_list:
                    dist = math.sqrt((asset['PositionX'] - missile['PositionX']) ** 2 + (
                                asset['PositionY'] - missile['PositionY']) ** 2)
                    magnitude = math.sqrt(missile['PositionX'] ** 2 + missile['PositionY'] ** 2)
                    print("Distance "+str(dist))
                    print("Magnitude "+str(magnitude))
                    if dist <= magnitude:
                        print(str(missile['TrackId']) + " will impact " + asset['AssetName'])
    def find_value(self, list_of_dicts, key1, value1, key2, target_id):
        """
        Find a value in a list of dictionaries that contain another list of dictionaries.
        Args:
        - list_of_dicts: a list of dictionaries that contain another list of dictionaries.
        - key1: the key to search for in the first level of dictionaries.
        - value1: the value to search for in the first level of dictionaries.
        - key2: the key to search for in the second level of dictionaries.
        Returns:
        - The value associated with the key2 in the first matching dictionary.
        - None if the value1 is not found in any of the dictionaries.
        """
        whosShooting = ""
        withWhat = ""
        # update after you send a message each time
        # ifYouShootShutUp=[]
        for dictionary in list_of_dicts:
            # if dictionary.get(key1) != value1:
            if dictionary.get(key1) != value1:  # and dictionary.get("AssetName") not in self.ifYouShootShutUp:
                # print(self.ifYouShootShutUp)
                whosShooting = dictionary.get("AssetName")
                for nested_dict in dictionary.get("weapons"):
                    if key2 in nested_dict.keys() and (
                    dictionary.get("AssetName"), nested_dict.get("SystemName")) not in self.ifYouShootShutUp:
                        withWhat = nested_dict.get("SystemName")
                        self.memory.append(target_id)
                        self.ifYouShootShutUp.append((whosShooting, withWhat))
                        return withWhat, whosShooting
        return "", ""

    def whoShootsFirst(self, assets, target_id) -> tuple:
        # start over because its not working
        return self.find_value(assets, "health", -1, "Quantity", target_id)
        # '''
        # (
        # ship_action.AssetName = "Galleon HVU"
        # ship_action.weapon = "Chainshot_System"
        # )
        # '''
        # # print(assets)
        # whosShooting = None
        # for asset in assets:
        #     if asset['health'] == -1:
        #         pass
        #     else:
        #         time.sleep(.5)
        #         for weapon in asset['weapons']:
        #             if (weapon['WeaponState'] == 'Ready') and ("Quantity" in weapon.keys()) and (asset['AssetName']!="HVU_Galleon_0"):
        #                 print(asset["AssetName"])
        #                 if weapon["Quantity"] >3:
        #                     # print(target_id)
        #                     self.memory.append(target_id)
        #                     print("memory: ", self.memory)
        #
        # # return "Cannon_System", "HVU_Galleon_0"
        #                     return  weapon['SystemName'], asset['AssetName']
        #                 else:
        #                     pass
        #
        #             else:
        #                 pass

    @staticmethod
    def calculateExecutionOrder(missle_list, origin) -> list:  # of tuples
        # t = (s/m) * m = s
        intercept_times = []
        # get position, divide by speed, sort by lowest time to intercept
        # exclude elevation for now
        for missle in missle_list:
            if ">" in missle.get("ThreatId"):
                pass
            else:

                # easier to read
                x = missle['PositionX']
                y = missle['PositionY']
                vx = missle['VelocityX']
                vy = missle['VelocityY']

                # distance eqn
                d = np.sqrt((x - origin[0]) ** 2 + (y - origin[1]) ** 2)
                # velocity eqn
                dv = np.sqrt(vx ** 2 + vy ** 2)

                # time to intercept
                t = d / dv
                print(missle)
                intercept_times.append((t, missle['TrackId']))

        return sorted(intercept_times, key=lambda x: x[0])

    @staticmethod
    def calculateOrigin(asset_list) -> np.ndarray:
        gamma = 1e-6
        origin = np.array([0, 0], dtype=float)
        for asset in asset_list:
            if 'health' in asset.keys():
                if asset['health'] == -1:
                    pass
                else:
                    origin += np.array([asset['PositionX'], asset['PositionY']])
            else:
                # when no healths just return zero (some bug)
                return 0
        return origin / (len(asset_list) - 1 + gamma)

    def cleanState(self, msg: StatePb):
        # StatePb
        message = PlannerProto_pb2.StatePb()
        # serialize the message
        msg = msg.SerializeToString()
        message.ParseFromString(msg)
        message_dict = MessageToDict(message)
        # json_str = json.dumps(message_dict)
        # print(json_str)
        return message_dict
        # print(json_str)
        # with open("./state.json", 'w') as f:
        #     f.write(json_str)
        # import sys
        # sys.exit()
