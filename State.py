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

class State(AiManager):
    # Constructor
    def __init__(self, publisher: Publisher):
        print("Constructing AI Manager")
        self.ai_pub = publisher
        self.memory = []
        # depends on model
        self.directory = Path("./output4/")
        self.base_file = "ttd_state.json"
        self.filepath = self.get_next_filepath(self.directory, self.base_file)
        self.ifYouShootShutUp = []
        self.targeted_track_ids=[]

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
            #print(json_dict["Tracks"])
            pattern=r":(\d+)"
            #make a memory of what you have shot at by parsing for "Chainshot_17>enemy_track:18"
            for track in json_dict["Tracks"]:
                if track["ThreatRelationship"]=="Friendly":
                    #print(track["ThreatRelationship"])
                    match = re.search(pattern, track["ThreatId"])
                    self.targeted_track_ids.append(int(match.group(1)))
            output_message: OutputPb = OutputPb()

            self.memory = [x for x in self.memory if x in self.targeted_track_ids]
                # if track with the track id has an enemy with the corresponding enemy value in memory

                # OR just make it so memory appends the track_ids
            execution_order = [enemy for enemy in execution_order if enemy[1] not in self.memory]
            # execution_order = [enemy for enemy in execution_order if enemy[1] not in self.active_defense]
            # print(execution_order)
            # make sure its still sorted (it should be lol)
            execution_order = sorted(execution_order, key=lambda x: x[0])

            # This should fix your memory issue
            #check tracking data to see if a friendly missle will hit (just make a memory from missles attacking enemies)
            # if len(self.memory) == 30:
            #     self.memory = []
                # self.ifYouShootShutUp = []

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
            #print("printing execution order")
            #print(execution_order)
            print("printing memory")
            #print(self.memory)
            self.ifYouShootShutUp = []
            #print("printing output message")
            #print(output_message.actions)
            return output_message

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
    def calculateProtectionOrder(self):
        executionOrder = []

        #sort by HVU
        for target in self.distanceList:
            if 'HVU' in target[0]:
                executionOrder.append(target)
                #self.memory.append(target[1])
         #sort by lowest health ship

        self.distanceList = sorted(self.distanceList, key=lambda x: x[3])

        for target in self.distanceList:
            if 'HVU' not in self.distanceList:
                executionOrder.append(target)
                #self.memory.append(target[1])
        return executionOrder
    def calculateWhoTargeted(self, missile_list, assets):
        closestDistance = 100000
        trackID = 0
        #distanceList = []
        for missile in missile_list:
            #distanceList = []
            for asset in assets:

                if asset['health'] == -1:
                    pass
                else:
                    dist = math.sqrt((asset['PositionX'] - missile['PositionX']) ** 2 + (
                    asset['PositionY'] - missile['PositionY']) ** 2)
                    magnitude = math.sqrt(missile['PositionX'] ** 2 + missile['PositionY'] ** 2)
                    # print(f"Missile {missile['TrackId']} Position {missile['TrackId']} X {missile['PositionX']} Missile Y {missile['PositionY']}")
                    # print(f"Missile Velocity {missile['TrackId']} XV {missile['VelocityX']} Missile YV {missile['VelocityY']}")
                    # print(f"Ship {asset['AssetName']} is at {asset['PositionX']}, {asset['PositionY']}")

                   # Given values
                    missile_x = missile['PositionX']
                    missile_y = missile['PositionY']
                    missile_vx = missile['VelocityX']
                    missile_vy = missile['VelocityY']

                    ship_x = asset['PositionX']
                    ship_y = asset['PositionY']

                    # Calculate slope and intercept of missile trajectory
                    m = missile_vy / missile_vx
                    b = missile_y - m * missile_x

                    # Calculate closest distance between missile trajectory and ship
                    distance = abs(-1 * m * ship_x + ship_y - b) / math.sqrt(m ** 2 + 1)

                    add = True

                    for tuple in self.distanceList:
                        if asset['AssetName']==tuple[0] and missile['TrackId']==tuple[1]:
                            add = False

                    if add == True:
                        self.distanceList.append((asset['AssetName'], missile['TrackId'],  distance, asset['health']))

            self.distanceList = sorted(self.distanceList, key=lambda x: x[2])



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
                #print(missle)
                intercept_times.append((t, missle['TrackId']))

        return intercept_times#, key=lambda x: x[0]

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

    # def printStateAsDict(self, msg:StatePb):
    #     from protobuf_to_dict import protobuf_to_dict, get_field_names_and_options
    #     for field, field_name, options in get_field_names_and_options(PlannerProto_pb2.StatePb):
    #         print('name: {}, options: {}'.format(field_name, options))

    def createActions(self, msg: StatePb):

        # ShipActionPb's go into an OutputPb message
        output_message: OutputPb = OutputPb()
        # print("**********************************")
        import time
        time.sleep(.25)
        # import sys
        # sys.exit()
        # ShipActionPb's are built using the same sytax as the printStateInfo function

        # sample shooting a ship heuristically
        ship_action: ShipActionPb = ShipActionPb()
        # for track in msg.Tracks:
        #     target = track.TrackId
        target = 1
        ship_action.TargetId = target
        # asset name
        for asset in msg.assets:
            myAsset = str(asset.AssetName)
        ship_action.AssetName = myAsset
        ship_action.AssetName = "Galleon HVU"
        ship_action.weapon = "Chainshot_System"
        # ship_action: ShipActionPb = ShipActionPb()
        # ship_action.TargetId = 2
        # ship_action.AssetName = "Galleon HVU"
        # ship_action.weapon = "Chainshot_System"

        # As stated, shipActions go into the OutputPb as a list of ShipActionPbs
        output_message.actions.append(ship_action)
        return output_message

    # Function to print state information and provide syntax examples for accessing protobuf messags
    def printStateInfo(self, msg: StatePb):
        pass
        # print("Time: " + str(msg.time))
        # print("Score: " + str(msg.score))

        # # Accessing asset fields.  Notice that is is simply the exact name as seen
        # # In PlannerProto.proto
        # print("Assets:")
        # for asset in msg.assets:
        #     print("1: " + str(asset.AssetName))
        #     print("2: " + str(asset.isHVU))
        #     print("3: " + str(asset.health))
        #     print("4: " + str(asset.PositionX))
        #     print("5: " + str(asset.PositionY))
        #     print("6: " + str(asset.PositionZ))
        #     print("7: " + str(asset.Lle))
        #     print("8: " + str(asset.weapons))
        # print("--------------------")

        # # Accessing track information is done the same way.
        # print("Tracks:")
        # for track in msg.Tracks:
        #     print("1: " + str(track.TrackId))
        #     print("2: " + str(track.ThreatId))
        #     print("3 " + str(track.ThreatRelationship))
        #     print("4: " + str(track.Lle))
        #     print("5: " + str(track.PositionX))
        #     print("6: " + str(track.PositionY))
        #     print("7: " + str(track.PositionZ))
        #     print("8: " + str(track.VelocityX))
        #     print("9 " + str(track.VelocityY))
        #     print("10: " + str(track.VelocityZ))
        # print("**********************************")



