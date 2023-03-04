#Imports
import math
import this

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
        self.targetedShips = {}

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

        #with open(self.filepath, "a") as file:
        #   file.write(json.dumps(json_dict) + "\n")

        # idle
        if 'Tracks' not in json_dict:
            print("idle")
            output_message: OutputPb = OutputPb()
            return output_message
        else:
            origin = self.calculateOrigin(json_dict['assets'])
            targets = self.calculateWhoTargeted(json_dict['Tracks'], json_dict['assets'])
            #now we have a list of tuples assetname, trackid, health

            #asset,
            if len(targets) > 0 and targets[0][0] !="nothing":
                print(f"dict: {self.targetedShips}")
                for tuple in targets:
                    # [(shipname, targetid, health),..]
                    if tuple[0] in self.targetedShips.keys(): #tuple[0]==shipname
                        if tuple[1] not in self.targetedShips[tuple[0]]: #tuple[1]==targetid
                            #values==[...],val
                            self.targetedShips[tuple[0]].append(tuple[1])
                            #{ assetname: [targetid, ...], health, ...}
                        # asset, TrackId, asset health
                    else:
                        self.targetedShips.update({tuple[0]:[tuple[1]]})
                        # self.targetedShips[tuple]=value

            #one gross mess later
            #we have a dict {assetname:[ids targeting it] ,....}
            print("targeted ships "+str(self.targetedShips))
            execution_order = self.calculateProtectionOrder()

            #just in case, make your list only contain unique values
            print("+"*50)
            print(execution_order)
            execution_order = list(set(execution_order))
            print(execution_order)
            print("+"*50)

            output_message: OutputPb = OutputPb()

            if len(self.memory) == 30:
                self.memory = []
                self.ifYouShootShutUp = []

            ####clark needs to make time to die callable

            ####st
            # either time to die exec order
            #        value order
            ######

            ###CHANGE WITHIN HERE #######
            for targetId in execution_order:
                ship_action: ShipActionPb = ShipActionPb()
                # set the target id to the missle id
                ship_action.TargetId = targetId
                #ship_action.AssetName = "HVU_Galleon_0"
                #ship_action.weapon = "Chainshot_System"
                #output_message.actions.append(ship_action)

                # ship, weapon, target
                ship_action.AssetName, ship_action_weapon, ship_action.TargetId \
                    = self.whoShootsFirst(json_dict['assets'], json_dict['Tracks'], targetId)
                #WE WANT A HYDRA. need a new way to assign weapons to targets

                if ship_action.weapon != "":
                    output_message.actions.append(ship_action)



                else:
                    self.ifYouShootShutUp = []
                    return output_message
            #################################
            # print("printing execution order")
            # print(execution_order)
            # print("printing memory")
            # print(self.memory)
            # self.ifYouShootShutUp = []
            # print("printing output message")
            # print(output_message.actions)

        return output_message


    def calculateProtectionOrder(self):
        executionOrder = []

        #sort by HVU
        for targetedShip in self.targetedShips.keys():
            if 'HVU' in targetedShip:
                executionOrder.extend(self.targetedShips[targetedShip])
        for targetedShip in self.targetedShips.keys():
            if 'HVU' not in self.targetedShips:
                executionOrder.extend(self.targetedShips[targetedShip])
        return executionOrder



    def calculateWhoTargeted(self, missile_list, assets):

        targets = []

        for missile in missile_list:
            if '>' in missile['ThreatId']:
                pass
            else:
                #distanceList = []
                howClose = []
                for asset in assets:

                    if asset['health'] == -1:
                        pass
                    else:




                        print(f"Missile {missile['TrackId']} Position {missile['TrackId']} X {missile['PositionX']} Missile Y {missile['PositionY']}")
                        print(f"Missile Velocity {missile['TrackId']} XV {missile['VelocityX']} Missile YV {missile['VelocityY']}")
                        print(f"Ship {asset['AssetName']} is at {asset['PositionX']}, {asset['PositionY']}")

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
                        howClose.append((asset['AssetName'], missile['TrackId'], asset['health'], distance))
                        #[(shipname, trackid, health, distance),...]

                    #sort how close ship comes to particular missile
                howClose = sorted(howClose, key=lambda x: x[3])
                #sort based on distance

                if len(howClose)>0:
                    targetedShip = howClose[0][0]
                #     targets.append(targetedShip)
                                        #assset name      trackid          health
                    targetedShip_info = (howClose[0][0], howClose[0][1], howClose[0][2])
                else:
                    targetedShip_info=("nothing")

                targets.append(targetedShip_info)

            #print(f"Targeted List: {targets}")
        #return a list of tuples that have the asset name , trackid and health
        return targets









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

    def whoShootsFirst(self, assets, missiles, target_id) -> tuple:
        # target_weapon_pair = []
        # shootOrder = []
        closest_asset=[]
        for asset in assets:
            if asset["health"]==-1:
                pass
            else:

                if "Chainshot_System" in asset['weapons'][1]['SystemName'] and "Quantity" in assets[1]["weapons"][1].keys():
                    for missile in missiles:
                        x1 = asset['PositionX']
                        y1 = asset['PositionY']
                        x2 = missile['PositionX']
                        y2 = missile['PositionY']

                        d = math.sqrt((x1-x2)**2+(y1-y2)**2)
                        closest_asset.append((d, asset["AssetName"], asset['weapons'][1]['SystemName'], target_id))

                elif "Cannon_System" in asset['weapons'][0]['SystemName']  and "Quantity" in assets[0]["weapons"][0].keys():
                    for missile in missiles:
                        x1 = asset['PositionX']
                        y1 = asset['PositionY']
                        x2 = missile['PositionX']
                        y2 = missile['PositionY']

                        d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                        closest_asset.append((d, asset["AssetName"], asset['weapons'][0]['SystemName'], target_id))

                else:
                    pass

        sort = sorted(closest_asset,key= lambda x: x[0])
        if len(sort)==0:
            return "", "",-1
        else:
            return sort[0][1], sort[0][2], sort[0][3]






        return self.find_value(assets, "health", -1, "Quantity", target_id)


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



