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

class Mellon(AiManager):
    # Constructor
    def __init__(self, publisher: Publisher):
        print("Constructing AI Manager")
        self.ai_pub = publisher

    # Is passed StatePb from Planner
    def receiveStatePb(self, msg: StatePb):
        # self.printStateAsDict(msg)
        output_message: OutputPb = OutputPb()
        output_message = self.act(msg)
        self.ai_pub.publish(output_message)

    # This method/message is used to notify of new scenarios/runs
    def receiveScenarioInitializedNotificationPb(self, msg: ScenarioInitializedNotificationPb):
        print("Scenario run: " + str(msg.sessionId))

    # This method/message is used to nofify that a scenario/run has ended
    def receiveScenarioConcludedNotificationPb(self, msg: ScenarioConcludedNotificationPb):
        print("Ended Run: " + str(msg.sessionId) + " with score: " + str(msg.score))

    # Example function for building OutputPbs, returns OutputPb
    def act(self, msg: StatePb):
        # idle
        if msg.Tracks:
            # spawn emptyy output message
            # collect convenient mapping info
            trackMap = self.trackMap(msg)# def assetMap(self, msg: StatePb) -> AssetPb:
            assetMap = self.assetMap(msg)# def trackMap(self, msg: StatePb)-> TrackPb:

            # grab the likely targeted assets at this time
            targetedAssets = self.targetedAssets(msg, trackMap, assetMap)# def targetedAsset(self, msg: StatePb, trackMap: dict, assetMap: dict) -> dict:
            #now we have the following DS
            #[ (angle_between(angleAsset, angleMissle), asset, track, self.timeToDie(track, asset)), ...]

            #TEST AND SEE IN GUI IF WE ARE CORRECTLY DECIDING ON TARGETS
            #TRACKID7 ENEMY5
            if msg.time>5 and msg.time<12:
                # ship_action.TargetId = missle[1]
                # ship_action.weapon
                # ship_action.AssetName 
                # output_message.actions.append(ship_action)
                output_message: OutputPb = OutputPb()
                ship_action : ShipActionPb = ShipActionPb()
                ship_action.TargetId = 7
                # shipAction.weapon = "Cannon_System"
                ship_action.AssetName = "Galleon_1"
                ship_action.weapon = "Chainshot_System"
                output_message.actions.append(ship_action)
                return output_message 
                # for target_id, tuple_asset_info in targetedAssets.items():
                #     if tuple_asset_info[2] == "HVU_Galleon_5":
                #         # if there are targeted assets, then we need to act
                #         shipAction : ShipActionPb = ShipActionPb()
                #         # just assume you have ammo in galleon 0 and shoot it
                #         shipAction.AssetName = "HVU_Galleon_5"
                #         shipAction.weapon = "Cannon_System"
                #         shipAction.TargetId = tuple_asset_info[2].TrackId
                #         output_message.actions.append(ship_action)
                #     else:
                #         pass
            else:
                output_message: OutputPb = OutputPb()
                return output_message

            # determine if any action is needed currently (define a threat threshold, ties into
            # custom policy)

            # generate an output message based on a custom policy (protect HUV, wait as long as
            # possible, and fire when angle is effectively 0)


        else:
            print("idle")
            output_message: OutputPb = OutputPb()
            return output_message


    # def timeToDie(self, track: TrackPb, asset: AssetPb) -> float:
            return output_message

    def Inventory(self, asset: AssetPb) -> dict:
        # doesn't need to be dict, should make taking inventory of
        # available weapons trivial
        pass


    def simulateRedirectedMissle(self, missle, assetMap, time):
        # you have a deterministic expectation of when you allow a ship to die, so 
        # calculate the new target based on location of missle at new time step
        pass

    # missleLikelihoods[track.TrackId] = degreeOfParallelism[0]
    # return missleLikelihoods
    # degreeOfParallelism.append((angle_between(unitAsset, unitMissle), track.TrackId, asset.AssetName))
    def priorityMissles(self, missleLikelihoods, assetMap):
        # HVU should always be prioritized

        # based on who is targeted, how much ammo we have, how long until arrival

        # firing policies should vary, but generally waiting until the last second is a good idea
        # or until angle is effectively 0
        pass


    # helper function, in reality protobufs are much cleaner to work with
    def msgToDictionary(self, msg: StatePb) -> dict:
        # StatePb
        message = PlannerProto_pb2.StatePb()
        # serialize the message
        msg = msg.SerializeToString()
        message.ParseFromString(msg)
        message_dict = MessageToDict(message)
        return message_dict

    def trackMap(self, msg: StatePb)-> TrackPb:
        trackMap = {}
        for track in msg.Tracks:
            trackMap[track.TrackId] = track
        return trackMap

    def assetMap(self, msg: StatePb) -> AssetPb:
        assetMap, weaponMap = {}, {}
        for asset in msg.assets:
            assetMap[asset.AssetName] = asset
        return assetMap

    def targetedAssets(self, msg: StatePb, trackMap: dict, assetMap: dict) -> dict:
        #https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python#:~:text=Here%20is%20a%20function%20which%20will%20correctly%20handle%20these%20cases%3A
        #linear algebra, if a the vector of the x,y of asset and x,y of missle are parallel, then the missle is headed towards the asset
        missleLikelihoods={}
        #iterate through each track in the dict (TrackId is repeated information and used only for key)
        for track_id, track in trackMap.items():
            if track.ThreatRelationship == "Hostile":
                degreeOfParallelism = []
                for asset_name, asset in assetMap.items(): #need to not include reference ship
                    if asset.AssetName !="Galleon_REFERENCE_SHIP":
                        def unit_vector(vector):
                            """ Returns the unit vector of the vector.  """
                            return vector / np.linalg.norm(vector)

                        def angle_between(v1, v2):
                            """ Returns the angle in radians between vectors 'v1' and 'v2'"""
                            v1_u = unit_vector(v1)
                            v2_u = unit_vector(v2)
                            #np.arcos must receieve a value between -1 and 1, so we clip the value
                            angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
                            # >>> an gle_between((1, 0, 0), (1, 0, 0)) == 0.0

                            # we sort by smallest so using unit circle we make our angle between 0 and pi/2
                            if angle>0 and angle<np.pi/2:
                                angle = angle
                            elif angle>np.pi/2 and angle<np.pi:
                                angle = np.pi-angle
                            elif angle>np.pi and angle<3*np.pi/2:
                                angle =  angle-np.pi
                            elif angle>3*np.pi/2 and angle<2*np.pi:
                                angle = 2*np.pi-angle
                            elif angle==0:
                                angle = 0
                            elif angle==np.pi:
                                angle = np.pi
                            else:
                                sys.exit("Error in angle_between function")
                            return angle

                        #calculate the angle between each asset and a given missle
                        v1 = (asset.PositionX, asset.PositionY, 0)
                        v2 = (track.PositionX, track.PositionY, 0)
                        #store all information pertinent about a missle that will potentially hit you
                        # (speed up if you calculate time to die later)
                        degreeOfParallelism.append((angle_between(v1, v2), asset,
                                                    track, self.timeToDie(track, asset)))

                #get the smallest angle to be the most likely target
                if len(degreeOfParallelism) > 0:
                    degreeOfParallelism = sorted(degreeOfParallelism, key=lambda x: x[0])
                else:
                    return None
                missleLikelihoods[track.TrackId] = degreeOfParallelism[0]
        return missleLikelihoods

    def timeToDie(self, track: TrackPb, asset: AssetPb) -> float:
        #track position and velocity
        x1,y1 = track.PositionX, track.PositionY
        vx1, vy1 = track.VelocityX, track.VelocityY

        #asset position and velocity
        x2, y2 = asset.PositionX, asset.PositionY
        vx2, vy2 = 0, 0

        # distance eqn
        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        # velocity eqn
        dv = np.sqrt((vx1 - vx2) ** 2 + (vy1 - vy2) ** 2)
        # time to intercept
        t = d / dv

        return t

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



