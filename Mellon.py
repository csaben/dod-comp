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
import heapq

class Mellon(AiManager):
    # Constructor
    def __init__(self, publisher: Publisher):
        print("Constructing AI Manager")
        self.ai_pub = publisher
        self.targetedAssetsList = []
        #if you continuously update your targeted assets with all of the new state info
        #you will probably begin shooting twice at the same missles; you should probably
        #double check new targeted assets with trackids aren't already in firing (TODO)
        self.inventory = []
        self.firing = []

    # Is passed StatePb from Planner
    def receiveStatePb(self, msg: StatePb):
        # self.printStateAsDict(msg)

        #update inventory
        for asset in msg.assets:
            if "REFERENCE" not in asset.AssetName:
                self.updateInventory(asset)

        output_message: OutputPb = OutputPb()
        output_message = self.act(msg)
        self.ai_pub.publish(output_message)

    # This method/message is used to notify of new scenarios/runs
    def receiveScenarioInitializedNotificationPb(self, msg: ScenarioInitializedNotificationPb):
        print("Scenario run: " + str(msg.sessionId))

    # This method/message is used to nofify that a scenario/run has ended
    def receiveScenarioConcludedNotificationPb(self, msg: ScenarioConcludedNotificationPb):
        self.inventory = []
        self.targetedAssetsList = []
        print("Ended Run: " + str(msg.sessionId) + " with score: " + str(msg.score))

    # Example function for building OutputPbs, returns OutputPb
    def act(self, msg: StatePb):
        output_message: OutputPb = OutputPb()
        # idle
        if msg.Tracks:
            # spawn emptyy output message
            # collect convenient mapping info
            trackMap = self.trackMap(msg)# def assetMap(self, msg: StatePb) -> AssetPb:
            assetMap = self.assetMap(msg)# def trackMap(self, msg: StatePb)-> TrackPb:

            # grab the likely targeted assets at this time
            self.targetedAssetsList = self.targetedAssets(msg, trackMap, assetMap)# def targetedAsset(self, msg: StatePb, trackMap: dict, assetMap: dict) -> dict:
            #now we have the following DS
            #[ (angle_between(angleAsset, angleMissle), asset, track, self.timeToDie(track, asset)), ...]

            # I am able to kill things, HVU_Galleon_5 has no ammo lol
            if msg.time>4 and msg.time<12:
                ship_action : ShipActionPb = ShipActionPb()
                ship_action.TargetId = 7
                ship_action.weapon = "Cannon_System"
                ship_action.AssetName = "Galleon_4"
                output_message.actions.append(ship_action)
                # print(ship_action)
                # print(output_message)
                return output_message

            # def hvu_heap(self, msg: StatePb, trackMap: dict, assetMap: dict) -> dict:
            hvu_heap = self.hvu_heap(msg, trackMap, assetMap)
            # BACK IN THE ACT FUNCTION
            if len(hvu_heap)==0:
                #go on to galleon heap
                pass
            else:
                # given whats left of the heap, decide if given how many galleons we have if strategy
                # needs to hard switch to protect all galleon's with closest strategy to maximize points
                # i.e. do a max point scenario calculation for current state and compare with hard switch
                # def pointsCalulation  X 2
                # def pointsCalculation(policy="hvu")
                # def pointsCalculation(policy="galleon")

                #handle score calculations and decide if we are updating policy and switching to
                #shoot closest (in this case I'm thinking we would just clear the shooting instance
                #variable and pass an un-mutilated current state inventory to heapGalleons to use)

                #if you switch strategy, just clear the ship_actions instance variable + pass an
                # un-mutilated current state inventory to heapGalleons to use. update a policy instance
                # variable such that next state we automatically jump to the heapGalleons strategy
                # (TODO)

                # def galleon_heap
                pass

            # if you don't switch strategy, still shoot with inventory that can't possibly help
            # remaining hvu's to protect the heap of galleons

            # def galleon_heap_with_unhelpful_defenders (make a 2nd inventory fn function to save
            # missles that just can't make it to save priority targets to be accessible for galleon
            # a different galleon heap low priority function) TODO
            
            #now we enter into the firing phase
            #[(assetName, systemName, targetId, scheduledFireTime),..]
            for wta in self.firing:
                if wta[-1] <=msg.time:
                    ship_action : ShipActionPb = ShipActionPb()
                    ship_action.TargetId = wta[2]
                    ship_action.weapon = wta[1]
                    ship_action.AssetName = wta[0]
                    output_message.actions.append(ship_action)
            
            return output_message


        else:
            print("idle")
            return output_message

    
    def pointsCalculation(self, policy):
        #use knowledge of points per shot, hit, and death of given ship and weapon to determine
        #current expected points of the game

        # if given hvu==policy:
            # calculate points based on current inventory and current state of the game

        # if given galleon==policy:
            # calculate points based on disregarding hvu and protecting rest

        pass


    #the heap aspect is rendered obsolete if I save ship_actions from "saveable" function
    def hvu_heap(self, msg: StatePb, trackMap: dict, assetMap: dict) -> dict:
        # make a hvu heap based on hvu calculation
        hvu_heap = []
        health_based = []
        #grab the hvus
        for asset in assetMap.values():
            if "HVU" in asset.AssetName:
                health_based.append((asset, asset.health))
        #compare health of the hvus
        health_based = sorted(health_based, key=lambda x: x[1])
        #check if hvu is targeted (notice this is missing the possibility of redirect hit, but is
        # fixed when we overwrite our targetedAssets info with the redirected info)

        ttd_based = []
        for asset in health_based:
            # for each occurence of the asset in targetedAssets, add to unsorted arry to be sorted
            # time to die prior to adding to the heap
            # [(angle_between(v1, v2), asset,
            #                             track, self.timeToDie(track, asset)))]
            for tup_asset in self.targetedAssetsList:
                print(self.targetedAssetsList)
                print(tup_asset)
                print(asset[0])
                if tup_asset[1].AssetName == asset[0].AssetName:
                    #[ (angle_between(angleAsset, angleMissle), asset, track, self.timeToDie(track, asset)), ...]
                    ttd_based.append((asset[0], tup_asset[3], tup_asset[2].TrackId, tup_asset))
                    #[(asset, ttd),..]
            ttd_based = sorted(ttd_based, key=lambda x: x[1])

        for ttd_asset in ttd_based:
            #[(asset, ttd, targetid),..]
            # check if saveable given current inventory for each hvu in order  (loop)
            # def saveable
            # if returned false then pop the hvu from the heap (i dont really see the need to hold onto
            # the heap variable though if I recalculate at each step)
            if self.savable(ttd_asset[0],ttd_asset[1], msg.time, ttd_asset[2], ttd_asset[3]):
                continue
            else:
                #pop the asset from the heap
                ttd_based.remove(ttd_asset)
                #calculate redirected missle (TODO)
                # if not saveable, calculate the simulateRedirectedMissle and update targetedAssets/ create
                # a special instance variable that stores the redirected missles and automatically updates
                # the original instance variable after every new state message is received

        return sorted(ttd_based, key=lambda x: x[1])
        # return the "hvuheap"
        # return heapq.heapify(ttd_based)
        

    def heapGalleons(self, msg: StatePb, trackMap: dict, assetMap: dict) -> dict:
        # sort by health, sort by time to die (pretty similar to hvu_heap)

        # check if saveable for each asset in order (loop)

        # if saveable and inventory contains enough ammo of the correct quality, add to instance
        # variable of ship_actions (if not saveable do a simulation of the redirected missle and
        # add to targetedAssets instance variable). also default shoot ammo if you only
        # have 3 states alive left and you have 3 ammo for a given weapon (this is 
        # the base case, at this point you need to shoot but higher order logic will only
        # recognize this in the next state; big flaw)

        # return the heapGalleons (not strictly necessary)
        pass

    def assignWeapons(self):
        # def inventory()
        # ... update inventory instance variable after you assign
        # ... set inventory instance to be reset at end of state message

        pass

    def savable(self, asset: AssetPb, ttd: float, time: float, targetid: int, targetedAsset:tuple) -> bool:
        #[(assetName, systemName, Quantity, (
        x,y = asset.PositionX, asset.PositionY
        # calculate if any of the weaponn systems in inventory are capable
        #make a capable of protecting list and right before shooting everything compare
        # bw all other things that need to be shot for to be protected, but for now just
        # assign locally and pop from inventory (choose to shoot with whichever will
        # barely make it)
        capable_defenders=[]
        # weapon_info =[(asset.AssetName, weapon.SystemName, weapon.Quantity, (asset.PositionX, asset.PositionY),ammo[weapon.SystemName])),..]
        # for all occurences of the asset in targetedAssets check if we have enough quality ammo now and
        # later to save (if ttd of asset is less then ttd of defending missles == quality ammo)
        for weapon_info in self.inventory:
            # calculate potential defense ttd
            x1,y1 = weapon_info[3]
            v = weapon_info[-1]

            # distance eqn
            d = np.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
            # time to intercept
            t = d / v

            #if my missle could hit my own ship faster than I use i consider it fast enough
            if t<ttd:
                capable_defenders.append((weapon_info,t))

        if len(capable_defenders)>0:
            #reverse =True; get the furthest away missle
            shooter= sorted(capable_defenders, key=lambda x: x[1], reverse=True)[0]
            #update inventory
            self.inventory.remove(shooter[0])
            #update the whos firing instance variable
            #[(assetName, systemName, targetId, scheduledFireTime),..]
            #if t=10 and ttd=20 then shoot at 28, so 10+20-2
            self.firing.append((shooter[0][0], shooter[0][1], targetid, time+ttd-6))#in main loop check if time <= the firing time and shoot if true
            #update the targetedAssets instance variable
            #remove key value targetid: targetedAsset
            print(self.targetedAssetsList)
            # self.targetedAssetsList.pop(targetid)
            self.targetedAssetsList.remove(targetedAsset)
            #[ (angle_between(angleAsset, angleMissle), asset, track, self.timeToDie(track, asset)), ...]
            return True

        else:
        # if not saveable, return false
            return False

    def updateInventory(self, asset: AssetPb) -> dict:
        ammo={"Cannon_System":972, "Chainshot_System":343}
        # doesn't need to be dict, should make taking inventory of
        weapon_info = []
        for weapon in asset.weapons:
            # we care about the (weaponname, (xyz), and speed)
            if weapon.Quantity:
                weapon_info.append((asset.AssetName, weapon.SystemName, weapon.Quantity, (asset.PositionX, asset.PositionY),ammo[weapon.SystemName]))

        for info in weapon_info:
            self.inventory.append(info)
        # available weapons trivial. need asset name, weapons, ammo, time to die
        return weapon_info #not strictly necessary


    def simulateRedirectedMissle(self, missle, assetMap, time):
        # you have a deterministic expectation of when you allow a ship to die, so 
        # calculate the new target based on location of missle at new time step
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
        missleLikelihoods=[]
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
                missleLikelihoods.append(degreeOfParallelism[0])
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

"""
not currently in use:

    # missleLikelihoods[track.TrackId] = degreeOfParallelism[0]
    # return missleLikelihoods
    # degreeOfParallelism.append((angle_between(unitAsset, unitMissle), track.TrackId, asset.AssetName))
    def priorityMissles(self, missleLikelihoods, assetMap):
        # HVU should always be prioritized

        # based on who is targeted, how much ammo we have, how long until arrival

        # firing policies should vary, but generally waiting until the last second is a good idea
        # or until angle is effectively 0
        pass

    def policy(self):
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
        pass


"""



