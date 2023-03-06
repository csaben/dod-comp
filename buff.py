from PlannerProto_pb2 import ScenarioConcludedNotificationPb, \
    ScenarioInitializedNotificationPb  # Scenario start/end notifications
from PlannerProto_pb2 import ErrorPb  # Error messsage if scenario fails
from PlannerProto_pb2 import StatePb, AssetPb, TrackPb  # Simulation state information
from PlannerProto_pb2 import OutputPb, ShipActionPb, WeaponPb
from publisher import Publisher
from AiManager import AiManager
from google.protobuf.json_format import MessageToDict, ParseDict
import PlannerProto_pb2

import sys
import time
import pprint


class practice(AiManager):
    # Constructor
    def __init__(self, publisher: Publisher):
        print("Constructing AI Manager")
        self.ai_pub = publisher
        # self.fired_at = []

    def receiveScenarioConcludedNotificationPb(self, msg: ScenarioConcludedNotificationPb):
        print("Ended Run: " + str(msg.sessionId) + " with score: " + str(msg.score))
        # self.fired_at = []
        # self.fired_weapons = []

    # Is passed StatePb from Planner
    def receiveStatePb(self, msg: StatePb):
        output_message: OutputPb = OutputPb()
        # self.fired_weapons=[]

        # weapons_info=[]
        # for asset in msg.assets:
        #     for weapon in asset.weapons:
        #         weapons_info.append((weapon.SystemName, weapon.Quantity, weapon.WeaponState,
        #                              asset.AssetName))
        
        # print("Weapons Info: ")
        # print(weapons_info)

        # tracks_info=[]
        # for track in msg.Tracks:
        #     tracks_info.append((track.TrackId, track.ThreatId, track.ThreatRelationship))

        # available_weapons=[]
        # available_weapons = [(weapon[0],weapon[3]) for weapon in weapons_info if weapon[1] > 0]
        # print("Available Weapons: ")
        # print(available_weapons)

        # # shoot at enemy tracks
        # for track in tracks_info:
        #     if track[2]=="Hostile":
        #         ship_action = self.fire_at_it(weapon, track, available_weapons)
        #         if ship_action:
        #             output_message.actions.append(ship_action)

        # self.ai_pub.publish(output_message)

#    def fire_at_it(self, weapon, track, available_weapons):
#        for weapon in available_weapons:
#            if (weapon[0], weapon[1]) not in self.fired_weapons and track[0] not in self.fired_at:
#                ship_action: ShipActionPb = ShipActionPb()
#                ship_action.TargetId = track[0]
#                #remember you shot at this target
#                self.fired_at.append(track[0])
#                ship_action.AssetName = weapon[1]
#                ship_action.weapon = weapon[0]
#                self.fired_weapons.append((weapon[0], weapon[1]))
#                print("FIRED AT")
#                print(self.fired_at)
#                return ship_action
                

