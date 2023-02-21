from AiManager import *
from PlannerProto_pb2 import ScenarioConcludedNotificationPb, ScenarioInitializedNotificationPb     #Scenario start/end notifications
from PlannerProto_pb2 import ErrorPb                                            #Error messsage if scenario fails
from PlannerProto_pb2 import StatePb, AssetPb, TrackPb                          #Simulation state information
from PlannerProto_pb2 import OutputPb, ShipActionPb,  WeaponPb
from publisher import Publisher
from AiManager import AiManager
import math
import numpy as np
from enum import Enum
from collections import OrderedDict

#customizing
import sys

class Availablitity(Enum):
    AVAILABLE = "Available",
    UNAVAILABLE = "Unavailable"


class Toy(AiManager):
    def __init__(self, publisher:Publisher):
        self.ai_pub = publisher
        self.i = 0
        self.roster = OrderedDict()
        self.state_memory = []
        self.fired_at = []

    #receiveStatePb predefined
    def receiveStatePb(self, msg: StatePb):
        state = self.generateState(msg)
        #is there a state to act on
        if state:
            self.createActions(msg, state)
            print(self.fired_at)
        
        else:
            return None

    #createActions predefined
    def createActions(self, msg: StatePb, state):
        # assign target to a given asset
        output_message: OutputPb = OutputPb
        for id in state:
            if id in self.fired_at:
                continue
            else:
                incoming = True
                while incoming:
                        ship_action: ShipActionPb = ShipActionPb()
                        ship_action.TargetId = id
                        try:
                            #cycle through roster and decide on asset and weapon
                            free = self.weaponSelector()
                            if free == None:
                                incoming = False
                            ship_action.AssetName, ship_action.weapon = free
                            output_message.actions.append(ship_action)
                            self.fired_at.append(id)
                        except Exception as e:
                            print(e)
                        
        return output_message

    def weaponSelector(self):
        available = {}
        for asset in self.roster :
            for weap in asset.weapons:
                if weap.WeaponState() == Availablitity.AVAILABLE:
                    available[weap.AssetName] = weap.SystemName
                else:
                    continue
        return next(iter(available.items()))

    def generateState(self, msg:StatePb):
        # Assume you have a protobuf message object called "message"

        # Get a list of all fields in the message
        for idx, ship in enumerate(msg.assets):
            self.parseShip(idx, ship)

        # if enemy ships are present generate a state
        if msg.Tracks:
            ttd_list = {}
            for track in msg.Tracks:
                ttd = self.calculations(track)
                ttd_list[track.TrackId] = ttd
            sorted_ttd = dict(sorted(ttd_list.items(), key=lambda x: x[1]))
            #simply will to ttd as our "state"
            return sorted_ttd

        # else nothing to do
        else:
            return None
        

    def parseShip(self, idx, ship):
        name = ship.AssetName
        #update the ship information for agent memory
        self.roster[idx] = ship
        self.roster.popitem(last=False)

    def calculations(self, track):
            x, y, z = track.PositionX, track.PositionY, track.PositionZ
            distance_from_origin = math.sqrt(x**2 + y**2 + z**2)
            time_to_die =  distance_from_origin / np.mean(np.array([track.VelocityX, track.VelocityY, track.VelocityZ]))
            # code currently breaks somtime during or/and after computing/using ttd
            return time_to_die

