from AiManager import *


#TODO: should inititialize a set of models, heuristic fn for testing, or a model manager

#Imports
from PlannerProto_pb2 import ScenarioConcludedNotificationPb, ScenarioInitializedNotificationPb     #Scenario start/end notifications
from PlannerProto_pb2 import ErrorPb                                            #Error messsage if scenario fails
from PlannerProto_pb2 import StatePb, AssetPb, TrackPb                          #Simulation state information
from PlannerProto_pb2 import OutputPb, ShipActionPb,  WeaponPb
from publisher import Publisher
from AiManager import AiManager


class heuristic_agent(AiManager):
    # Constructor
    def __init__(self, publisher:Publisher):
        print("Constructing AI Manager")
        self.ai_pub = publisher
        self.ships=None
        self.assets={}
        self.tracks={}
        self.targets=[]
        self.timer = 0

    def receiveStatePb(self, msg:StatePb):

        # Call function to print StatePb information
        self.printStateInfo(msg)

        # Call function to show example of building an action
        output_message = self.createActions(msg)
        print("Time: " + str(msg.time))
        self.timer = msg.time
        # print(output_message) 
        print("Score: " + str(msg.score))


        # To advance in step mode, its required to return an OutputPb
        self.ai_pub.publish(output_message)
        #self.ai_pub.publish(OutputPb())
        
        with open("./sample.txt", 'w') as f:
            f.write("assets")
            f.write("\n")
            f.write(str(self.assets))
            f.write("tracked targets")
            f.write('\n')
            f.write(str(self.tracks))

    def createActions(self, msg:StatePb):
        # # ShipActionPb's go into an OutputPb message
        output_message: OutputPb = OutputPb()

        # # ShipActionPb's are built using the same sytax as the printStateInfo function
        # ship_action: ShipActionPb = ShipActionPb()
        
        for asset in msg.assets:
            # print(str(asset.AssetName))
            self.assets[str(asset.AssetName)] = [
                str(asset.isHVU),
                str(asset.weapons)
            ]
        
        for track in msg.Tracks:
            self.tracks[int(track.TrackId)] = [
                str(track.TrackId),
                str(track.ThreatId),
                str(track.PositionX),
                str(track.PositionY),
                str(track.PositionZ),
                str(track.VelocityX),
                str(track.VelocityY),
                str(track.VelocityZ)
            ]

        #heurist defense strategy (launch all your assets at the first id you get)
        self.targets = list(self.tracks.keys())       
        #you have one less set of cannons then you think because one ship is teh reference ship

        self.assets.pop("Galleon_REFERENCE_SHIP")
        print(len(self.assets))
        ct=0
        shots = 0
        for (k,v) in self.assets.items():

            if self.timer>210 and shots==0:
                ship_action_1: ShipActionPb = ShipActionPb()
                ship_action_2: ShipActionPb = ShipActionPb()
                # print("+"*50)
                # print(ship_action_1)
                # print(ship_action_2)
                # print("+"*50)

                try:
                    print("=====================================")
                    ship_action_1.TargetId = self.targets[ct]
                    ship_action_1.AssetName = k
                    ship_action_1.weapon = "Cannon_System"
                    ct+=1
                except Exception as e:
                    print(e)

                try:
                    ship_action_2.TargetId = self.targets[ct]
                    ship_action_2.AssetName = k
                    ship_action_2.weapon = "Chainshot_System"
                    ct+=1
                except:
                    continue
                print(ship_action_1)
                print(ship_action_2)
                output_message.actions.append(ship_action_1)
                output_message.actions.append(ship_action_2)
                return output_message

            
        else:
            return output_message


        output_message.actions.append(ship_action_1)
        output_message.actions.append(ship_action_2)
        return output_message
        



    # Function to print state information and provide syntax examples for accessing protobuf messags
    def printStateInfo(self, msg:StatePb):
        #don't print anything for now
        pass



## Clark notes to self:


#the 0=bool, 1=str, 2=list
#list[name of weapon, quantity of ammo, string readiness,  ....] so 0=weapon_name and 3=weapon name, but every ship has these weapons so you can always assume there there and you only need to keep track of ammo and readiness to fire

    


# for asset in self.assets:
#     self.generateShipAction(asset)
    #so this indexing strategy does collect all the necessary info

    #AFTER THAT JUST GO RESEARCH WAYS TO MEANINGFUL STORE
    #THESE STATES SO THAT YOU CAN MAKE A MODEL THAT GIVEN A 
    #A STATE IT CAN GENERATE SOME OUTPUT THAT COULD be
    #interpolated into a asset-target pairing (more on this
    # after call with prashant and we can also look into
    # generating artifial data then too)

# with open("./sample.txt", 'w') as f:
#     f.write("assets")
#     f.write("\n")
#     f.write(str(self.assets))
#     f.write("tracked targets")
#     f.write('\n')
#     f.write(str(self.tracks))

# message AssetPb {
#   string AssetName = 1;                       // Name of the assets (AssetName in ShipActionPb)
#   bool isHVU = 2;                             // Whether or not this asset is a high value unit
#   int32 health = 3;                           // Total health of the asset
#   double PositionX = 4;                       // Relative position East (meters)
#   double PositionY = 5;                       // Relative position North (meters)
#   double PositionZ = 6;                       // Relative position Up (meters)
#   repeated double Lle = 7;                    // Latitude, longitude, elevation of asset
#   repeated WeaponPb weapons = 8;              // State of asset's deployers
# }
