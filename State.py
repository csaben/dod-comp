#Imports
from PlannerProto_pb2 import ScenarioConcludedNotificationPb, ScenarioInitializedNotificationPb     #Scenario start/end notifications
from PlannerProto_pb2 import ErrorPb                                            #Error messsage if scenario fails
from PlannerProto_pb2 import StatePb, AssetPb, TrackPb                          #Simulation state information
from PlannerProto_pb2 import OutputPb, ShipActionPb,  WeaponPb
from publisher import Publisher
from AiManager import AiManager
from google.protobuf.json_format import MessageToDict, ParseDict
import json
import PlannerProto_pb2

class State(AiManager):
    # Constructor
    def __init__(self, publisher:Publisher):
        print("Constructing AI Manager")
        self.ai_pub = publisher
   
    # Is passed StatePb from Planner
    def receiveStatePb(self, msg:StatePb):
        # self.printStateAsDict(msg)
        output_message: OutputPb = OutputPb()
        self.ai_pub.publish(output_message)
        # self.cleanState(msg)
        self.generateState(msg)

    # This method/message is used to notify of new scenarios/runs
    def receiveScenarioInitializedNotificationPb(self, msg:ScenarioInitializedNotificationPb):
        print("Scenario run: " + str(msg.sessionId))

    # This method/message is used to nofify that a scenario/run has ended
    def receiveScenarioConcludedNotificationPb(self, msg:ScenarioConcludedNotificationPb):
        print("Ended Run: " + str(msg.sessionId) + " with score: " + str(msg.score))
    # Example function for building OutputPbs, returns OutputPb

    def generateState(self, msg:StatePb):
        json_dict = self.cleanState(msg)

        #bunch of if thens to decide what state we are in

        #idle
        if 'Tracks' not in json_dict:
            with open("./dev.json", 'a') as f:
                f.write(json.dumps("idle"))
            print("idle")
        else:
            with open("./dev.json", 'a') as f:
                f.write(json.dumps(json_dict))
            print(json_dict['Tracks'])





    def cleanState(self, msg:StatePb):
        #StatePb
        message = PlannerProto_pb2.StatePb()
        #serialize the message
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


    def createActions(self, msg:StatePb):

        # ShipActionPb's go into an OutputPb message
        output_message: OutputPb = OutputPb()
        # print("**********************************")
        import time
        time.sleep(.25)
        # import sys
        # sys.exit()
        # ShipActionPb's are built using the same sytax as the printStateInfo function

        #sample shooting a ship heuristically
        ship_action: ShipActionPb = ShipActionPb()
        # for track in msg.Tracks:
        #     target = track.TrackId
        target = 1
        ship_action.TargetId = target
        #asset name
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
    def printStateInfo(self, msg:StatePb):
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




