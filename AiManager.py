#Imports
from PlannerProto_pb2 import ScenarioConcludedNotificationPb, ScenarioInitializedNotificationPb     #Scenario start/end notifications
from PlannerProto_pb2 import ErrorPb                                            #Error messsage if scenario fails
from PlannerProto_pb2 import StatePb, AssetPb, TrackPb                          #Simulation state information
from PlannerProto_pb2 import OutputPb, ShipActionPb,  WeaponPb
from publisher import Publisher

# This class is the center of action for this example client.  Its has the required functionality 
# to receive data from the Planner and send actions back.  Developed AIs can be written directly in here or
# this class could be used toolbox that a more complex AI classes reference.

# The word "receive" is protected in this class and should NOT be used in function names
# "receive" is used to notify the subscriber that "this method wants to receive a proto message"

# The second part of the function name is the type of proto message it wants to receive, thus proto
# message names are also protected
class AiManager:
    # Constructor
    def __init__(self, publisher:Publisher):
        print("Constructing AI Manager")
        self.ai_pub = publisher
   
    # Is passed StatePb from Planner
    def receiveStatePb(self, msg:StatePb):

        # Call function to print StatePb information
        self.printStateInfo(msg)

        # Call function to show example of building an action
        output_message = self.createActions(msg)
        print(output_message)

        # To advance in step mode, its required to return an OutputPb
        self.ai_pub.publish(output_message)
        #self.ai_pub.publish(OutputPb())

    # This method/message is used to notify of new scenarios/runs
    def receiveScenarioInitializedNotificationPb(self, msg:ScenarioInitializedNotificationPb):
        print("Scenario run: " + str(msg.sessionId))

    # This method/message is used to nofify that a scenario/run has ended
    def receiveScenarioConcludedNotificationPb(self, msg:ScenarioConcludedNotificationPb):
        print("Ended Run: " + str(msg.sessionId) + " with score: " + str(msg.score))


    #this is where the magic happens
    #TODO: this is where you would make a function that will be scraped by the planner
    """

    assume a model was initialized in the init

    def generateActions(self, msg:StatePb):
        output_message: OutputPb = OutputPb()
        ship_action: ShipActionPb = ShipActionPb()

        #hypothetically you can read msg apriori and only use specific models 
        #to generate actions for specific ships.

        out = model.predict(msg)
        ship_action.TargetId = out[0]
        ship_action.AssetName = out[1]
        ship_action.weapon = out[2]

        output_message.actions.append(ship_action)
        return output_message
    """
    # Example function for building OutputPbs, returns OutputPb
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
        """
        TLDR; i need to parse my assets and use a ship I actually have to shoot with
        (log files indicate the "Galleon HVU" is a ship I don't have)

        similar vibe for choosing targets

        but assuming I can simply check my asset, check my hostiles and update my target to be shot
        with an asset and its gun then I have atleast know how to shoot things and we can begin to
        actually divide and conquer with parsing msg's into output to be used to generate model
        outputs (i suggest a basic clf with output indices being actions we softmax based on input
        of kinematics or id but the ids represent a given gun or asset if there is a discrete
        number of assets so we will ultimately have a multi clf problem)

        i need to collect some game instances to actually make said clf pipeline
        """
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
        print("Time: " + str(msg.time))
        print("Score: " + str(msg.score))

        # Accessing asset fields.  Notice that is is simply the exact name as seen 
        # In PlannerProto.proto
        print("Assets:")
        for asset in msg.assets:
            print("1: " + str(asset.AssetName))
            print("2: " + str(asset.isHVU))
            print("3: " + str(asset.health))
            print("4: " + str(asset.PositionX))
            print("5: " + str(asset.PositionY))
            print("6: " + str(asset.PositionZ))
            print("7: " + str(asset.Lle))
            print("8: " + str(asset.weapons))
        print("--------------------")

        # Accessing track information is done the same way.  
        print("Tracks:")
        for track in msg.Tracks:
            print("1: " + str(track.TrackId))
            print("2: " + str(track.ThreatId))
            print("3 " + str(track.ThreatRelationship))
            print("4: " + str(track.Lle))
            print("5: " + str(track.PositionX))
            print("6: " + str(track.PositionY))
            print("7: " + str(track.PositionZ))
            print("8: " + str(track.VelocityX))
            print("9 " + str(track.VelocityY))
            print("10: " + str(track.VelocityZ))
        print("**********************************")



