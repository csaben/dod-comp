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
import numpy as np

class State(AiManager):
    # Constructor
    def __init__(self, publisher:Publisher):
        print("Constructing AI Manager")
        self.ai_pub = publisher
        self.memory = []
   
    # Is passed StatePb from Planner
    def receiveStatePb(self, msg:StatePb):
        # self.printStateAsDict(msg)
        output_message: OutputPb = OutputPb()
        output_message = self.generateState(msg)
        self.ai_pub.publish(output_message)

    # This method/message is used to notify of new scenarios/runs
    def receiveScenarioInitializedNotificationPb(self, msg:ScenarioInitializedNotificationPb):
        print("Scenario run: " + str(msg.sessionId))

    # This method/message is used to nofify that a scenario/run has ended
    def receiveScenarioConcludedNotificationPb(self, msg:ScenarioConcludedNotificationPb):
        print("Ended Run: " + str(msg.sessionId) + " with score: " + str(msg.score))
    # Example function for building OutputPbs, returns OutputPb

    def generateState(self, msg:StatePb):
        json_dict = self.cleanState(msg)
        # (Shane) Append state as dictionary to output file
        with open("outputStates.txt", "a") as file:
            file.write(json.dumps(json_dict)+"\n")
        file.close()

        #bunch of if thens to decide what state we are in

        #idle
        if 'Tracks' not in json_dict:
            print("idle")
            output_message: OutputPb = OutputPb()
            return output_message
        else:
            origin = self.calculateOrigin(json_dict['assets'])
            #execution order is a list of tuples (time, id)
            execution_order = self.calculateExecutionOrder(json_dict['Tracks'], origin)
            output_message: OutputPb = OutputPb()

            execution_order = [enemy for enemy in execution_order if enemy[1] not in self.memory]
            print(execution_order)
            #make sure its still sorted (it should be lol)
            execution_order = sorted(execution_order, key=lambda x: x[0])

            for missle in execution_order:
                ship_action: ShipActionPb = ShipActionPb()
                #set the target id to the missle id
                ship_action.TargetId = missle[1]
                ship_action.weapon, ship_action.AssetName \
                        = self.whoShootsFirst(json_dict['assets'], missle[1])

                output_message.actions.append(ship_action)
            return output_message


    def whoShootsFirst(self, assets, target_id)-> tuple:
        '''
        (
        ship_action.AssetName = "Galleon HVU"
        ship_action.weapon = "Chainshot_System"
        )
        '''
        for asset in assets:
            if asset['health'] == -1:
                pass
            else:
                for weapon in asset['weapons']:
                    print(weapon)
                    #BUG: quantity sometimes not given by the planner
                    try:
                        if weapon['WeaponState'] == 'Ready' and weapon["Quantity"] >0:
                            print(target_id)
                            self.memory.append(target_id)
                            print("memory: ", self.memory)
                            # print(weapon['SystemName'], asset['AssetName'])
                            # with open("./issue1.json", 'a') as f:
                            #     f.write(str(target_id))
                            #     f.write(str(self.memory))
                            # import sys
                            # sys.exit()

                            return  weapon['SystemName'], asset['AssetName']
                        else:
                            pass
                    except Exception as e:
                        pass


    def calculateExecutionOrder(self, missle_list, origin) -> list: #of tuples
        # t = (s/m) * m = s
        intercept_times=[]
        #get position, divide by speed, sort by lowest time to intercept
        #exclude elevation for now
        for missle in missle_list:
            #easier to read
            x = missle['PositionX']
            y = missle['PositionY']
            vx = missle['VelocityX']
            vy = missle['VelocityY']

            #distance eqn
            d = np.sqrt((x-origin[0])**2 + (y-origin[1])**2)
            #velocity eqn
            dv = np.sqrt(vx**2 + vy**2)

            #time to intercept
            t = d/dv
            intercept_times.append((t, missle['TrackId']))

        return sorted(intercept_times, key=lambda x: x[0])

    def calculateOrigin(self, asset_list) -> np.ndarray:
        origin = np.array([0,0], dtype=float)
        for asset in asset_list:
            if asset['health'] ==-1:
                pass
            else:
                # print(asset)
                origin+= np.array([asset['PositionX'], asset['PositionY']])
        return origin/(len(asset_list)-1)











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




