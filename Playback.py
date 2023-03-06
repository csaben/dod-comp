from PlannerProto_pb2 import ScenarioConcludedNotificationPb, \
    ScenarioInitializedNotificationPb  # Scenario start/end notifications
from PlannerProto_pb2 import ErrorPb  # Error messsage if scenario fails
from PlannerProto_pb2 import StatePb, AssetPb, TrackPb  # Simulation state information
from PlannerProto_pb2 import OutputPb, ShipActionPb, WeaponPb
from publisher import Publisher
from AiManager import AiManager
from google.protobuf.json_format import MessageToDict, ParseDict
import PlannerProto_pb2

import os
import sys
import time
import pprint
from Strategy import Strategy
import glob
import fire

#https://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python#:~:text=import%20re%0Anumbers%20%3D%20re.compile(r%27(%5Cd%2B)%27)%0Adef%20numericalSort(value)%3A%0A%20%20%20%20parts%20%3D%20numbers.split(value)%0A%20%20%20%20parts%5B1%3A%3A2%5D%20%3D%20map(int%2C%20parts%5B1%3A%3A2%5D)%0A%20%20%20%20return%20parts%0A%0A%20for%20infile%20in%20sorted(glob.glob(%27*.txt%27)%2C%20key%3DnumericalSort)%3A%0A%20%20%20%20print%20%22Current%20File%20Being%20Processed%20is%3A%20%22%20%2B%20infile
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def main(strategy:str, state:str):
    if state == "all":
        for file in sorted(glob.glob(f"sample_games/{strategy}/*.bin"), key=numericalSort):
            print(f"Current File Being Processed is: {file}")
            #print each msg from that game continuously
    else:    
        state = int(state)
        with open(f"sample_games/{strategy}/state_pb_{state}.bin", "rb") as f:
            msg = PlannerProto_pb2.Message()
            msg.ParseFromString(f.read())
            print(msg)

if __name__ == '__main__':
    fire.Fire(main)

