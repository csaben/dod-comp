# To ensure proper functionality, ensure these versions are used for protobuf and pyzmq
# pip install protobuf==3.20.0
# pip install pyzmq==24.0.0
# Developed on python 3.10.9

from publisher import Publisher
from subscriber import Subscriber
from AiManager import AiManager
from Ai import heuristic_agent

if __name__ == '__main__':
    print("Initializing AI client")
    
    #Initialize Publisher
    publisher = Publisher()

    #Initialize Subscriber
    subscriber = Subscriber()

    #Initialize AiManager
    # ai_manager = AiManager(publisher)
    ai_manager = heuristic_agent(publisher)

    #Register subscriber functions of Ai manager and begin listening for messages
    subscriber.registerSubscribers(ai_manager)
    subscriber.startSubscriber()
