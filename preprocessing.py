import numpy as np
import os
import pickle
import pathlib
from pathlib import Path
import json
from config import *
from State import State
import torch

from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logging.set_verbosity(40) #only show errors
from transformers import BertTokenizer, BertModel
# MODEL = BertModel.from_pretrained("bert-base-cased")
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")


def main():
    # 1. Load data
    data_dir = './ShaneOutput/'
    data_files = os.listdir(data_dir)
    paths_of_files = [os.path.join(data_dir, basename) for basename in data_files]

# open the file
    with open(paths_of_files[0], 'r', encoding='utf-8') as f:
        #load json string from json file
        data = f.read().replace('\n', ',')
        data = '['+data[:-1]+']'
        data = json.loads(data)

    # preprocess(data)
    # print(data)
    preprocess(data)

def preprocess(game):
    import sys
    # Extract values from dictionary

    #make a parent tensor
    parent_tensor = torch.tensor([], dtype=torch.float32)
    for data in game:
        #remove AssetName reference ship
        data['assets'] = [asset for asset in data['assets'] if asset['AssetName'] !=
                          'Galleon_REFERENCE_SHIP']

        time = data['time']
        try:
            score = data['score']
        except:
            score = 0


        asset_names = []
        healths = []
        positions = []
        lle = []
        weapons_quantities = []
        weapons_states = []

        for asset in data['assets']:
            asset_names.append(asset['AssetName'])
            healths.append(asset['health'])
            positions.append([asset['PositionX'], asset['PositionY']])
            lle.append(asset['Lle'])

            for weapon in asset['weapons']:
                try:
                    weapons_quantities.append(weapon['Quantity'])
                except:
                    weapons_quantities.append(0)
                weapons_states.append(weapon['WeaponState'])

        track_ids = []
        threat_ids = []
        threat_relationships = []
        track_positions = []

        try:
            for track in data['Tracks']:
                track_ids.append(track['TrackId'])
                threat_ids.append(track['ThreatId'])
                threat_relationships.append(track['ThreatRelationship'])
                track_positions.append([track['PositionX'], track['PositionY']])
        except:
            pass

        # Convert lists to numpy arrays
        asset_names = asset_names[:5] + [0] * (5 - len(asset_names))
        asset_names = np.array(asset_names)

        healths = np.array(healths)
        positions = np.array(positions)
        lle = np.array(lle)
        weapons_quantities = np.array(weapons_quantities)

        track_ids = track_ids[:30] + [0] * (30 - len(track_ids))
        track_ids = np.array(track_ids)

        # print(track_positions)
        track_positions = track_positions[:90] + [[0,0]] * (90 - len(track_positions))
        # print(track_positions)
        track_positions = np.array(track_positions)

        # Query library to get numerical values for asset names and threat ids

        #use tokenizer for when shots are being fired
        converter = lambda x: int("".join(map(str, x))) / 1e45
        asset_name_numbers = []
        threat_id_numbers = []
        for name in asset_names:
            if name in library:
                asset_name_numbers.append(library[name])
            else:
                name = converter(TOKENIZER.encode(name, add_special_tokens=False))
                asset_name_numbers.append(name)

        for threat in threat_ids:
            if threat in library:
                threat_id_numbers.append(library[threat])
            else:
                threat = converter(TOKENIZER.encode(threat, add_special_tokens=False))
                threat_id_numbers.append(threat)

        #add execution order label
        origin = State.calculateOrigin(data['assets'])
        try:
            execution_order = State.calculateExecutionOrder(data['Tracks'], origin)
            #pad for the rest if not 30
            # print(execution_order)
            execution_order = execution_order[:30] + [(-1,-1)] * (30 - len(execution_order))
            # print(execution_order)
        except:
            execution_order = [(-1,-1)] * 30
            # print(execution_order)

        # Create tensor
        tensor = np.array([
            time,
            score,
            *asset_name_numbers,
            *healths,
            *positions.flatten(),
            *lle.flatten(),
            *weapons_quantities,
            *track_ids,
            *threat_id_numbers,
            *track_positions.flatten(),
            *execution_order
        ])

        # Reshape tensor to desired shape
        tensor = tensor.reshape(1, -1)

        # Convert to pytorch tensor
        tensor = torch.tensor(tensor, dtype=torch.float32)




        # print(tensor.shape)
        #append tensor to parent tensor
        try:
            parent_tensor = torch.cat((parent_tensor, tensor), 0)
        except:
            pass
    print(parent_tensor.shape)

def prev(game):
    import sys
    # Extract values from dictionary

    #make a parent tensor
    parent_tensor = torch.tensor([], dtype=torch.float32)
    for data in game:
        #remove AssetName reference ship
        data['assets'] = [asset for asset in data['assets'] if asset['AssetName'] !=
                          'Galleon_REFERENCE_SHIP']

        time = data['time']
        try:
            score = data['score']
        except:
            score = 0


        asset_names = []
        healths = []
        positions = []
        lle = []
        weapons_quantities = []
        weapons_states = []

        for asset in data['assets']:
            asset_names.append(asset['AssetName'])
            healths.append(asset['health'])
            positions.append([asset['PositionX'], asset['PositionY']])
            lle.append(asset['Lle'])
            
            for weapon in asset['weapons']:
                try:
                    weapons_quantities.append(weapon['Quantity'])
                except:
                    weapons_quantities.append(0)
                weapons_states.append(weapon['WeaponState'])

        track_ids = []
        threat_ids = []
        threat_relationships = []
        track_positions = []

        try:
            for track in data['Tracks']:
                track_ids.append(track['TrackId'])
                threat_ids.append(track['ThreatId'])
                threat_relationships.append(track['ThreatRelationship'])
                track_positions.append([track['PositionX'], track['PositionY'], track['PositionZ']])
        except:
            pass

        # Convert lists to numpy arrays
        healths = np.array(healths)
        positions = np.array(positions)
        lle = np.array(lle)
        weapons_quantities = np.array(weapons_quantities)
        track_ids = np.array(track_ids)
        track_positions = np.array(track_positions)

        # Query library to get numerical values for asset names and threat ids

        #use tokenizer for when shots are being fired
        converter = lambda x: int("".join(map(str, x))) / 1e45
        asset_name_numbers = []
        threat_id_numbers = []
        for name in asset_names:
            if name in library:
                asset_name_numbers.append(library[name])
            else:
                name = converter(TOKENIZER.encode(name, add_special_tokens=False))
                asset_name_numbers.append(name)

        for threat in threat_ids:
            if threat in library:
                threat_id_numbers.append(library[threat])
            else:
                threat = converter(TOKENIZER.encode(threat, add_special_tokens=False))
                threat_id_numbers.append(threat)


        # Create tensor
        tensor = np.array([
            time,
            score,
            *asset_name_numbers,
            *healths,
            *positions.flatten(),
            *lle.flatten(),
            *weapons_quantities,
            *track_ids,
            *threat_id_numbers,
            *track_positions.flatten()
        ])

        # Reshape tensor to desired shape
        tensor = tensor.reshape(1, -1)

        # Convert to pytorch tensor
        tensor = torch.tensor(tensor, dtype=torch.float32)

        print(tensor.shape)
        #append tensor to parent tensor
        parent_tensor = torch.cat((parent_tensor, tensor), 0)
    print(parent_tensor.shape)



# 2. Preprocess data
#def preprocess(data, defenders=5, attackers=30):
#    zeroes = np.zeros((1,30))
#    #defining a single frame in a sequence of frames
#    # labels = np.array([])
## execution_order = self.calculateExecutionOrder(json_dict['Tracks'], origin)
#    import sys
#    # array = np.zeros((defenders+attackers, 7))
#    array = [-1]
#    for d in data:
#        try:
#            labels.append(State.calculateExecutionOrder(d['Tracks'],
#                                                       State.calculateOrigin(d['assets'])))
#        except:
#            #initialize zeroes np array with size (1,30)
#            labels.append(zeroes)
#        for asset in d['assets']:
#            array[0] = library[asset['AssetName']]
#            print(array[0])
#            print(labels)

#            sys.exit()
#        #dont forget about time and score!

if __name__ == "__main__":
    main()







## 2. Preprocess data

## 3. Save data as a pickle  

## trainer.py

## 1. Load data ()



##1 model that is multiclassification; two types of breed of dog

##2 models; # one for should we shoot binary clf
#           # one for execution order
#           # with what do we shoot with

##is reference ship the origin?

## Galleon_0  = [0 2 3 4 7]
##Galleon_0  = [0 2 3 4 7]
##Galleon_0  = [0 2 3 4 7]
##Galleon_0  = [0 2 3 4 7]
##Galleon_0  = [0 2 3 4 7]

###as you combine into a new directory, you can also rename the files


##300 states per game

##sequence 100 -model> execution order [0,1,...,30]
##sequence 50 -model> execution order [0,1,...,30]
##sequence 20 -model> execution order [0,1,...,30]

##ensemble




