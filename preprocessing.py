import numpy as np
import os
import pickle
import pathlib
from pathlib import Path
import json
from config import *
from State import State
import torch
from utils import *
import sys

from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logging.set_verbosity(40) #only show errors
from transformers import BertTokenizer, BertModel
# MODEL = BertModel.from_pretrained("bert-base-cased")
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")


def main():
    # 1. Load data
    data_dir = './output3/'
    data_files = os.listdir(data_dir)
    paths_of_files = [os.path.join(data_dir, basename) for basename in data_files]
    
    for path in paths_of_files:
        with open(path, 'r', encoding='utf-8') as f:
            #load json string from json file
            data = f.read().replace('\n', ',')
            data = '['+data[:-1]+']'
            data = json.loads(data)
            data = preprocess(data)
            save_pickle(data, './tensor_pkls/', 'game')

def save_pickle(data, directory, base_filename):
    # Get next available file path
    directory = Path(directory)
    #remove the .json
    file_path = get_next_filepath(directory, base_filename)
    file_path = file_path.replace(".json", ".pkl")

    # Save data to file
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def preprocess(game):
    # Extract values from dictionary
    #for g in game:
    #    print(g)
    #    sys.exit()

    #make a parent tensor
    parent_tensor = torch.tensor([], dtype=torch.float32)
    for data in game:
        time = data.get("time")

        #grab all asset info IGNORE REFERENCE SHIP
        state={}
        hvu_info=[] # (1, root=6 + Lle=9 + weapons=6) note that PositionZ==0 effectively
        galleon_info=[]


        for asset in data:
            #exclude the reference ship
            if asset.get("AssetName")!="Galleon_REFERENCE_SHIP":


            #no else needed

        #grab all track info

        #grab memory info
        memory = data.get("memory")

        #PUT INTO A TENSOR WITH FOLLOWING LOGIC

        # #put each part of the data into a {key: [ 1d array of info ], ... }
        # #then we unpack our dict into a tensor
        # state = np.array([*state.values()]).flatten()
        # #check dimension is (1,info_size)
        # state = torch.tensor(state)
        # #return

'''
put into json beautify for reference

{"time": 7.04, "assets": [{"AssetName": "Galleon_REFERENCE_SHIP", "health": -1, "Lle": [25.0, -85.0, 0.0]}, {"AssetName": "HVU_Galleon_0", "isHVU": true, "health": 4, "PositionX": -2090.7503700031953, "PositionY": -1912.9656896674192, "PositionZ": -9.313225746154785e-10, "Lle": [24.982794848101822, -85.02074343642636, -9.313225746154785e-10], "weapons": [{"SystemName": "Cannon_System", "Quantity": 2, "WeaponState": "Ready"}, {"SystemName": "Chainshot_System", "Quantity": 4, "WeaponState": "Ready"}]}, {"AssetName": "Galleon_0", "health": 4, "PositionX": 7990.763960490963, "PositionY": 762.3615578821006, "Lle": [25.006835064375593, -84.92070390431094, 0.0], "weapons": [{"SystemName": "Cannon_System", "Quantity": 2, "WeaponState": "Ready"}, {"SystemName": "Chainshot_System", "Quantity": 4, "WeaponState": "Ready"}]}, {"AssetName": "Galleon_1", "health": 4, "PositionX": 6957.581669786882, "PositionY": -6417.88484277356, "Lle": [24.942266662046222, -84.93099289933922, 0.0], "weapons": [{"SystemName": "Cannon_System", "Quantity": 2, "WeaponState": "Ready"}, {"SystemName": "Chainshot_System", "Quantity": 4, "WeaponState": "Ready"}]}, {"AssetName": "Galleon_2", "health": 4, "PositionX": 32.76703285792545, "PositionY": -20.605749014037492, "Lle": [24.999814687693693, -84.9996748559751, 0.0], "weapons": [{"SystemName": "Cannon_System", "Quantity": 2, "WeaponState": "Ready"}, {"SystemName": "Chainshot_System", "Quantity": 4, "WeaponState": "Ready"}]}, {"AssetName": "Galleon_3", "health": 4, "PositionX": -693.0984865226911, "PositionY": 10136.633202978932, "Lle": [25.091160774106736, -85.0068826702868, 0.0], "weapons": [{"SystemName": "Cannon_System", "Quantity": 2, "WeaponState": "Ready"}, {"SystemName": "Chainshot_System", "Quantity": 4, "WeaponState": "Ready"}]}], "Tracks": [{"TrackId": 7, "ThreatId": "ENEMY_29", "ThreatRelationship": "Hostile", "Lle": [25.40612367205538, -85.28340844850591, 616.3077166797593], "PositionX": -28466.1029544425, "PositionY": 45188.796130634626, "PositionZ": 616.3077166797593, "VelocityX": 427.2232954378035, "VelocityY": -536.8693627181444, "VelocityZ": 9.451500926583016}], "memory": [7]}

'''





    #USE BELOW AS REFERECNE
    ######################################################################
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

        #add execution order label
        origin = State.calculateOrigin(data['assets'])
        if 'Tracks' in data.keys():
            execution_order = State.calculateExecutionOrder(data['Tracks'], origin)
        else:
            execution_order = [(-1,-1)] * 30
            # print(execution_order)

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

        #parse execution order into a 1d array
        # execution_order = np.array(execution_order).flatten()
        # print(execution_order.shape)
        # print()
        #unpack the tuples in the execution order
        execution_order = [item for sublist in execution_order for item in sublist]



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

    # pad parent tensor to (n, 300)
    # if parent_tensor.shape[0] < 300:
    #     min = torch.zeros((300-parent_tensor.shape[0], parent_tensor.shape[1]), dtype=torch.float32)
    #     parent_tensor = torch.cat((parent_tensor, min), 0)

    print(parent_tensor.shape)
    # sys.exit()

    return parent_tensor
    #index a tensor and see its execution order

    #how to print out the label
    # print(parent_tensor[0, -60:])
    #should only have been thirty but we allowed exec order to be a tuple

    #save to a pkl or return 


if __name__ == '__main__':
    main()

