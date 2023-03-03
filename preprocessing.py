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
import hashlib

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
    file_path = str(file_path).replace(".json", ".pkl")
    # file_path = file_path[:-4] + ".pkl"

    # Save data to file
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def preprocess(game):
    # Extract values from dictionary


    #make a parent tensor
    parent_tensor = torch.tensor([], dtype=torch.float32)
    for data in game:

        # Extract Labels
        #add execution order label
        origin = State.calculateOrigin(data['assets'])
        if 'Tracks' in data.keys():
            #TODO (FIX THIS AND MAKE FRESH DATA):
            #so here in the state.py calculateExecutionOrder we need to also check for the
            # threat relationship prior to considering in our calculation
            # if data["Tracks"]["ThreatRelationship"] == "Hostile"

            #this filters out friendlies AFTER YOU FIX IT
            execution_order = State.calculateExecutionOrder(data['Tracks'], origin)
            #unpack you list of tuples [(a,b), (c,d),..] into [a,b,c,d,...]
            transposed = zip(*execution_order)
            #flatten the list
            t_id = [item for sublist in transposed for item in sublist]

            #we need to pad the list with -1's to make it 30 long
            t_id = t_id + [-1] * (30 - len(t_id))

        else:
            #if you don't have tracks, then you don't have execution order
            t_id = [-1] * 60 #i.e. no time and no ids 

        #grab all asset info IGNORE REFERENCE SHIP
        state={}
        hvu_info=[]     # (1, root=6 + Lle=3 + weapons=6) note that PositionZ==0 effectively
        galleon_info=[] # 5x(1, root=4 + Lle=3 + weapons=6) note; pad for 5
        tracks=[]       # 30x(1, root=3, Lle=3, posvos=6) note; pad for 30*2 (include friendlies, this is why your memory buffer keeps breaking)
        hidden_state_info=[] #(1, memory=30, miscallaneous=270)

        if "score" in data.keys():
            score = data.get("score")
        else:
            score = 0

        time = data.get("time")

        for asset in data["assets"]:

            #exclude the reference ship
            if asset.get("AssetName")=="Galleon_REFERENCE_SHIP":
                pass

            elif asset.get("isHVU"):

                #get HVU info
                hvu_info.append(library[asset.get("AssetName")])
                hvu_info.append(library[asset.get("isHVU")])
                hvu_info.append(asset.get("health"))
                hvu_info.append(asset.get("PositionX"))
                hvu_info.append(asset.get("PositionY"))
                hvu_info.append(asset.get("PositionZ"))
                hvu_info.append(asset.get("Lle")[0])
                hvu_info.append(asset.get("Lle")[1])
                hvu_info.append(asset.get("Lle")[2])
                for weapon in asset.get("weapons"):
                    hvu_info.append(library[weapon.get("SystemName")])
                    if weapon.get("Quantity") is None:
                        hvu_info.append(-1)
                    else:
                        hvu_info.append(weapon.get("Quantity"))
                    hvu_info.append(library[weapon.get("WeaponState")]) #if error, add value to library
            else:

                #get galleon info
                galleon_info.append(library[asset.get("AssetName")])
                galleon_info.append(asset.get("health"))
                galleon_info.append(asset.get("PositionX"))
                galleon_info.append(asset.get("PositionY"))
                galleon_info.append(asset.get("Lle")[0])
                galleon_info.append(asset.get("Lle")[1])
                galleon_info.append(asset.get("Lle")[2])
                for weapon in asset.get("weapons"):
                    galleon_info.append(library[weapon.get("SystemName")])
                    if weapon.get("Quantity") is None:
                        galleon_info.append(-1)
                    else:
                        galleon_info.append(weapon.get("Quantity"))
                    galleon_info.append(library[weapon.get("WeaponState")])

        #grab all track info
        for track in data.get("Tracks"):
            tracks.append(track.get("TrackId"))
            #handle when threat id is
            # KeyError: 'Cannon_Ball_1>enemy_track:7'
            if track.get("ThreatId") not in library.keys():
                raw=track.get("ThreatId")
                ids=list(np.array(TOKENIZER.encode(raw, add_special_tokens=False)).flatten())
                encoding=int("".join([str(id_) for id_ in ids]))
                encoding=encoding % 1e6
                tracks.append(encoding)
            else:
                tracks.append(library[track.get("ThreatId")])
            tracks.append(library[track.get("ThreatRelationship")])
            tracks.append(track.get("Lle")[0])
            tracks.append(track.get("Lle")[1])
            tracks.append(track.get("Lle")[2])
            tracks.append(track.get("PositionX"))
            tracks.append(track.get("PositionY"))
            tracks.append(track.get("PositionZ"))
            tracks.append(track.get("VelocityX"))
            tracks.append(track.get("VelocityY"))
            tracks.append(track.get("VelocityZ"))

        #grab memory info
        memory = data.get("memory")

        # hvu_info=[]     # (1, root=6 + Lle=3 + weapons=6) note that PositionZ==0 effectively
        # galleon_info=[] # 5x(1, root=4 + Lle=3 + weapons=6) note; pad for 5
        # tracks=[]       # 30x(1, root=3, Lle=3, posvos=6) note; pad for 30

        #pad galleon info
        if len(galleon_info)<5*13:
            galleon_info.extend([0]*(5*13-len(galleon_info)))

        #pad tracks (note tracks can include friendlies so 30*2 actually)
        if len(tracks)<2*30*12:
            tracks.extend([0]*(2*30*12-len(tracks)))

        #pad hidden state info
        if len(memory)<300:
            memory.extend([0]*(300-len(memory)))

        assert len(galleon_info)==5*13
        assert len(tracks)==2*30*12 #likely saving a Friendly track on top of 30 enemy tracks
        assert len(memory)==300
        assert len(t_id)==30


        #WE STILL NEED TO SAVE THE EXECUTION ORDER and TIME to be scraped later 


        #put into a tensor with following logic
        state = torch.tensor([time, score, *hvu_info, *galleon_info, *tracks, *memory, *t_id
                             dtype=torch.float32).flatten()

        state = state.reshape(1, -1)

        # In order to unpack label state[N,:-30] and of those 30,
        # [time_missle_fastest, id_missle_fastest, ... ] , 

                              #sike
                              #loss calculated after you do a
                              # sorted(execution_order, lambda x: x[1]) and st the model call in
                              # State.py looks like this:
                              # execution_order = model(json_state) #sorted by ascending order of
                              # id values
                              # execution_order = sorted(execution_order, lambda x: x[0])
        # use the time in order to calculate the loss
        # use the id in order to 

        #have a dictionary instantate {0:id, 1:id, 2:id, ... 29:id}
        #model gives some (1,30) of times
        #   
            #we then use the ids associated with each index to 


        print(parent_tensor.shape)
        print(state.shape)

        #append to parent tensor so we have (300,info_size)
        parent_tensor = torch.cat((parent_tensor, state), dim=0)

    #sometimes games end early so we need to pad, (300, info_size) st info_size.size==1132 (with the labels)
    if parent_tensor.shape[0]<300:
        parent_tensor = torch.cat((parent_tensor, torch.zeros((300-parent_tensor.shape[0], 1132))), dim=0)

    print(parent_tensor.shape)

    return parent_tensor


if __name__ == '__main__':
    main()

