import torch
import torch.nn as nn
import torch.nn.functional as F
from tester import *
import fire

def main(STATE=10):
    model = Seq2Seq()

    # Load the model
    model.load_state_dict(torch.load('./sequence_model/model_1.pt'))

    # Load the data from the .pkl file
    with open('./new_tensor_pkls/game_1.pkl', 'rb') as f:
        data = pickle.load(f)

    X = torch.unsqueeze(data[:100, :-60], 0)
    y = torch.unsqueeze(data[:, -60:-30], 0)

    output = model(X)
    output = output.squeeze_(0)
    output = F.relu(output)
    output = output.detach().numpy()
    #repeat for y
    y = y.squeeze_(0)
    y = y.detach().numpy()

    id_output = [(time, id) for time, id in enumerate(output[STATE,:])]
    id_y = [(time, id) for time, id in enumerate(y[STATE,:])]

    exec_order = sorted(id_y, key=lambda x: x[1])
    pred_exec_order = sorted(id_output, key=lambda x: x[1])

    exec_order = [x[0] for x in exec_order]
    pred_exec_order = [x[0] for x in pred_exec_order]

    print("Actual execution order: ", exec_order)
    print("Predicted execution order: ", pred_exec_order)

if __name__ == '__main__':
    fire.Fire(main)

