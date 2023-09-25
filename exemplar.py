import torch
import pandas as pd
import argparse
import numpy as np
from torch.utils.data import DataLoader
import speechbrain as sb
from sklearn.model_selection import train_test_split
from data_handling import get_disjoint_val_set

from constants import DATAROOT


def get_ex_set(data, args):
    
    data = data.sort_values('correctness')

    correctness = data.correctness.values
    num_ex_sets, remainder = divmod(len(correctness), args.ex_size)

    # print(f"remainder: {remainder}")

    ex_set_ids = torch.zeros(len(correctness))
    ex_set_ids[torch.randperm(len(correctness))[0:remainder]] = -9999
    # print(ex_set_ids[torch.randperm(len(correctness))[0:remainder]])
    # print(len(ex_set_ids[ex_set_ids == -9999]))

    ex_sets = torch.zeros(args.ex_size, num_ex_sets)
    for i in range(args.ex_size):
        ex_sets[i] = torch.randperm(num_ex_sets)
    try:
        ex_set_ids[ex_set_ids != -9999] = ex_sets.flatten()
    except:
        print(f"ex_set_ids size: {ex_set_ids.size()}")
        print(f"ex_set_ids[ex_set_ids == -9999] size: {ex_set_ids[ex_set_ids == -9999].size()}")
        print(f"ex_set_ids[ex_set_ids != -9999] size: {ex_set_ids[ex_set_ids != -9999].size()}")
        print(f"ex_sets size: {ex_sets.flatten().size()}")
        quit()


    data['ex_set_id'] = ex_set_ids.tolist()
    data = data.sort_values('ex_set_id')
    data = data[data.ex_set_id != -9999]

    return data

def get_listener_ex_set(data, args):

    unique_listeners = data.listener.unique()

    # for listener in unique_listeners:




def main(args):

    print(args.in_json_file)

    data = pd.read_json(args.in_json_file)
    data["subset"] = "CEC1"
    data2 = pd.read_json(args.in_json_file.replace("CEC1","CEC2"))
    data2["subset"] = "CEC2"
    data = pd.concat([data, data2])
    data = data.drop_duplicates(subset = ['signal'], keep = 'last')
    data["predicted"] = np.nan  
    data = get_disjoint_val_set(args, data)
    train_data, val_data = train_test_split(data[data.validation == 0], test_size=0.1)

    unique_listeners = train_data.listener.unique()
    unique_systems = train_data.system.unique()
    unique_scenes = train_data.scene.unique()
    ex_data = get_ex_set(train_data[train_data.subset == "CEC2"], args)

    print(train_data[train_data.subset == "CEC2"])
    print(ex_data)


    ex_dataloader = DataLoader(ex_data,args.ex_size,collate_fn=sb.dataio.batch.PaddedBatch, shuffle = False)
    # len_ex = len(ex_dataloader)
    # print(f"len_ex: {len_ex}")
    # ex_dataloader = iter(ex_dataloader)
    # ex_used = 0

    for batch in ex_dataloader:
        print(batch)

    # for batchID in range(1000):
    #     # if len_ex - ex_used < args.ex_size:
    #     #     ex_data = get_ex_set(ex_data, args)
    #     #     ex_dataloader = DataLoader(ex_data,args.ex_size,collate_fn=sb.dataio.batch.PaddedBatch, shuffle = False)
    #     #     ex_dataloader = iter(ex_dataloader)   
    #     #     ex_used = 0
    #     print(batchID)
    #     ex_batch = next(ex_dataloader)
    #     ex_used += args.ex_size
    #     print(ex_batch)




if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", help="random seed for repeatability", default=1234,
    )

    parser.add_argument(
        "--N", help="train split" , default=1, type=int
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size", help="batch size" , default=None, type=int
    )
    parser.add_argument(
        "--n_epochs", help="number of epochs", default=None, type=int
    )

    # Exemplar bits
    parser.add_argument(
        "--ex_size", help="train split" , default=8, type=int
    )
    parser.add_argument(
        "--p_factor", help="exemplar model p_factor" , default=None, type=int
    )

    args = parser.parse_args()
    
    args.dataroot = DATAROOT
    args.in_json_file = f"{DATAROOT}/metadata/CEC1.train.{args.N}.json"
    print(args.in_json_file)


    main(args)