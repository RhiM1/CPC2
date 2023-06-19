import torch
import argparse
import pandas as pd
import numpy as np

def get_disjoint_val_set(args, data):

    CEC2_data = data.loc[data['subset'] == "CEC2"]
    print(CEC2_data[:10])

    unique_CEC2_listeners = CEC2_data.listener.unique()
    unique_CEC2_listeners.sort()
    val_listeners = unique_CEC2_listeners[-2:]
    print(val_listeners)
    unique_CEC1_listeners = data.loc[data['subset'] == "CEC1"].listener.unique()
    unique_CEC1_listeners.sort()
    print(unique_CEC2_listeners)
    print(unique_CEC1_listeners)
    print(val_listeners)

    print()

    return None, None

def main(args):

    data = pd.read_json(args.in_json_file)
    data["subset"] = "CEC1"
    data2 = pd.read_json(args.in_json_file.replace("CEC1","CEC2"))
    data2["subset"] = "CEC2"
    data = pd.concat([data, data2])
    data["predicted"] = np.nan  # Add column to store intel predictions

    # print(data[:50])

    train_data, val_data = get_disjoint_val_set(args, data)

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_json_file", help="location of CEC1 metadata file", default = "/home/acp20rm/exp/data/clarity_CPC2_data/clarity_data/metadata/CEC1.train.1.json"
        # "--in_json_file", help="location of CEC1 metadata file", default = "~/data/clarity_CPC2_data/clarity_data/metadata/CEC1.train.1.json"
    )
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)
    