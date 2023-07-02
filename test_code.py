from datasets import load_dataset

DATAROOT = "/home/acp20rm/exp/data/clarity_CPC2_data/clarity_data/"

if __name__ == "__main__":
    dataset = load_dataset(DATAROOT)

    print(dataset)