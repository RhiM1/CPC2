import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from constants import DATAROOT, DATAROOT_CPC1


def main(args):

    df_val_base = pd.read_csv(args.data_file_val_base)
    df_test_base = pd.read_csv(args.data_file_test_base)
    df_val_ex = pd.read_csv(args.data_file_val_ex)
    df_test_ex = pd.read_csv(args.data_file_test_ex)
    # df_val_ex2 = pd.read_csv(args.data_file_val_ex2)
    # df_test_ex2 = pd.read_csv(args.data_file_test_ex2)

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    fig, axs = plt.subplots(3, 2, sharey = True, sharex = True)
    axs[0, 0].hist(df_val_base["predicted"], bins = bins, density = True)
    axs[0, 1].hist(df_test_base["predicted"], bins = bins, density = True)
    axs[1, 0].hist(df_val_ex["predicted"], bins = bins, density = True)
    axs[1, 1].hist(df_test_ex["predicted"], bins = bins, density = True)
    sys_com_val = (df_val_base.predicted.values + df_val_ex.predicted.values) / 2
    sys_com_test = (df_test_base.predicted.values + df_test_ex.predicted.values) / 2
    axs[2, 0].hist(sys_com_val, bins = bins, density = True)
    axs[2, 1].hist(sys_com_test, bins = bins, density = True)
        # axs[i, 3].hist(df["predicted_iso"], bins = bins, density = True)
        # axs[i, 4].hist(df["predicted_quant"], bins = bins, density = True)
    # axs[0, 0].set_title("Truth")
    # axs[0, 1].set_title("Predicted")
    # axs[0, 2].set_title("Neg predicted")
    # axs[0, 0].set_ylabel("Val density")
    # axs[1, 0].set_ylabel("Dis density")
    # axs[2, 0].set_ylabel("Dis lis density")
    # axs[3, 0].set_ylabel("Dis sys density")
    # axs[4, 0].set_ylabel("Dis scene density")
    # axs[4, 0].set_xlabel("Correctness")
    # axs[4, 1].set_xlabel("Correctness")
    # axs[4, 2].set_xlabel("Correctness")
    # for CEC in args.CEC:
    #     for N in args.N:
    #         this_data = data.loc[(data['CEC'] == CEC) & (data['N'] == N)]
    #         this_data.boxplot(column = "correctness", by = "listener", ax = axs[N - 1, CEC - 1], rot = 45)
    #         axs[N - 1, CEC - 1].set(xlabel = "listener", ylabel = "correctness (%)", title = "")
    #         axs
            # axs[N - 1, CEC - 1].hist(data["correctness"], density = True)
            # axs[N - 1, CEC - 1].set_title(f"N{N} CEC{CEC}")
            # axs[N - 1, CEC - 1].set(xlabel = "correctness (%)", ylabel = "frequency density")
            # axs[N - 1, CEC - 1].label_outer()
            # print(f"N: {N}, CEC: {CEC}\n{datas[(CEC - 1) * 3 + N - 1]}\n")
    # data.boxplot(column = 'correctness', by = ['N', 'CEC', 'listener'], rot = 45)

    plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", help="save directory", default = "save/"
    )
    parser.add_argument(
        "--data_file_base", help="file to plot", default = "007final_N1_WhisperFull_LSTM_layers"
    )
    parser.add_argument(
        "--data_file_ex", help="file to plot", default = "015final_N1_WhisperFull_ExLSTM_layers"
    )
    parser.add_argument(
        "--data_file_ex2", help="file to plot", default = "021test2final_N1_WhisperFull_ExLSTM_layers"
    )
    parser.add_argument(
        "--neg_data_file", help="file to plot", default = None
    )
    # parser.add_argument(
    #     "--skip_wandb", help="skip logging via WandB", default=False, action='store_true'
    # )

    args = parser.parse_args()
    
    args.data_file_val_base = f"{args.data_dir}{args.data_file_base}_val_preds.csv"
    args.data_file_test_base = f"{args.data_dir}{args.data_file_base}_test_preds.csv"
    args.data_file_val_ex = f"{args.data_dir}{args.data_file_ex}_val_preds.csv"
    args.data_file_test_ex = f"{args.data_dir}{args.data_file_ex}_test_preds.csv"
    args.data_file_val_ex2 = f"{args.data_dir}{args.data_file_ex2}_val_preds.csv"
    args.data_file_test_ex2 = f"{args.data_dir}{args.data_file_ex2}_test_preds.csv"
    

    main(args)