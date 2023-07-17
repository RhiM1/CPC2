import torch
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

    
def logit_func(x,a,b):
    return 1 / (1 + np.exp(a * x + b))


def get_cal_curve(df_list):

    for df in df_list:
        logreg_y, logreg_x = calibration_curve(df["correctness"] / 100, df["predicted"] / 100, n_bins = 10)
        # print(logreg_x, logreg_y)
        fig, ax = plt.subplots()
        plt.plot(logreg_x, logreg_y)
        quit()


def isotonic_regression(df_list):
    
    iso_reg = IsotonicRegression(y_min = 0, y_max = 100, out_of_bounds = 'clip').fit(df_list[0]["predicted"].to_numpy(), df_list[0]["correctness"].to_numpy())

    for df in df_list:
        predicted_iso = iso_reg.predict(df["predicted"].to_numpy())
        df["predicted_iso"] = predicted_iso


    return df_list


def do_hist_plot(df_list):

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    fig, axs = plt.subplots(len(df_list), 5, sharey = True, sharex = True)
    for i, df in enumerate(df_list):
        axs[i, 0].hist(df["correctness"], bins = bins, density = True)
        axs[i, 1].hist(df["predicted"], bins = bins, density = True)
        axs[i, 2].hist(df["predicted_fitted"], bins = bins, density = True)
        axs[i, 3].hist(df["predicted_iso"], bins = bins, density = True)
        axs[i, 4].hist(df["predicted_quant"], bins = bins, density = True)
    axs[0, 0].set_title("Truth")
    axs[0, 1].set_title("Predicted")
    axs[0, 2].set_title("Logistic")
    axs[0, 3].set_title("Isotonic")
    axs[0, 4].set_title("Quantile")
    axs[0, 0].set_ylabel("Val density")
    axs[1, 0].set_ylabel("Dis density")
    axs[2, 0].set_ylabel("Dis lis density")
    axs[3, 0].set_ylabel("Dis sys density")
    axs[4, 0].set_ylabel("Dis scene density")
    axs[4, 0].set_xlabel("Correctness")
    axs[4, 1].set_xlabel("Correctness")
    axs[4, 2].set_xlabel("Correctness")
    axs[4, 3].set_xlabel("Correctness")
    axs[4, 4].set_xlabel("Correctness")
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


def calibrate(df_list):

    #normalise the predictions
    correctness = df_list[0]["correctness"].to_numpy() / 100
    predictions = df_list[0]["predicted"].to_numpy() / 100


    # logistic mapping curve fit to get the a and b parameters
    popt,_ = curve_fit(logit_func, predictions, correctness)
    a,b = popt

    for i, df in enumerate(df_list):
        df["predicted_fitted"] = logit_func(df["predicted"].to_numpy() / 100, a, b) * 100

    return df_list

def silly_correction(df_list):

    correctness = torch.tensor(df_list[0]["correctness"])
    values, counts = torch.unique(correctness, return_counts = True)

    counts = counts / counts.sum()

    # for i in range(1, len(counts)):
    #     counts[i] = counts[i] + counts[i - 1]

    # for value, count in zip(values, counts):
    #     print(f"{value.item():0.3f}  \t {count.item():0.3f}")

    for df in df_list:
        predictions = torch.tensor(df["predicted"])
        sorted, idx = torch.sort(predictions)
        location = 0
        len_pred = len(predictions)
        preds_quant = torch.zeros(len_pred, dtype = torch.float)
        for i in range(len(counts)):
            preds_quant[location:location + (counts[i] * len_pred).to(torch.long)] = values[i]
            location = location + (counts[i] * len_pred).to(torch.long)

        # preds_quant[idx] = preds_quant.clone()
        df["predicted_quant"] = preds_quant[idx]

    return df_list

def get_rmse(df_list):

    rmse_list = []
    rmse_fitted_list = []
    rmse_iso_list = []
    rmse_quant_list = []

    for df in df_list:
        correctness = df["correctness"].to_numpy()
        predicted = df["predicted"].to_numpy()
        predicted_fitted = df["predicted_fitted"].to_numpy()
        predicted_iso = df["predicted_iso"].to_numpy()
        predicted_quant = df["predicted_quant"].to_numpy()

        rmse = (predicted - correctness)**2
        rmse_fitted = (predicted_fitted - correctness)**2
        rmse_iso = (predicted_iso - correctness)**2
        rmse_quant = (predicted_quant - correctness)**2
        rmse_list.append(np.sqrt(rmse.mean()))
        rmse_fitted_list.append(np.sqrt(rmse_fitted.mean()))
        rmse_iso_list.append(np.sqrt(rmse_iso.mean()))
        rmse_quant_list.append(np.sqrt(rmse_quant.mean()))
    
    return rmse_list, rmse_fitted_list, rmse_iso_list, rmse_quant_list


def do_scatter_plots(df_list):

    fig, axs = plt.subplots(len(df_list), 3, sharey = True, sharex = True)
    for i, df in enumerate(df_list):
        axs[i, 0].scatter(df["correctness"],df["predicted"])
        axs[i, 1].scatter(df["correctness"],df["predicted_fitted"])
        axs[i, 2].scatter(df["correctness"],df["predicted_iso"])
    
    plt.show()

    

def do_cal_plot(df_list):

    hist, predicted_hist, fitted_hist, iso_hist, quant_hist = [], [], [], [], []
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # correctness_hist = np.histogram(df_list[0]["correctness"].to_numpy(), bins = bins)
    # print(correctness_hist)

    fig, axs = plt.subplots(len(df_list), sharey = True, sharex = True)
    for i, df in enumerate(df_list):
        correctness = df["correctness"].to_numpy()
        predicted = df["predicted"].to_numpy()
        predicted_fitted = df["predicted_fitted"].to_numpy()
        predicted_iso = df["predicted_iso"].to_numpy()
        predicted_quant = df["predicted_quant"].to_numpy()

        hist = np.cumsum(np.histogram(correctness, bins = bins)[0]) / len(correctness)
        predicted_hist = np.cumsum(np.histogram(predicted, bins = bins)[0]) / len(predicted)
        fitted_hist = np.cumsum(np.histogram(predicted_fitted, bins = bins)[0]) / len(predicted_fitted)
        iso_hist = np.cumsum(np.histogram(predicted_iso, bins = bins)[0]) / len(predicted_iso)
        quant_hist = np.cumsum(np.histogram(predicted_quant, bins = bins)[0]) / len(predicted_quant)

        # print(hist)
        # print(quant_hist)
        # print()
        
        p = axs[i].plot(hist, predicted_hist, marker = "x")
        pf = axs[i].plot(hist, fitted_hist, marker = "x")
        pi = axs[i].plot(hist, iso_hist, marker = "x")
        pq = axs[i].plot(hist, quant_hist, marker = "x")

        line = mlines.Line2D([0, 1], [0, 1], color='black')
        # transform = axs[i].transAxes
        # line.set_transform(transform)
        axs[i].add_line(line)

        
        plt.legend(
            ('Predicted', 'Fitted', 'Iso', 'Quant')
        )

        # plt.legend(
        #     (p, pf, pi, pq),
        #     ('Predicted', 'Fitted', 'Iso', 'Quant'),
        #     scatterpoints=1,
        #     loc='lower right'
        # )
        # hist.append(np.cumsum(np.histogram(correctness, bins = bins)[0])/len(correctness))
        # predicted_hist.append(np.cumsum(np.histogram(predicted, bins = bins)[0])/len(predicted))
        # fitted_hist.append(np.cumsum(np.histogram(predicted_fitted, bins = bins)[0])/len(predicted_fitted))
        # iso_hist.append(np.cumsum(np.histogram(predicted_iso, bins = bins)[0])/len(predicted_iso))
        # quant_hist.append(np.cumsum(np.histogram(predicted_quant, bins = bins)[0])/len(predicted_quant))
        # print(np.histogram(correctness, bins = bins))
        # print(np.histogram(correctness, bins = bins)[0])
        # print(np.cumsum(np.histogram(correctness, bins = bins)[0])/len(correctness))
        # print(len(correctness))
        # quit()
        # predicted_hist.append(np.cumsum(np.histogram(predicted, bins = bins, density = True)[0]))
        # fitted_hist.append(np.cumsum(np.histogram(predicted_fitted, bins = bins, density = True)[0]))
        # iso_hist.append(np.cumsum(np.histogram(predicted_iso, bins = bins, density = True)[0]))
        # quant_hist.append(np.cumsum(np.histogram(predicted_quant, bins = bins, density = True)[0]))
 
    # fig, axs = plt.subplots(len(df_list), 3, sharey = True, sharex = True)
    # for i, df in enumerate(df_list):
    #     axs[i, 0].scatter(hist[i][0],predicted_hist[i][0])
    #     # axs[i, 1].scatter(df["correctness"],df["predicted_fitted"])
    #     # axs[i, 2].scatter(df["correctness"],df["predicted_iso"])
    
    plt.show()



def main(args):

    val_df = pd.read_csv(args.data_file_val)
    dis_df = pd.read_csv(args.data_file_dis)
    dis_lis_df = pd.read_csv(args.data_file_dis_lis)
    dis_sys_df = pd.read_csv(args.data_file_dis_sys)
    dis_scene_df = pd.read_csv(args.data_file_dis_scene)

    df_list = [val_df, dis_df, dis_lis_df, dis_sys_df, dis_scene_df]

    # for df in df_list:
    #     print(df)

    df_list = calibrate(df_list)
    df_list = isotonic_regression(df_list)
    df_list = silly_correction(df_list)
    rmse, rmse_fitted, rmse_iso, rmse_quant = get_rmse(df_list)

    print(rmse)
    print(rmse_fitted)
    print(rmse_iso)
    print(rmse_quant)


    # for df in df_list:
    #     print(f"min correct: {min(df['correctness'])}, min predicted: {min(df['predicted'])}, min fitted: {min(df['predicted_fitted'])}, min iso: {min(df['predicted_iso'])}")


    # do_scatter_plots(df_list)

    # do_hist_plot(df_list)    

    # do_cal_plot(df_list)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", help="save directory", default = "save/"
    )
    parser.add_argument(
        # "--data_file", help="file to plot", default = "001_N1_WhisperFull_ExLSTM_log"
        # "--data_file", help="file to plot", default = "004_N3_WhisperFull_LSTM_layers"
        "--data_file", help="file to plot", default = "011_N3_WhisperFull_ExLSTM_layers"
    )
    # parser.add_argument(
    #     "--skip_wandb", help="skip logging via WandB", default=False, action='store_true'
    # )

    args = parser.parse_args()
    
    args.data_file_val = f"{args.data_dir}{args.data_file}_val_preds.csv"
    args.data_file_dis = f"{args.data_dir}{args.data_file}_dis_val_preds.csv"
    args.data_file_dis_lis= f"{args.data_dir}{args.data_file}_dis_lis_val_preds.csv"
    args.data_file_dis_sys = f"{args.data_dir}{args.data_file}_dis_sys_val_preds.csv"
    args.data_file_dis_scene = f"{args.data_dir}{args.data_file}_dis_scene_val_preds.csv"

    

    main(args)