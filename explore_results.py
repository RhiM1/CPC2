import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from constants import DATAROOT, DATAROOT_CPC1

def plot_rmse_by_correctness(data, model = "combination"):

    correctness = data.correctness.values
    if model == "combination":
        predicted = data.combination.values
    elif model == "base":
        predicted = data.base.values
    elif model == "exemplar":
        predicted = data.ex.values

    rmse = (correctness - predicted)**2
    rmse = rmse.mean()**0.5
    print(rmse)
    
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.01]

    bin_correct = []

    for i in range(10):
        bin_correcness = correctness[np.logical_and(predicted >= bins[i], predicted < bins[i+1])]
        bin_predicted = predicted[np.logical_and(predicted >= bins[i], predicted < bins[i+1])]
        bin_rmse = (bin_correcness - bin_predicted)**2
        bin_rmse = bin_rmse.mean()**0.5
        bin_correct.append(bin_rmse)

    print(bin_correct)

    plt.bar(np.arange(5.25, 100, 10), bin_correct, width = 9.5)
    plt.xlabel("Correctness")
    plt.ylabel("RMSE")
    plt.show()
        

def plot_rmse_by_listener(data, model = "combination"):

    correctness = data.correctness.values
    if model == "combination":
        predicted = data.combination.values
    elif model == "base":
        predicted = data.base.values
    elif model == "exemplar":
        predicted = data.ex.values
    listeners = data.listener.values
    unique_listeners = data.listener.unique()
    unique_listeners = np.sort(unique_listeners)
    listener_rmses = []
    print(unique_listeners)
    for listener in unique_listeners:
        listener_correctness = correctness[listeners == listener]
        listener_predicted = predicted[listeners == listener]
        listener_rmse = (listener_predicted - listener_correctness)**2
        listener_rmses.append(listener_rmse.mean()**0.5)

    plt.bar(unique_listeners, listener_rmses)
    plt.xlabel("Listener")
    plt.ylabel("RMSE")
    plt.show()

    
def plot_rmse_by_system(data, model = "combination"):

    correctness = data.correctness.values
    if model == "combination":
        predicted = data.combination.values
    elif model == "base":
        predicted = data.base.values
    elif model == "exemplar":
        predicted = data.ex.values
    systems = data.system.values
    unique_systems = data.system.unique()
    unique_systems = np.sort(unique_systems)
    system_rmses = []
    print(unique_systems)
    for system in unique_systems:
        system_correctness = correctness[systems == system]
        system_predicted = predicted[systems == system]
        system_rmse = (system_predicted - system_correctness)**2
        system_rmses.append(system_rmse.mean()**0.5)

    plt.bar(unique_systems, system_rmses)
    plt.xlabel("System")
    plt.ylabel("RMSE")
    plt.show()


def plot_rmse_by_listener2(data, model = "combination"):

    correctness = data.correctness.values
    if model == "combination":
        predicted = data.combination.values
    elif model == "base":
        predicted = data.base.values
    elif model == "exemplar":
        predicted = data.ex.values
    listeners = data.listener.values
    unique_listeners = data.listener.unique()
    unique_listeners = np.sort(unique_listeners)
    listener_rmses = []
    listener_corrects = []
    print(unique_listeners)
    for listener in unique_listeners:
        listener_correctness = correctness[listeners == listener]
        listener_predicted = predicted[listeners == listener]
        listener_rmse = (listener_predicted - listener_correctness)**2
        listener_corrects.append(listener_correctness.mean())
        listener_rmses.append(listener_rmse.mean()**0.5)

    print(listener_corrects)
    print(listener_rmses)

    plt.scatter(listener_corrects, listener_rmses)
    for i, system in enumerate(unique_listeners):
        plt.annotate(system, (listener_corrects[i] + 0.7, listener_rmses[i]))
    plt.xlabel("Mean listener correctness")
    plt.ylabel("Listener RMSE")
    plt.show()


def plot_rmse_by_system2(data, model = "combination"):

    correctness = data.correctness.values
    if model == "combination":
        predicted = data.combination.values
    elif model == "base":
        predicted = data.base.values
    elif model == "exemplar":
        predicted = data.ex.values
    systems = data.system.values
    unique_systems = data.system.unique()
    unique_systems = np.sort(unique_systems)
    system_rmses = []
    system_corrects = []
    print(unique_systems)
    for system in unique_systems:
        system_correctness = correctness[systems == system]
        system_predicted = predicted[systems == system]
        system_rmse = (system_predicted - system_correctness)**2
        system_corrects.append(system_correctness.mean())
        system_rmses.append(system_rmse.mean()**0.5)

    print(system_corrects)
    print(system_rmses)

    plt.scatter(system_corrects, system_rmses)
    for i, system in enumerate(unique_systems):
        plt.annotate(system, (system_corrects[i] + 0.7, system_rmses[i]))
    plt.xlabel("Mean system correctness")
    plt.ylabel("System RMSE")
    plt.show()


def plot_correctness_hist(data):

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.hist(data['correctness'], bins = bins, density = True)
    plt.show()



def main(args):

    data1 = pd.read_csv(args.out_csv_file_1)
    data2 = pd.read_csv(args.out_csv_file_2)
    data3 = pd.read_csv(args.out_csv_file_3)

    data1 = pd.concat([data1, data2])
    data = pd.concat([data1, data3])
    data = data.drop_duplicates()

    # Figure 3a (model performance by correctness)
    plot_rmse_by_correctness(data)

    # plot_correctness_hist(data)
    # plot_rmse_by_listener(data)
    # plot_rmse_by_system(data)
    # plot_rmse_by_listener2(data, model = "exemplar")

    # Figure 4 (model performance by mean HA system performance)
    plot_rmse_by_system2(data, model = "combination")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_csv_file_1", help="Name of ouput CSV files" , default="save/analysis/eval1.csv", type=str
    )
    parser.add_argument(
        "--out_csv_file_2", help="Name of ouput CSV files" , default="save/analysis/eval2.csv", type=str
    )
    parser.add_argument(
        "--out_csv_file_3", help="Name of ouput CSV files" , default="save/analysis/eval3.csv", type=str
    )
    
    args = parser.parse_args()

    main(args)