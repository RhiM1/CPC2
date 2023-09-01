import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from constants import DATAROOT, DATAROOT_CPC1

def plot_correctness_histogram(args, data):

    data.hist(column = 'correctness', by = ['N', 'CEC'], sharex = True, sharey = True, density = True)
    plt.show()

def plot_listener_boxplots(args, data):
    
    # fig, axs = plt.subplots(3, 2, sharey = True)
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
    data.boxplot(column = 'correctness', by = ['N', 'CEC', 'listener'], rot = 45)

    plt.show()

def plot_scene_boxplots(args, data):
    
    # fig, axs = plt.subplots(3, 2, sharey = True)
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
    data.boxplot(column = 'correctness', by = ['scene'], rot = 45)

    plt.show()

def compare_testCPC1_to_trainCPC2(args, datas, test_data):

    test_data = test_data[0]
    # test_data['lis_sys']
    # for i in range(test_data):
    # test_data["lis_sys_scene"] = f"{test_data['listener']}_{test_data['system']}_{test_data['scene']}"

    for data in datas:
        # data["lis_sys_scene"] = f"{data['listener']}_{data['system']}_{data['scene']}"
        for i in range(10):
            # print(test_data['lis_sys_scene'] == data['lis_sys_scene'])
            # print(test_data.loc(test_data['lis_sys_scene']).isin(data['lis_sys_scene']))
            # print(i, data.iloc[i]['lis_sys_scene'])
            # print(i, data.iloc[i]['listener'], test_data.iloc[i]['listener'])
            overlap = data.loc[ 
                (data['listener'] == test_data['listener'].iloc[i]) & \
                (data['system'] == test_data['system'].iloc[i]) & \
                (data['scene'] == test_data['scene'].iloc[i])
            ]
            if not overlap.empty:
                print(i)
                print(overlap)
                print(test_data.iloc[i])
                print()
        
        print()

def get_listener_stats(args, data):

    ## Unique listeners
    # for CEC in [1, 2]:
    #     for N in [1, 2, 3]:
    #         unique_listeners = data.loc[(data['CEC'] == f"CEC{CEC}") & (data['N'] == f"N{N}")].listener.unique()
    #         unique_listeners.sort()
    #         print(f"unique listeners for CEC{CEC} N{N}:\n{unique_listeners}")


    # # Unique systems        
    # for CEC in [1, 2]:
    #     for N in [1, 2, 3]:
    #         unique_systems = data.loc[(data['CEC'] == f"CEC{CEC}") & (data['N'] == f"N{N}")].system.unique()
    #         unique_systems.sort()
    #         print(f"{unique_systems}")

    
    data1 = data.loc[data['N'] == 'N1']
    data2 = data.loc[data['N'] == 'N2']
    data3 = data.loc[data['N'] == 'N3']

    print(f"total data length: {len(data)}")
    print(f"number of listeners: {len(data.listener.unique())}")
    print(f"number of systems: {len(data.system.unique())}")
    print(f"number of scenes: {len(data.scene.unique())}")
    print()
    
    print(f"total data length N1: {len(data1)}")
    print(f"number of listeners: {len(data1.listener.unique())}")
    print(f"number of systems: {len(data1.system.unique())}")
    print(f"number of scenes: {len(data1.scene.unique())}")
    print()

    print(f"total data length N1: {len(data2)}")
    print(f"number of listeners: {len(data2.listener.unique())}")
    print(f"number of systems: {len(data2.system.unique())}")
    print(f"number of scenes: {len(data2.scene.unique())}")
    print()

    print(f"total data length N1: {len(data3)}")
    print(f"number of listeners: {len(data3.listener.unique())}")
    print(f"number of systems: {len(data3.system.unique())}")
    print(f"number of scenes: {len(data3.scene.unique())}")
    print()

    # Unique scenes 
    # unique_scenes_list = [[0] * 3] * 2      
    # for CEC in [1, 2]:
    #     unique_scenes_list.append([0])
    #     for N in [1, 2, 3]:
    #         unique_scenes = data.loc[(data['CEC'] == f"CEC{CEC}") & (data['N'] == f"N{N}")].scene.unique()
    #         unique_scenes.sort()
    #         unique_scenes_list[CEC-1][N-1] = unique_scenes
    #         print(f"unique scenes for CEC{CEC} N{N}:\n{unique_scenes}")



def main(args):
    
    test_datas = []
    datas = []
    for N in args.N:
        json_in = f"{args.meta_dir}CEC2.test.{N}.json"
        test_datas.append(pd.read_json(json_in))
        test_datas[N-1]["CEC"] = f"CEC2"
        json_in = f"{args.meta_dir}CEC1.train.{N}.json"
        data = pd.read_json(json_in)
        data["CEC"] = f"CEC1"
        json_in = f"{args.meta_dir}CEC2.train.{N}.json"
        data2 = pd.read_json(json_in)
        data2["CEC"] = f"CEC2"
        datas.append(pd.concat([data, data2]))
        # for CEC in args.CEC:
        #     json_in = f"{args.meta_dir}CEC{CEC}.train.{N}.json"
        #     data = pd.read_json(json_in)
        #     data["CEC"] = f"CEC{CEC}"
        #     data["N"] = f"N{N}"
        #     datas.append(data)

    for data in datas:
        print(data)

    for data in test_datas:
        print(data)

    # for N in range(3):
    #     lis_test = test_datas[N].listener.unique()
    #     lis_train = datas[N].listener.unique()
    #     sys_test = test_datas[N].system.unique()
    #     sys_train = datas[N].system.unique()
    #     scene_test = test_datas[N].scene.unique()
    #     scene_train = datas[N].scene.unique()
        # print(lis_test)
        # print(list_train)
        # for listener in lis_test:
        #     if listener in lis_train:
        #         print(listener)
        #     else:
        #         print("not in train!")
        # for system in sys_test:
        #     if system in sys_train:
        #         print(system)
        #     else:
        #         print("not in train!")
        # for scene in scene_test:
        #     if scene in scene_train:
        #         print(scene)
        #     else:
        #         print("not in train!")
    
    
    # data = pd.concat(datas)
    # unique_corr = data.correctness.unique()
    # print(unique_corr)
    # print(unique_corr.sort())
    # print(unique_corr.shape)

    # cpc1_eval_data = [pd.read_json(f"{args.cpc1_eval_dir}CPC1.test.json")]

    # compare_testCPC1_to_trainCPC2(args, datas, cpc1_eval_data)

    # plot_correctness_histogram(args, data)
    # plot_listener_boxplots(args, data)

    # get_listener_stats(args, data)
    # plot_scene_boxplots(args, data)

    # fig, axs = plt.subplots(nrows=2, ncols=2)
    # datas[0].boxplot(column = "correctness", by = "listener", ax = axs[0, 0])
    # plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--N", help="partion to explore", default=0, type=int
    # )
    parser.add_argument(
        "--model", help="model type" , default="999", type=str
    )
    parser.add_argument(
        "--meta_dir", help="Directory of metadata json files" , default=DATAROOT + "metadata/", type=str
    )
    parser.add_argument(
        # "--json_dir", help="Directory of metadata json files" , default="/store/store1/data/clarity_CPC2_data/clarity_data/metadata/", type=str
        # "--cpc1_eval_dir", help="Directory of metadata json files" , default="~/data/clarity_CPC1_eval_data/metadata/", type=str
        "--cpc1_eval_dir", help="Directory of metadata json files" , default="/home/acp20rm/exp/data/clarity_CPC1_eval_data/metadata/", type=str
    )
    # parser.add_argument(
    #     "--CEC", help="CEC 1 or 2; leave blank for both" , default=0, type=int
    # )

    args = parser.parse_args()
    args.CEC = [1, 2]
    args.N = [1, 2, 3]

    main(args)