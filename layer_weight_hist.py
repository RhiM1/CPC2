
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    with open("save/layer_weights.txt", 'r') as f:
        temp = f.read().split("\n")

    weights = []
    for mod_weights in temp:
        mod_weights = mod_weights.split(" ")
        print(mod_weights)
        weights.append([float(mod_weight) for mod_weight in mod_weights])

    print(weights)
    layer = np.arange(12)

    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    # ax.set_xlabel()
    # ax.bar(layer + 0.00, weights[0], color = 'lightgray', width = 0.25)
    # ax.bar(layer + 0.25, weights[1], color = 'dimgray', width = 0.25)
    # ax.bar(layer + 0.50, weights[2], color = 'gray', width = 0.25)
    # ax.set_xticks(layer, layer + 1)

    # # plt.bar(layer, height = weights)
    # plt.show()


    barWidth = 0.25


    fig = plt.subplots(figsize =(7.5, 3))
    
    
    # Set position of bar on X axis
    br1 = np.arange(len(weights[0]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    # Make the plot
    plt.bar(br1, weights[0], color ='lightgray', width = barWidth,
            edgecolor ='black', label ='split 1')
    plt.bar(br2, weights[1], color ='dimgray', width = barWidth,
            edgecolor ='black', label ='split 2')
    plt.bar(br3, weights[2], color ='gray', width = barWidth,
            edgecolor ='black', label ='split 3')
    
    # Adding Xticks
    plt.xlabel('Whisper decoder layer', fontweight ='bold', fontsize = 15)
    plt.ylabel('Learned weight', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(weights[0]))],
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    
    plt.legend()
    plt.show()

