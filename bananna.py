import matplotlib.pyplot as plt
import numpy as np
fig=plt.figure(figsize=(6,2)) # 8x6 is the default (too large here)

item = ['j','z', 'm', 'v', 'd',
        'n','ng','e', 'b', 'l',
        'u','i', 'o', 'a', 'r',
        'p','h', 'ch','sh','g',
        'k','f','s','th']

item_HL = [40, 30, 40, 30 ,40,
            43, 46, 50, 40, 50, 
            54, 40 ,45, 40, 54, 
            38, 40, 45, 48, 40, 
            38, 25, 28 ,26]

item_freq = [ 218,  277,  282,  340,  345, 
                352,  345,  345,  405,  410,
                407,  573,  573,  700,  811, 
                1158, 1440, 1440, 1680, 1717, 
                2698, 4385, 5382, 5821]
item_freq_log = 10*np.log10(item_freq)

banana_area = np.array([
                [125, 11],   [102,8],   [102, 30],  [125, 40],  [205, 50],
                [250, 52],  [364, 57],  [500, 60],  [727, 62],  [1000, 63],
                [1459, 62], [2000, 59], [2869, 55], [4000, 50], [5804, 44], 
                [6777, 40], [8000, 34], [9726, 30], [9999, 20], [9780, 16],
                [8000, 17], [6204, 20], [5733, 22], [4000, 28], [3398, 30],
                [2912, 32], [2000, 33], [1436, 35], [1000, 35], [728, 33],
                [500, 32],  [411, 30],  [250, 24],  [203, 20],  [125, 11]])

banana_area_x = banana_area[:,0]
banana_area_y = banana_area[:,1]
banana_area_x_log = 10*np.log10(banana_area_x)


fig=plt.figure(figsize=(6,2)) # 8x6 is the default (too large here)
ax = plt.gca()


ax.set_xlabel("f / Hz")
ax.set_ylabel('Sound Intensity / dBHL')
f_audiogram     = [125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000]
xticks = np.arange(len(f_audiogram))
#print(xticks)
#print([-0.5, xticks[-1] + 0.5])
#ax.set_xlim([-0.5, xticks[-1] + 0.5])
ax.set_ylim([-20, 120])
major_ticks = np.arange(-20, 120, 10)
minor_ticks = np.arange(-20, 120, 5)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')
ax.invert_yaxis()

#ax.set_yscale('log') # needs to be plotted on semilogx (not natively possible for fill (change a-points)
plt.fill(banana_area_x,banana_area_y,color='grey')
for n in range(len(item)):
    plt.text(item_freq[n], item_HL[n], s=item[n],
                verticalalignment='center', horizontalalignment='center',)
fig.savefig("banana.png")
fig.savefig("banana.svg")