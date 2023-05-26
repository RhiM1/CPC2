import matplotlib.pyplot as plt

import pandas as pd


in_data_CEC1_1 = pd.read_json("data/clarity_data/metadata/CEC1.train.1.json")
#in_data_CEC1_2 = pd.read_json("data/clarity_data/metadata/CEC1.train.2.json")
#in_data_CEC1_3 = pd.read_json("data/clarity_data/metadata/CEC1.train.3.json")
in_data_CEC2_1 = pd.read_json("data/clarity_data/metadata/CEC2.train.1.json")
#in_data_CEC2_2 = pd.read_json("data/clarity_data/metadata/CEC2.train.2.json")
#in_data_CEC2_3 = pd.read_json("data/clarity_data/metadata/CEC2.train.3.json")
#in_data = pd.concat([in_data_CEC1_1,in_data_CEC1_2,in_data_CEC1_3,in_data_CEC2_1,in_data_CEC2_2,in_data_CEC2_3])
in_data = pd.concat([in_data_CEC1_1,in_data_CEC2_1])
#print(in_data["correctness"])

fig = plt.figure(figsize=(10, 6), dpi=80)
plt.rcParams.update({'font.size': 20})
plt.subplot(2,1,1)

average_correctness = in_data["correctness"].mean()
values,bins,bars = plt.hist(in_data["correctness"],histtype='bar', ec='black',bins=10,alpha=0.5,color='orange')
plt.bar_label(bars,fmt='%d\n\n',padding=1,label_type='center',size="small")

plt.axvline(x=average_correctness, color='k', linestyle='--',label="Average Correctness")
plt.xlabel(r"Correctness $i$")
plt.ylabel(r"Count")
plt.tight_layout()


l_dict = {}
for listener in in_data["listener"].unique():
    average_correctness = in_data[in_data["listener"]==listener]["correctness"].mean()
    print(listener,average_correctness)
    l_dict[listener] = average_correctness
overall_average_correctness = in_data["correctness"].mean()
l_dict = dict(sorted(l_dict.items(), key=lambda item: item[1]))
plt.subplot(2,1,2)
plt.bar(l_dict.keys(),l_dict.values(),color='red',alpha=0.5,edgecolor='black')
for v in l_dict:
    plt.text(v, l_dict[v]-20, r'{:.1f}'.format(l_dict[v]),size="small", horizontalalignment='center',verticalalignment='center', rotation=90)
plt.axhline(y=overall_average_correctness, color='k', linestyle='--',label="Overall Average Correctness")
#plt.text(0.15, overall_average_correctness/100+0.02, r'Overall Average $i$',size="small", horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes)
#plt.xticks(rotation=45)
plt.xticks(rotation=45, ha="right",rotation_mode="anchor",size="small")
plt.ylim(0,100)

plt.xlabel(r"Listener")
plt.ylabel(r"Average Correctness $i$")
fig.align_ylabels()
plt.tight_layout()
plt.savefig("corr_bar.png")
plt.savefig("corr_bar.svg")

plt.close()