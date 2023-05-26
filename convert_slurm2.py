import sys
with open(sys.argv[1]) as f:
    in_file = f.readlines()

for i in range(len(in_file)):

    in_file[i] = in_file[i].replace("main_out_new","main_out_new_learn")
    in_file[i] = in_file[i].replace("HL_", "HL_learn_")
    in_file[i] = in_file[i].replace("MAIN_HL_predict_intel_correctness","MAIN_HL_predict_intel_correctness_learn")
print(in_file[-1])
with open("learn_"+sys.argv[1], "w") as f:
    f.writelines(in_file)