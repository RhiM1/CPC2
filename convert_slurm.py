import sys
with open(sys.argv[1]) as f:
    in_file = f.readlines()

for i in range(len(in_file)):

    in_file[i] = in_file[i].replace("_HL_","_").replace("HL_","_").replace("HL-","").replace("_HL_","_").replace("_hl_","_").replace("main_out_new","main_out_new_no_HL")
print(in_file[-1])
with open("new_"+sys.argv[1].strip("_HL"), "w") as f:
    f.writelines(in_file)