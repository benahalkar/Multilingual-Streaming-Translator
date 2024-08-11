import os

curr_path = os.getcwd()

input_file = os.path.join(curr_path, "dataset_files", "pmindia.v1.hi-en.tsv")
output_file = os.path.join(curr_path, "dataset_files", "dataset.txt")

with open(input_file, "r") as in_file, open(output_file, 'w') as out_file:
    for line in in_file:
        line = line.split('\t')[0] + " "
        out_file.write(line)
            
    in_file.close()
    out_file.close()