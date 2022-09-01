import os
import csv
import shutil

base_folder = "../datasets/traffic/Test_2/"
# for i in range(42):
#     path = base_folder + str(i)
#     os.mkdir(path)

data_path = "../datasets/traffic/Test.csv"

with open(data_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            line_count += 1
            # print(row)
            # img_paths.append(row[-1])
            # labels.append(int(row[-2]))
            img_path = row[-1]
            img_label = int(row[-2])
            if img_label == 42:
                dst_folder = base_folder + str(img_label)
                final_file_path = "../datasets/traffic/" + img_path
                shutil.copy(final_file_path, dst_folder)
