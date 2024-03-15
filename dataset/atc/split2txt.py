import os
import random

# Define the root directory of the dataset
root_dir = './data'

# Lists to hold all file paths
all_files = []

# Walk through the directory structure
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.wav'):
            # Construct relative path w.r.t. root_dir for class label consistency
            relative_path = os.path.relpath(subdir, root_dir)
            all_files.append(f"{relative_path}/{file}")

# Shuffle the data to randomize the train/test split
random.shuffle(all_files)

# Define the split ratio
split_ratio = 0.8
split_index = int(len(all_files) * split_ratio)

# Split the data
train_files = all_files[:split_index]
test_files = all_files[split_index:]

# Function to write files to txt
def write_to_txt(file_list, file_name):
    with open(file_name, 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)

# Write train and test files
write_to_txt(train_files, './train.txt')
write_to_txt(test_files, './test.txt')
