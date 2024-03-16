import os
import random

# Define the root directory of the dataset and the unknown rate
root_dir = './data'
unknown_rate = 1.0  # Use 50% of files from the unknown folder

# Lists to hold all file paths
all_files = []
unknown_files = []

# Walk through the directory structure
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.wav'):
            relative_path = os.path.relpath(subdir, root_dir)
            if relative_path == 'unknown':  # Check if the file is in the unknown class
                unknown_files.append(f"{relative_path}/{file}")
            else:
                all_files.append(f"{relative_path}/{file}")

# Apply unknown rate to the unknown files
selected_unknown_files = random.sample(unknown_files, int(len(unknown_files) * unknown_rate))

for i in range(len(selected_unknown_files)):
    selected_unknown_files.append(f"unknown/silence")
# Add the selected unknown files to the all_files list
all_files.extend(selected_unknown_files)

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
write_to_txt(train_files, './train_withunkeyword.txt')
write_to_txt(test_files, './test_withunkeyword.txt')
