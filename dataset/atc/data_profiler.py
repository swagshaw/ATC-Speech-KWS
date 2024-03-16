def profile_dataset(filenames):
    """
    Reads dataset files, counts the number of clips per class, and prints the dataset profile.
    :param filenames: List of filenames (e.g., ['train.txt', 'test.txt']) to profile.
    """
    from collections import defaultdict

    # Initialize a dictionary to hold counts of clips per class
    class_counts = defaultdict(int)

    # Process each file
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                class_label = line.strip().split('/')[0]  # Extract class label
                class_counts[class_label] += 1

    # Output the profile
    total_clips = sum(class_counts.values())
    print(f"Total number of clips: {total_clips}")
    print(f"Number of classes: {len(class_counts)}")
    print("Clips per class:")
    for class_label, count in class_counts.items():
        print(f"  {class_label}: {count}")
def profile_split(filename):
    """
    Profiles a single dataset split, counting the number of clips per class.
    :param filename: Filename of the dataset split to profile.
    :return: A dictionary with class labels as keys and clip counts as values, and the total clip count.
    """
    from collections import defaultdict

    class_counts = defaultdict(int)

    with open(filename, 'r') as file:
        for line in file:
            class_label = line.strip().split('/')[0]  # Extract class label
            class_counts[class_label] += 1

    total_clips = sum(class_counts.values())

    return class_counts, total_clips

def print_profile(class_counts, total_clips, split_name):
    """
    Prints the profile of a dataset split.
    :param class_counts: Dictionary with class labels and clip counts.
    :param total_clips: Total number of clips in the split.
    :param split_name: Name of the dataset split (e.g., 'train' or 'test').
    """
    print(f"\n{split_name.capitalize()} Dataset Profile")
    print(f"Total number of clips: {total_clips}")
    print(f"Number of classes: {len(class_counts)}")
    print("Clips per class:")
    for class_label, count in class_counts.items():
        print(f"  {class_label}: {count}")


# Assuming 'train.txt' and 'test.txt' are in the current directory
filenames = ['./train_withunkeyword.txt', './test_withunkeyword.txt']
profile_dataset(filenames)
class_counts, total_clips = profile_split(filenames[0])
print_profile(class_counts, total_clips, "train")
class_counts, total_clips = profile_split(filenames[1])
print_profile(class_counts, total_clips, "test")