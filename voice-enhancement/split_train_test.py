import os
import shutil
import random


def split_flac_files(source_dir, train_dir, test_dir, train_ratio=0.8):
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return

    # Create train and test directories if they don't already exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # List all .flac files in the source directory
    all_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.flac')]
    total_files = len(all_files)

    if total_files == 0:
        print("No .flac files found in the source directory.")
        return

    print(f"Total .flac files found: {total_files}")

    # Shuffle the list of files randomly
    random.shuffle(all_files)

    # Calculate the number of files for train and test
    train_count = int(total_files * train_ratio)
    test_count = total_files - train_count

    print(f"Number of files to be placed in the train directory: {train_count}")
    print(f"Number of files to be placed in the test directory: {test_count}")

    # Split the file list
    train_files = all_files[:train_count]
    test_files = all_files[train_count:]

    # Function to copy files
    def copy_files(file_list, destination_dir):
        for file_name in file_list:
            src_path = os.path.join(source_dir, file_name)
            dest_path = os.path.join(destination_dir, file_name)
            shutil.copy2(src_path, dest_path)
            print(f"Copied: {src_path} -> {dest_path}")

    # Copy files into the train directory
    print("\nStarting to copy files into the train directory...")
    copy_files(train_files, train_dir)

    # Copy files into the test directory
    print("\nStarting to copy files into the test directory...")
    copy_files(test_files, test_dir)

    print("\nCompleted splitting files into train and test directories.")


if __name__ == "__main__":
    source_directory = "/Users/triductran/SpartanDev/my-work/Speech-enhancement/data/Total/clean_voice_audio"
    train_directory = "/Users/triductran/SpartanDev/my-work/Speech-enhancement/data/Train/clean_voice"
    test_directory = "/Users/triductran/SpartanDev/my-work/Speech-enhancement/data/Test/clean_voice"

    split_flac_files(source_directory, train_directory, test_directory, train_ratio=0.8)
