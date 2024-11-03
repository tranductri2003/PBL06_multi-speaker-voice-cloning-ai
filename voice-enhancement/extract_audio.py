import os
import shutil


def extract_flac_files(source_dir, destination_dir):
    """
    Move all .flac files from source_dir and its subfolders into destination_dir.
    """
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Created destination directory: {destination_dir}")

    # Traverse through all folders and files in the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.flac'):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(destination_dir, file)

                # Check if the file already exists in the destination directory
                if os.path.exists(destination_file):
                    # If it does, append a counter to the file name to avoid overwriting
                    base, extension = os.path.splitext(file)
                    count = 1
                    new_file = f"{base}_{count}{extension}"
                    destination_file = os.path.join(destination_dir, new_file)
                    while os.path.exists(destination_file):
                        count += 1
                        new_file = f"{base}_{count}{extension}"
                        destination_file = os.path.join(destination_dir, new_file)

                # Copy the file
                shutil.copy2(source_file, destination_file)
                print(f"Copied: {source_file} -> {destination_file}")

    print("Completed extracting .flac files.")


if __name__ == "__main__":
    source_directory = "/Users/triductran/SpartanDev/my-work/Speech-enhancement/data/Total/clean_voice/LibriSpeech/train-clean-360"
    destination_directory = "/Users/triductran/SpartanDev/my-work/Speech-enhancement/data/Total/clean_voice_audio"

    extract_flac_files(source_directory, destination_directory)
