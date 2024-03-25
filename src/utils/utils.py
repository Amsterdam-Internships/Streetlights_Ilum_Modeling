import os
import laspy
import shutil
import random
import numpy as np

def average_points_in_laz_files(directory):
    total_points = 0
    file_count = 0

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".laz"):
            file_path = os.path.join(directory, filename)
            try:
                with laspy.open(file_path) as laz_file:
                    las = laz_file.read()
                    total_points += len(las.points)
                    file_count += 1
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Calculate the average if any files were processed
    if file_count > 0:
        average_points = total_points / file_count
        return average_points
    else:
        return "No LAZ files found or readable."
    
def modify_labels(input_folder, output_folder):
    """
    Modifies the 'label' scalar field in LAZ files within the specified input folder and 
    saves the modified point clouds to a new folder. This function changes the 'label' value 
    from 2 to 1, and from 1 to 0, then writes the modified point cloud to a new LAZ file in the output folder.

    Parameters:
    input_folder (str): Path to the folder containing the original LAZ files.
    output_folder (str): Path to the folder where modified LAZ files will be saved.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".laz"):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            with laspy.open(input_file_path) as file:
                las = file.read()

                # Check if 'label' field exists
                if 'label' in las.point_format.dimension_names:
                    # Modify labels
                    labels = las.label
                    labels[labels == 2] = 1
                    labels[labels == 1] = 0
                    las.label = labels

                    # Write to new file
                    las.write(output_file_path)
                else:
                    print(f"'label' field not found in file {filename}")

def split_dataset_into_folders(base_folder, train_ratio=0.8, val_ratio=0.1):
    """
    Splits the files in a folder into train, validation, and test folders 
    according to specified ratios.

    Parameters:
    base_folder (str): Path to the base folder containing the files to split.
    train_ratio (float): Proportion of files to be used for training.
    val_ratio (float): Proportion of files to be used for validation.

    The test ratio is automatically calculated as (1 - train_ratio - val_ratio).
    """

    # Paths for train, validation, and test folders
    train_folder = os.path.join(base_folder, 'train')
    val_folder = os.path.join(base_folder, 'val')
    test_folder = os.path.join(base_folder, 'test')

    # Create directories if they don't exist
    for folder in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # List all files in the base folder
    all_files = [f for f in os.listdir(base_folder) if os.path.isfile(os.path.join(base_folder, f))]
    random.shuffle(all_files)  # Shuffle to randomize file selection

    # Calculate the number of files for train, val, and test
    num_files = len(all_files)
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    num_test = num_files - num_train - num_val

    # Split files into train, val, and test
    for i, file in enumerate(all_files):
        src_file = os.path.join(base_folder, file)

        if i < num_train:
            dest_folder = train_folder
        elif i < num_train + num_val:
            dest_folder = val_folder
        else:
            dest_folder = test_folder

        shutil.move(src_file, os.path.join(dest_folder, file))

def is_point_cloud_match(file1, file2):
    """
    Compares two LAZ files based on their xyz coordinates and RGB values.
    Returns True if they match, False otherwise.
    """
    with laspy.open(file1) as f1, laspy.open(file2) as f2:
        pc1, pc2 = f1.read(), f2.read()

        # Compare xyz coordinates
        coords_match = np.array_equal(pc1.xyz, pc2.xyz)

        # Check if RGB data exists and compare it
        rgb_match = True  # Assume true if RGB data is not present
        if all(hasattr(pc1, color) and hasattr(pc2, color) for color in ['red', 'green', 'blue']):
            rgb1 = np.vstack((pc1.red, pc1.green, pc1.blue)).T
            rgb2 = np.vstack((pc2.red, pc2.green, pc2.blue)).T
            rgb_match = np.array_equal(rgb1, rgb2)

        return coords_match and rgb_match

def copy_distinct_laz_files(folder1, folder2, output_folder):
    """
    Copies LAZ files from folder2 to output_folder if they are distinct
    (in terms of xyz and RGB) from all files in folder1.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file2 in os.listdir(folder2):
        if file2.endswith(".laz"):
            file2_path = os.path.join(folder2, file2)
            match_found = False

            for file1 in os.listdir(folder1):
                if file1.endswith(".laz"):
                    file1_path = os.path.join(folder1, file1)
                    if is_point_cloud_match(file1_path, file2_path):
                        match_found = True
                        break
            
            if not match_found:
                shutil.copy(file2_path, os.path.join(output_folder, file2))

# Example usage
# folder1 = 'data/bb_extracted_data_labelled/bb_with_rgb'
# folder2 = 'data/bb_extracted_data_unlabelled'
# output_folder = 'data/bb_extracted_data_labelled/bb_to_label'
# copy_distinct_laz_files(folder1, folder2, output_folder)

# # Example usage
# base_folder_path = 'data/bb_extracted_data_labelled'
# split_dataset_into_folders(base_folder_path)

def rename_files_to_pattern(directory):
    """
    Renames all files in the specified directory to 'bb_<number>.laz',
    where <number> starts at 1 and increments for each file.
    """
    # Get a list of all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Iterate through the files and rename them
    for index, file_name in enumerate(files, start=1):
        new_file_name = f'bb_{index}.laz'
        original_file_path = os.path.join(directory, file_name)
        new_file_path = os.path.join(directory, new_file_name)

        try:
            os.rename(original_file_path, new_file_path)
            print(f"Renamed: {file_name} to {new_file_name}")
        except Exception as e:
            print(f"Error renaming {file_name}: {e}")

# Usage
directory = 'data/bb_extracted_data_labelled/full dataset'  # Replace with your directory path
rename_files_to_pattern(directory)