import os
import laspy
import shutil
import random
import numpy as np
import torch
import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import math
import ast
import re 
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

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

split_dataset_into_folders('data/datasets_for_models/new_full_dataset', train_ratio=0.8, val_ratio=0.1)


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


def find_southwest_corner(folder_path):
    """
    Find the southwest corner of all point clouds in the specified folder.
    
    Args:
    - folder_path (str): Path to the folder containing the point cloud files.

    Returns:
    - (float, float): Coordinates of the southwest corner (min_x, min_y).
    """
    min_x, min_y = float('inf'), float('inf')

    for filename in os.listdir(folder_path):
        if filename.endswith('.las') or filename.endswith('.laz'):
            file_path = os.path.join(folder_path, filename)
            las_data = laspy.read(file_path)
            min_x = min(min_x, las_data.header.x_min)
            min_y = min(min_y, las_data.header.y_min)

    return min_x, min_y
def count_points_per_class(folder_path):
    """
    Count the number of points per class in a folder of LAZ point clouds.

    Args:
    - folder_path (str): Path to the folder containing LAZ point cloud files.

    Returns:
    - dict: A dictionary where keys are class labels and values are point counts.
    """
    class_counts = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.laz') or filename.endswith('.las'):
            file_path = os.path.join(folder_path, filename)
            las_data = laspy.read(file_path)

            # Assuming 'classification' attribute holds the class information
            for class_label in las_data.label:
                if class_label not in class_counts:
                    class_counts[class_label] = 0
                class_counts[class_label] += 1

    return class_counts

def read_laz_file(file_path):
    """
    Read a .laz file and extract the coordinate, color, intensity, and label data.
    """
    with laspy.open(file_path) as f:
        laz_data = f.read()
        
    coord = np.vstack((laz_data.x, laz_data.y, laz_data.z)).transpose()
    color = np.vstack((laz_data.red, laz_data.green, laz_data.blue)).transpose()
    intensity = laz_data.intensity
    segment = laz_data.label  # Assume this is already in the correct numeric format

    return coord, color, intensity, segment

def normalize_data(coord, color, intensity, use_rgb=True, use_intensity=True):
    """
    Normalize the data components.
    """
    # Coordinate offset normalization
    x_offset, y_offset = 122200, 483000
    coord[:, 0] -= x_offset
    coord[:, 1] -= y_offset

    # RGB Normalization
    if use_rgb:
        max_c = np.max(color)
        if max_c <= 1.:
            pass  # Already normalized
        elif max_c < 2**8:
            color = color / 255.
        elif max_c < 2**16:
            color = color / 65535.
        else:
            print('RGB more than 16 bits, not implemented. Aborting...')
            sys.exit(1)

    # Intensity Normalization
    if use_intensity:
        intensity = intensity / 65535.

    return coord, color, intensity

def convert_and_save(input_path, output_path):
    """
    Convert a .laz file to a .pth file with normalized structure.
    """
    coord, color, intensity, segment = read_laz_file(input_path)

    # Normalize the data
    coord, color, intensity = normalize_data(coord, color, intensity)

    # Update segment labels
    segment = np.where(segment == 2, 1, np.where(segment == 1, 0, segment))

    # Save the normalized data
    torch.save({
        "coord": coord,
        "color": color,
        "intensity": intensity,
        "segment": segment
    }, output_path)


def max_points_in_laz_folder(folder_path):
    """
    Returns the number of points and the filename of the largest point cloud file within a given folder.

    Args:
    folder_path (str): The path to the directory containing the .laz files.

    Returns:
    tuple: A tuple containing the number of points in the largest point cloud found in the folder and the filename of that point cloud.
    """
    max_points = 0  # Variable to keep track of the maximum number of points found
    max_file = ''   # Variable to store the name of the file with the most points

    # Iterate through each file in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".laz"):  # Check if the file is a LAZ file
            file_path = os.path.join(folder_path, filename)  # Create full file path
            with laspy.open(file_path) as file:  # Open the file using laspy
                las = file.read()  # Read the LAS data
                num_points = len(las.points)  # Count the number of points in the point cloud
                if num_points > max_points:
                    max_points = num_points  # Update max_points if current file has more points
                    max_file = filename     # Update max_file to the current file

    return max_points, max_file

def read_and_count_labels(laz_file_path):
    """
    Reads a LAZ file, counts occurrences of unique values in the 'final_label' scalar field, and returns the total number of points in the point cloud.
    
    Parameters:
    laz_file_path (str): Path to the LAZ file to be processed.
    
    Returns:
    tuple: A dictionary where keys are the unique values found in 'final_label' and values are the counts of each unique value, 
           along with the total number of points in the point cloud.
    
    Raises:
    ValueError: If the 'final_label' field does not exist in the point cloud.
    
    Example:
    label_counts, total_points = read_and_count_labels('path_to_your_file.laz')
    print("Label Counts:", label_counts)  # Output might look like: {1: 500, 2: 1500}
    print("Total Points:", total_points)  # Output might look like: 20000
    """
    # Open the LAZ file
    with laspy.open(laz_file_path) as file:
        # Read the point records from the file
        points = file.read()

        # Total number of points in the point cloud
        total_points = len(points)

        # Check if 'label_pred' exists in extra dimensions
        extra_dims = [dim.name for dim in points.point_format.extra_dimensions]
        if 'label_pred' not in extra_dims:
            raise ValueError("The scalar field 'label_pred' does not exist in this file.")

        # Retrieve the values from 'final_label' scalar field
        labels = points['label_pred']

        # Calculate and print counts of each unique value
        unique_labels, counts = np.unique(labels, return_counts=True)

    return dict(zip(unique_labels, counts)), total_points
    
def filter_and_save_excel(original_file_path, new_file_path):
    # Load the Excel file
    df = pd.read_csv(original_file_path, delimiter=';', encoding='latin1', header=0) 
   
    # Columns to check for non-empty values
    columns_to_check = ['Grond X', 'Grond Y', 'Grond Z', 'Top X', 'Top Y', 'Top Z']

    # New DataFrame with rows where all specified columns have non-empty values
    df_filtered = df.dropna(subset=columns_to_check)

    # Columns to keep in the new DataFrame
    columns_to_keep = [
        'Stadsdeel','Id', 'Code', 'Type', 'Oormerk Mast',
        'Grond X', 'Grond Y', 'Grond Z',
        'Top X', 'Top Y', 'Top Z',
        'Hoogte', 'Hoek', 'Afstand'
    ]

    # New DataFrame with only the desired columns
    df_final = df_filtered[columns_to_keep]

    # Save the new DataFrame to a new Excel file
    df_final.to_csv(new_file_path, index=False)

    # Output to confirm completion
    print(f"Filtered Excel file has been saved to: {new_file_path}")

def process_all_excel_files(source_folder, new_folder):
    # Iterate over all the files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file is an Excel file
        if filename.endswith('.csv'):
            # Construct the full path for the source and new file
            original_file_path = os.path.join(source_folder, filename)
            new_file_path = os.path.join(new_folder, filename)
            
            # Call the filter_and_save_excel function for each file
            filter_and_save_excel(original_file_path, new_file_path)


def combine_csv_files(folder_path):
    # List to hold DataFrames
    dfs = []
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # Read the CSV file and append to the list
            df = pd.read_csv(file_path, delimiter=',', quotechar='"')
            dfs.append(df)
    
    # Concatenate all DataFrames in the list into one DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save the combined DataFrame back to the same folder
    combined_csv_path = os.path.join(folder_path, 'combined_files_new.csv')
    combined_df.to_csv(combined_csv_path, index=False)
    
    return f"Combined CSV file has been saved to: {combined_csv_path}"


def assign_groups_by_similarity(df, cutoff, save_path=None):
    """
    Assigns group labels to the dataframe based on the similarity of strings in the 'Type' column,
    and optionally saves the modified DataFrame to a CSV file.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'Type' column to group by similarity.
        cutoff (int): The cutoff percentage score for fuzzy matching to consider two strings as similar.
        save_path (str, optional): Path to save the modified DataFrame as a CSV file. If None, no file is saved.

    Returns:
        pd.DataFrame: A modified DataFrame with an additional column indicating the group label.
    """
    # Get unique types and preprocess them
    strings = df['Type'].unique().tolist()
    strings = [str(item) for item in strings if item is not None]

    # This will store the groups
    groups = []

    # Mapping of string to group label
    string_to_group = {}

    for string in strings:
        best_match = None
        best_score = cutoff  # Only consider matches above the cutoff

        # Check each group to find the best match
        for group_index, group in enumerate(groups):
            score = process.extractOne(string, group)[1]
            if score > best_score:
                best_score = score
                best_match = group_index

        # If a suitable group is found, add to it
        if best_match is not None:
            groups[best_match].append(string)
            string_to_group[string] = f'Group {best_match + 1}'
        else:
            # If no existing group is similar enough, start a new group
            groups.append([string])
            string_to_group[string] = f'Group {len(groups)}'

    # Apply the mapping to the DataFrame to create a new column for the group labels
    df[f'fuzzywuzzy_cutoff_{cutoff}'] = df['Type'].apply(lambda x: string_to_group.get(str(x), 'No Group'))

    # Save the DataFrame to a CSV file if a path is provided
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"DataFrame saved to {save_path}")

    print("Number of groups : ", len(groups))

    return df

def assign_groups_by_text_clustering(df, eps, min_samples, save_path=None):
    """
    Assigns group labels to the dataframe based on the clustering of strings in the 'Type' column,
    using text similarity and DBSCAN clustering, and optionally saves the modified DataFrame to a CSV file.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'Type' column to cluster by text similarity.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        save_path (str, optional): Path to save the modified DataFrame as a CSV file. If None, no file is saved.

    Returns:
        pd.DataFrame: A modified DataFrame with an additional column indicating the cluster label.
    """
    # Get unique types and preprocess them
    types = df['Type'].dropna().unique()
    
    # Vectorize the types using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform(types)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(vectorizer)
    similarity_matrix = np.clip(similarity_matrix, 0, 1)

    # Cluster with DBSCAN using a precomputed similarity matrix (converted to distances)
    dbscan = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(1 - similarity_matrix)  # Convert similarity to distance

    # Map each unique type to its cluster
    type_to_cluster = {type_: f"Cluster {cluster}" for type_, cluster in zip(types, clusters)}

    # Apply the mapping to the DataFrame to create a new column for the cluster labels
    df[f'text_cluster_{eps}_{min_samples}'] = df['Type'].map(type_to_cluster).fillna('No Cluster')

    # Save the DataFrame to a CSV file if a path is provided
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"DataFrame saved to {save_path}")

    print("Number of clusters: ", len(set(clusters) - {-1}))  # Exclude noise points

    return df

def open_and_display_images(filenames, folder_path):
    """
    Opens and displays image files in a specified folder that match a list of filenames.

    Parameters:
        filenames (list): List of filenames to match, expected to be image files.
        folder_path (str): Path to the folder where files are stored.
    """
    folder_path = os.path.join(folder_path, '')  # Ensure folder path ends with a separator

    for filename in filenames:
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            img = mpimg.imread(full_path)  # Load the image file
            plt.imshow(img)  # Display the image
            plt.title(filename)  # Set the title of the figure to the filename
            plt.axis('off')  # Turn off axis numbers and ticks
            plt.show()  # Display the plot window
        else:
            print(f"No file found for {filename}")

def transfer_and_label_files(mapping, input_folder, output_folder):
    """
    Transfer files from input_folder to output_folder and label them.

    Parameters:
        mapping (dict): A dictionary where keys are string identifiers and values are numerical identifiers (file names).
        input_folder (str): Path to the directory containing the source files.
        output_folder (str): Path to the directory where files should be transferred and labeled.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Font settings (default font with size 20)
    font = ImageFont.load_default()
    font_size = 20
    color = 'black'
    
    for string_id, number_id in mapping.items():
        # Check if the string identifier is NaN
        if pd.isna(string_id):
            continue
        safe_string_id = sanitize_filename(string_id)

        # Construct file paths
        src_file_name = f"{number_id}.png"
        src_path = os.path.join(input_folder, src_file_name)
        dst_file_name = f"{safe_string_id}.png"
        dst_path = os.path.join(output_folder, dst_file_name)
        # Check if file exists before proceeding
        if os.path.exists(src_path):
            # Open the image
            with Image.open(src_path) as img:
                # Draw the string identifier on the image
                draw = ImageDraw.Draw(img)
                # Correctly calculate the text size
                text_width, text_height = draw.textbbox((0, 0), string_id, font=font)[2:]
                # Position text at the bottom right
                x = img.width - text_width - 10
                y = img.height - text_height - 10
                draw.text((x, y), string_id, font=font, fill=color)
                
                # Save the modified image to the output folder
                img.save(dst_path)
        else:
            print(f"File {src_path} does not exist.")

def sanitize_filename(name):
    """
    Sanitize the filename by replacing invalid characters with a safe alternative.
    
    Parameters:
        name (str): The original filename.
        
    Returns:
        str: The sanitized filename, replacing or removing illegal characters.
    """
    # Replace slashes with a safe character
    safe_name = name.replace("/", "-")
    # Remove or replace other potentially problematic characters
    illegal_chars = '<>:"/\\|?*'
    for char in illegal_chars:
        safe_name = safe_name.replace(char, "")
    # Replace non-ASCII characters with an underscore
    safe_name = ''.join([c if ord(c) < 128 else '_' for c in safe_name])
    return safe_name.strip()

def map_lowest_ids_to_types(df):
    """
    Maps each unique string in the 'Type' column of the DataFrame to the lowest 'Id' of a streetlight.

    Parameters:
        df (pd.DataFrame): The DataFrame to process, which must contain 'Type' and 'Id' columns.

    Returns:
        dict: A dictionary with keys as unique 'Type' strings and values as the lowest 'Id's.
    """
    # Sort the DataFrame by 'Id' in ascending order
    sorted_df = df.sort_values(by='Id', ascending=True)

    # Drop duplicate 'Type' values, keeping the first occurrence with the lowest 'Id'
    unique_types = sorted_df.drop_duplicates(subset='Code')

    # Create a dictionary mapping 'Type' to the lowest 'Id'
    type_to_id_map = pd.Series(unique_types['Id'].values, index=unique_types['Code']).to_dict()

    return type_to_id_map

def organize_images(txt_file, folder_path):
    # Create a dictionary to hold the file names under their respective labels
    files_dict = {}

    # Read the txt file and fill the dictionary
    with open(txt_file, 'r') as file:
        for line in file:
            if ':' in line:  # Check if the line contains a label and list
                label, files_str = line.split(':')
                label = label.strip()
                try:
                    # Use ast.literal_eval for safe evaluation
                    files_list = ast.literal_eval(files_str.strip())
                except SyntaxError as e:
                    print(f"Error processing line: {line.strip()}")
                    print(e)
                    continue
                files_dict[label] = files_list

    # Iterate over the dictionary and organize files
    for label, files_list in files_dict.items():
        # Create the subfolder if it doesn't exist
        subfolder_path = os.path.join(folder_path, label)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        
        # Move each file to the new subfolder
        for filename in files_list:
            original_file = os.path.join(folder_path, sanitize_filename(filename) + '.png')
            if os.path.exists(original_file):  # Check if the file exists before moving
                shutil.move(original_file, subfolder_path)
            else:
                print(f"File not found: {original_file}")

def sanitize_filename(name):
    safe_name = name.replace("/", "-")
    illegal_chars = '<>:"/\\|?*'
    for char in illegal_chars:
        safe_name = safe_name.replace(char, "")
    safe_name = ''.join([c if ord(c) < 128 else '_' for c in safe_name])
    return safe_name.strip()

def assign_cluster(df, folder_path, save_path = None):
    # Dictionary to hold the mapping from sanitized filenames back to subfolder names
    file_to_cluster = {}

    df['Code'] = df['Code'].astype(str)

    # Loop over each subfolder in the main folder
    for cluster in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, cluster)
        if os.path.isdir(subfolder_path):
            # Loop over each file in the subfolder
            for filename in os.listdir(subfolder_path):
                # Remove file extension if needed (assuming all are .png for example)
                raw_name = os.path.splitext(filename)[0]
                file_to_cluster[raw_name] = cluster

    # Sanitize the 'Code' values in the DataFrame to match the filenames
    df['Sanitized_Code'] = df['Code'].apply(sanitize_filename)

    # Map sanitized codes to cluster based on the file_to_cluster dictionary
    df['Cluster'] = df['Sanitized_Code'].map(file_to_cluster)

    # Optionally, remove the 'Sanitized_Code' column if it's no longer needed
    df.drop('Sanitized_Code', axis=1, inplace=True)
    if save_path:
        df.to_csv(save_path, index=False)

    return df

def retrieve_clusters(csv_file, point_cloud_folder):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert the coordinate separators from commas to dots and to float
    df['Grond X'] = df['Grond X'].apply(lambda x: float(x.replace(',', '.')))
    df['Grond Y'] = df['Grond Y'].apply(lambda x: float(x.replace(',', '.')))

    # Dictionary to store cluster counts and unlabelled count
    cluster_counts = {}
    unlabeled_count = 0

    # Iterate over each point cloud file
    for filename in os.listdir(point_cloud_folder):
        # Check if the file is a point cloud file
        if filename.endswith('.laz'):
            # Extract the coordinates from the filename
            match = re.match(r'final_(\d+)_(\d+).laz', filename)
            if match:
                # Multiply extracted coordinates by 50 to match the scale of 'Grond X' and 'Grond Y'
                tile_x, tile_y = int(match.group(1)) * 50, int(match.group(2)) * 50

                # Find rows where the tile coordinates match, and Stadsdeel is 'Oost'
                matching_rows = df[(df['Grond X'] >= tile_x) & (df['Grond X'] < tile_x + 50) &
                                   (df['Grond Y'] >= tile_y) & (df['Grond Y'] < tile_y + 50) &
                                   (df['Stadsdeel'] == 'Oost')]

                # Update cluster counts and track unlabeled entries
                for index, row in matching_rows.iterrows():
                    cluster = row['Cluster']
                    if pd.isna(cluster) or cluster == '':
                        unlabeled_count += 1
                    else:
                        if cluster in cluster_counts:
                            cluster_counts[cluster] += 1
                        else:
                            cluster_counts[cluster] = 1

    return cluster_counts, unlabeled_count

def sample_diverse_streetlights_from_cluster(csv_path, cluster_name, num_samples, save_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    df = df[df['Stadsdeel'] != 'Westpoort']
    # Filter the DataFrame for the specified cluster
    df_cluster = df[df['Cluster'] == cluster_name]

    # Shuffle the DataFrame to randomize the row order
    df_cluster = df_cluster.sample(frac=1).reset_index(drop=True)

    # Container to hold selected samples
    sampled_indices = set()

    # First, try to pick as many unique 'Code' values as possible
    unique_codes = pd.unique(df_cluster['Code'])
    for code in unique_codes:
        if len(sampled_indices) < num_samples:
            # Select a random sample for each unique 'Code'
            code_entries = df_cluster[df_cluster['Code'] == code]
            selected_index = code_entries.sample(n=1).index.item()
            sampled_indices.add(selected_index)
        else:
            break

    # If not enough samples, fill the rest randomly
    if len(sampled_indices) < num_samples:
        needed_samples = num_samples - len(sampled_indices)
        remaining_indices = set(df_cluster.index) - sampled_indices
        additional_samples = pd.Index(remaining_indices).to_series().sample(n=needed_samples)
        sampled_indices.update(additional_samples.values)

    # Get the final sampled DataFrame
    sampled_df = df_cluster.loc[list(sampled_indices)]

    # Save the sampled DataFrame to the specified path
    sampled_df.to_csv(save_path, index=False)
    
    return sampled_df

def update_cluster_value(input_csv_path, output_csv_path):
    """
    Loads a DataFrame from a CSV file, changes specific values in the 'Cluster' column, 
    and saves the modified DataFrame to a new file.

    Parameters:
    input_csv_path (str): The path to the input CSV file.
    output_csv_path (str): The path where the modified CSV file will be saved.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv_path)

    # Replace 'CHELOU SUR LA TIGE' with 'MULTIPLE LIGHTS' in the 'Cluster' column
    df['Cluster'] = df['Cluster'].replace('CHELOU SUR LA TIGE', 'MULTIPLE LIGHTS')

    # Save the modified DataFrame to the new path
    df.to_csv(output_csv_path, index=False)


def count_files_without_label_field(directory):
    """
    Counts the number of .laz files in a specified directory that do not contain a 'label' scalar field.

    Parameters:
    directory (str): The path to the directory containing .laz files.

    Returns:
    int: The number of files without the 'label' scalar field.
    """
    count = 0
    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.laz'):
            file_path = os.path.join(directory, filename)
            try:
                with laspy.open(file_path) as file:
                    las = file.read()
                    # Check if 'label' field is present in the point format
                    if 'label' not in las.point_format.dimension_names:
                        count += 1
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
    
    return count

def downsample_lidar_point_cloud(input_file, output_file, voxel_size=0.05):
    """
    Downsamples a LiDAR point cloud using voxel downsampling, preserving color information.

    Parameters:
    - input_file (str): Path to the input LiDAR point cloud file.
    - output_file (str): Path where the downsampled point cloud should be saved.
    - voxel_size (float): Size of the voxel grid (smaller values keep more detail).

    Returns:
    - None
    """

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(input_file)

    # Check if the point cloud has colors
    if pcd.colors:
        print("Point cloud has color information. Downsampling will preserve colors.")
    else:
        print("Point cloud has no color information.")

    # Voxel downsampling
    down_pcd = pcd.voxel_down_sample(voxel_size)

    # Save the downsampled point cloud
    o3d.io.write_point_cloud(output_file, down_pcd)
    print(f"Downsampled point cloud saved to {output_file}")
downsample_lidar_point_cloud('data\points_clouds_laz_ds/bb_529 - Cloud.las', 'data\points_clouds_laz_ds/final_2456_9707.las', voxel_size=0.05)

# print(count_files_without_label_field('data/datasets_for_models/missing_bb_dataset'))
# update_cluster_value('data\sheets\processed_sheets\combined_files_clustered.csv', 'data\sheets\processed_sheets\combined_files_clustered_final.csv')
# current_counts = {
#     'GRACHT': 72, 'ONDULE': 38, 'CONIQUES': 67, 'LONG': 69, 
#     'DOUBLE': 10, 'UNLABELED': 28, 'CHELOU SUR LA TIGE': 3
# }

# sample_diverse_streetlights_from_cluster('data/sheets/processed_sheets/combined_files_clustered.csv', 'SQUARED PEND', 40, 'data/sheets/samples_to_add/sq_p_samples.csv')
# df = pd.read_csv('data/sheets/processed_sheets/combined_files_new.csv')
# print(retrieve_clusters('data/sheets/processed_sheets/combined_files_clustered.csv', 'data/points_clouds_laz'))

# def count_files_in_subfolders(folder_path):
#     # Dictionary to hold the count of files in each subfolder
#     file_counts = {}
#     total_files = 0

#     # Walk through the directory structure
#     for root, dirs, files in os.walk(folder_path):
#         for d in dirs:
#             subfolder_path = os.path.join(root, d)
#             count = len([name for name in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, name))])
#             file_counts[d] = count
#             total_files += count

#     # Calculate the percentage of total files for each subfolder
#     file_percentages = {folder: (count / total_files * 100) if total_files > 0 else 0 for folder, count in file_counts.items()}

#     return file_counts, file_percentages

# print(count_files_in_subfolders('data/images/unique_codes'))
# df = pd.read_csv('data/sheets/processed_sheets/combined_files_new.csv')
# assign_cluster(df, 'data/images/unique_codes', 'data/sheets/processed_sheets/combined_files_new_clustered.csv')
# type_to_id_map = map_lowest_ids_to_types(df)
# transfer_and_label_files(type_to_id_map, 'data/images/matches', 'data/images/unique_codes')
# combine_csv_files('data/processed_sheets')
# df = pd.read_csv('data/processed_sheets/combined_files_new_grouped.csv')
# types = df['Code'].dropna().unique()
# X = TfidfVectorizer().fit_transform(types)

# distance_matrix = 1 - cosine_similarity(X)

# # Sort each row and select the k-th nearest, consider k as the square root of the number of samples
# k = int(np.sqrt(len(types)))
# kth_distances = np.sort(distance_matrix, axis=1)[:, k]

# # Plotting the k-distance graph
# plt.plot(np.sort(kth_distances))
# plt.ylabel('k-th Nearest Distance')
# plt.title('K-Distance Plot')
# plt.show()