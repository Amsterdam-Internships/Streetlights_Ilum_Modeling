import os
import numpy as np
import open3d as o3d
import laspy 

from pyntcloud import PyntCloud
from plyfile import PlyData, PlyElement


"""
Point Cloud Processing Toolkit

This collection of Python functions is designed to facilitate the manipulation and processing of point cloud data, particularly in PLY format. Utilizing libraries such as `plyfile` and `numpy`, these functions offer a range of capabilities to work with point cloud files effectively. 

Key Features:

1. Modification of Scalar Fields:
   - `add_label_streetlight_to_ply(input_directory, output_directory)`: This function iterates through all PLY files in a given input directory. For each point cloud, it reads an existing scalar field named 'label'. Based on this field, it creates a new scalar field called 'label_streetlight'. The value of 'label_streetlight' is set to 2 if the original 'label' is 2, and 1 otherwise. The modified point clouds are saved in a specified output directory, preserving the original data structure and additional attributes.

2. Conversion Between PLY and LAZ Formats:
   - `ply_to_laz(directory, output_directory)`: This function allows the conversion of PLY files to the compressed LAZ format. It's particularly useful for reducing the size of point cloud files for storage or transmission. The function maintains the integrity of the original data, including custom scalar fields like 'label'. Converted files are stored in a separate output directory.

Usage and Application:
These functions are ideal for scenarios where point cloud data needs to be preprocessed, such as adding new data fields based on existing ones or converting file formats for compatibility with different point cloud processing tools. They can be particularly useful in domains like 3D modeling, geographic information systems (GIS), and LiDAR data processing.

Prerequisites:
- Python environment with packages `numpy`, `plyfile`, and `laspy` installed.
- Input PLY files should be structured correctly with necessary fields, especially for the `label` field required by some functions.
- Ensure you have read/write permissions for the specified directories.

Example Usage:
# To add a new scalar field 'label_streetlight' to PLY files:
# add_label_streetlight_to_ply("path/to/input/ply_directory", "path/to/output/ply_directory")

# To convert PLY files to LAZ format:
# ply_to_laz("path/to/input/ply_directory", "path/to/output/laz_directory")

Note:
This toolkit is designed for batch processing of files within directories. It is advisable to backup your data before performing batch operations. The toolkit is easily extendable for more custom point cloud processing functionalities.
"""


def delete_files_without_label_2(directory):
    """
    Delete all PLY files in a specified directory that do not have label 2.

    This function iterates over all files in the provided directory. For each file,
    if it is a PLY file and does not have the label '2.000000', it will be deleted.

    Parameters:
    directory (str): The path to the directory where the PLY files are located.

    Returns:
    None

    Note: This function depends on a helper function 'has_label_2' to check for the label.
    Ensure that 'has_label_2' is defined and correctly implemented.
    """
    for filename in os.listdir(directory):
        # Check if the file is a PLY file
        if filename.endswith('.ply'):
            file_path = os.path.join(directory, filename)
            
            # Check for label 2 and delete the file if it doesn't have it
            if not has_label_2(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")

def has_label_2(file_path):
    """
    Check if a PLY file contains any point with the label 2.000000.

    This function reads the PLY file located at the given file path. It then checks
    if there is a 'label' field in the vertex data of the PLY file. If the 'label' field
    exists, the function checks whether any of the points in the file have a label
    value of 2.000000.

    Parameters:
    file_path (str): The path to the PLY file.

    Returns:
    bool: True if any point in the PLY file has label 2.000000, False otherwise.

    Note: This function relies on the 'plyfile' module for reading PLY files.
    Ensure that 'plyfile' is installed and imported.
    """
    ply_data = PlyData.read(file_path)
    
    # Check if 'label' field exists in the vertex data
    if 'label' in ply_data['vertex'].data.dtype.names:
        labels = ply_data['vertex']['label']
        return any(label == 2.000000 for label in labels)
    else:
        return False

def add_label_streetlight_to_ply(input_directory, output_directory):
    """
    Add a new scalar field 'label_streetlight' to each PLY file in the input directory and save 
    the modified files in the output directory. For each point in the point cloud, 'label_streetlight' 
    is set to 1 if 'label' is not 2, and set to 2 if 'label' is 2.

    Parameters:
    input_directory (str): The path to the directory containing the original PLY files.
    output_directory (str): The path to the directory where modified files will be saved.

    Returns:
    None
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('.ply'):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename)
            ply_data = PlyData.read(input_file_path)

            if 'label' in ply_data['vertex'].data.dtype.names:
                labels = ply_data['vertex']['label']
                label_streetlight = np.array([1 if label == 2 else 0 for label in labels], dtype=np.int32)
                new_dtype = ply_data['vertex'].data.dtype.descr + [('label_streetlight', 'i4')]
                new_data = np.empty(ply_data['vertex'].count, dtype=new_dtype)
                for prop in ply_data['vertex'].data.dtype.names:
                    new_data[prop] = ply_data['vertex'][prop]
                new_data['label_streetlight'] = label_streetlight
                new_vertex = PlyElement.describe(new_data, 'vertex')
                new_ply_data = PlyData([new_vertex], text=ply_data.text)
                new_ply_data.write(output_file_path)
                print(f"Updated {input_file_path} and saved as {output_file_path}")

if __name__ == "__main__":
    poles_path = "data/pole_data/poles_ply"
    poles_path_modified = "data/pole_data/poles_modified_ply"
    add_label_streetlight_to_ply(poles_path, poles_path_modified)
