import os
import laspy
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def read_laz_file(file_path):
    """
    Read a .laz file and extract the point cloud data along with color, intensity, and normals.
    """
    with laspy.open(file_path) as f:
        laz_data = f.read()
        
    points = np.vstack((laz_data.x, laz_data.y, laz_data.z)).transpose()
    color = np.vstack((laz_data.red, laz_data.green, laz_data.blue)).transpose() 
    intensity = laz_data.intensity
    normals = np.vstack((laz_data.normal_x, laz_data.normal_y, laz_data.normal_z)).transpose()
    segment = laz_data.label
    return points, color, intensity, normals, segment

def random_downsample_point_cloud(points, color=None, intensity=None, normals=None, segment=None, sample_ratio=0.1):
    """
    Downsample a point cloud using random sampling.
    """
    num_points = len(points)
    num_sampled_points = int(num_points * sample_ratio)
    
    indices = np.random.choice(num_points, num_sampled_points, replace=False)
    
    downsampled_points = points[indices]
    downsampled_colors = color[indices] if color is not None else None
    downsampled_intensity = intensity[indices] if intensity is not None else None
    downsampled_normals = normals[indices] if normals is not None else None
    downsampled_segment = segment[indices] if segment is not None else None
    
    return downsampled_points, downsampled_colors, downsampled_intensity, downsampled_normals, downsampled_segment

def save_laz_file(points, colors=None, intensity=None, normals=None, segments=None, output_path=None):
    """
    Save a point cloud to a .laz file.
    """
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    
    if colors is not None:
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]
    if intensity is not None:
        las.intensity = intensity
    if normals is not None:
        las.add_extra_dim(laspy.ExtraBytesParams(name="normal_x", type=np.float32, description="Normal X component"))
        las.add_extra_dim(laspy.ExtraBytesParams(name="normal_y", type=np.float32, description="Normal Y component"))
        las.add_extra_dim(laspy.ExtraBytesParams(name="normal_z", type=np.float32, description="Normal Z component"))
        las.normal_x = normals[:, 0]
        las.normal_y = normals[:, 1]
        las.normal_z = normals[:, 2]
    
    # Create and assign custom dimension for labels
    if segments is not None:
        las.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.uint8, description="Segment label"))
        las.label = segments
    
    las.write(output_path)

def process_folder(input_folder, output_folder, sample_ratio=0.1):
    """
    Iterate over all .laz files in the input folder, downsample each,
    and save to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.laz'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            
            points, color, intensity, normals, segment = read_laz_file(input_path)
            downsampled_points, downsampled_colors, downsampled_intensity, downsampled_normals, downsampled_segment = random_downsample_point_cloud(
                points, color=color, intensity=intensity, normals=normals, segment=segment, sample_ratio=sample_ratio)
            save_laz_file(downsampled_points, colors=downsampled_colors, intensity=downsampled_intensity, normals=downsampled_normals, segments=downsampled_segment, output_path=output_path)
            
            print(f'Randomly downsampled and saved: {input_path} -> {output_path}')

# Example usage
sample_ratio = 0.33
input_folder = 'data/test_normals'  # Replace with your input folder path
output_folder = 'data/test_normals_random_downsample_{}'.format(sample_ratio)  # Replace with your output folder path
process_folder(input_folder, output_folder, sample_ratio=sample_ratio)
