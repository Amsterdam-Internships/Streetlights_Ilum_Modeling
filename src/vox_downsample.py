import os

import laspy
import numpy as np
import open3d as o3d
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
    normals = np.vstack(
        (laz_data.normal_x, laz_data.normal_y, laz_data.normal_z)
    ).transpose()
    segment = laz_data.label
    return points, color, intensity, normals, segment


def downsample_point_cloud(
    points, color=None, intensity=None, normals=None, segment=None, voxel_size=0.05
):
    """
    Downsample a point cloud using a voxel grid filter.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            color / 255.0
        )  # Normalize colors to [0, 1]
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    downsampled_points = np.asarray(downsampled_pcd.points)
    downsampled_colors = (
        (np.asarray(downsampled_pcd.colors) * 255.0).astype(np.uint16)
        if color is not None
        else None
    )
    downsampled_normals = (
        np.asarray(downsampled_pcd.normals) if normals is not None else None
    )

    if intensity is not None:
        tree = cKDTree(points)
        _, idx = tree.query(downsampled_points)
        downsampled_intensity = intensity[idx]
    else:
        downsampled_intensity = None

    if segment is not None:
        tree = cKDTree(points)
        _, idx = tree.query(downsampled_points)
        downsampled_segment = segment[idx]
    else:
        downsampled_segment = None

    return (
        downsampled_points,
        downsampled_colors,
        downsampled_intensity,
        downsampled_normals,
        downsampled_segment,
    )


def save_laz_file(
    points, colors=None, intensity=None, normals=None, segments=None, output_path=None
):
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
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="normal_x", type=np.float32, description="Normal X component"
            )
        )
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="normal_y", type=np.float32, description="Normal Y component"
            )
        )
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="normal_z", type=np.float32, description="Normal Z component"
            )
        )
        las.normal_x = normals[:, 0]
        las.normal_y = normals[:, 1]
        las.normal_z = normals[:, 2]

    if segments is not None:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="label", type=np.uint8, description="Segment label"
            )
        )
        las.label = segments

    las.write(output_path)


def process_folder(input_folder, output_folder, voxel_size=0.05):
    """
    Iterate over all .laz files in the input folder, downsample each,
    and save to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".laz"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            points, color, intensity, normals, segment = read_laz_file(input_path)
            (
                downsampled_points,
                downsampled_colors,
                downsampled_intensity,
                downsampled_normals,
                downsampled_segment,
            ) = downsample_point_cloud(
                points,
                color=color,
                intensity=intensity,
                normals=normals,
                segment=segment,
                voxel_size=voxel_size,
            )
            save_laz_file(
                downsampled_points,
                colors=downsampled_colors,
                intensity=downsampled_intensity,
                normals=downsampled_normals,
                segments=downsampled_segment,
                output_path=output_path,
            )

            print(f"Downsampled and saved: {input_path} -> {output_path}")


# Example usage
voxel_size = 0.028
input_folder = "data/test_normals"
output_folder = f"data/test_normals_downsample_{voxel_size}"
process_folder(input_folder, output_folder, voxel_size=voxel_size)
