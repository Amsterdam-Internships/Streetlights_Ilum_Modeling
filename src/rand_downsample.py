import os

import laspy
import numpy as np


def read_laz(file_path):
    """
    Read a .laz file and extract the point cloud data along with color, intensity, and normals.
    """
    with laspy.open(file_path) as f:
        laz = f.read()

    pts = np.vstack((laz.x, laz.y, laz.z)).transpose()
    clr = np.vstack((laz.red, laz.green, laz.blue)).transpose()
    intensity = laz.intensity
    norms = np.vstack((laz.normal_x, laz.normal_y, laz.normal_z)).transpose()
    seg = laz.label
    return pts, clr, intensity, norms, seg


def random_downsample(pts, clr=None, intensity=None, norms=None, seg=None, ratio=0.1):
    """
    Downsample a point cloud using random sampling.
    """
    num_pts = len(pts)
    num_sampled_pts = int(num_pts * ratio)

    indices = np.random.choice(num_pts, num_sampled_pts, replace=False)

    ds_pts = pts[indices]
    ds_clr = clr[indices] if clr is not None else None
    ds_intensity = intensity[indices] if intensity is not None else None
    ds_norms = norms[indices] if norms is not None else None
    ds_seg = seg[indices] if seg is not None else None

    return ds_pts, ds_clr, ds_intensity, ds_norms, ds_seg


def save_laz(pts, clr=None, intensity=None, norms=None, seg=None, output_path=None):
    """
    Save a point cloud to a .laz file.
    """
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)

    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]

    if clr is not None:
        las.red = clr[:, 0]
        las.green = clr[:, 1]
        las.blue = clr[:, 2]
    if intensity is not None:
        las.intensity = intensity
    if norms is not None:
        las.add_extra_dim(laspy.ExtraBytesParams(
            name="normal_x", type=np.float32, description="Normal X component"))
        las.add_extra_dim(laspy.ExtraBytesParams(
            name="normal_y", type=np.float32, description="Normal Y component"))
        las.add_extra_dim(laspy.ExtraBytesParams(
            name="normal_z", type=np.float32, description="Normal Z component"))
        las.normal_x = norms[:, 0]
        las.normal_y = norms[:, 1]
        las.normal_z = norms[:, 2]

    # Create and assign custom dimension for labels
    if seg is not None:
        las.add_extra_dim(laspy.ExtraBytesParams(
            name="label", type=np.uint8, description="Segment label"))
        las.label = seg

    las.write(output_path)


def process_folder(input_folder, output_folder, ratio=0.1):
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

            pts, clr, intensity, norms, seg = read_laz(input_path)
            ds_pts, ds_clr, ds_intensity, ds_norms, ds_seg = random_downsample(
                pts, clr=clr, intensity=intensity, norms=norms, seg=seg, ratio=ratio)
            save_laz(ds_pts, clr=ds_clr, intensity=ds_intensity,
                     norms=ds_norms, seg=ds_seg, output_path=output_path)

            print(f'Randomly downsampled and saved: {input_path} -> {output_path}')


# Example usage
sample_ratio = 0.33
input_folder = 'data/test_normals'  # Replace with your input folder path
output_folder = 'data/test_normals_random_ds_{}'.format(sample_ratio)
process_folder(input_folder, output_folder, sample_ratio=sample_ratio)
