import os
import laspy
import numpy as np
import open3d as o3d
import argparse

def process_laz_files(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".laz"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Read the LAZ file
            las = laspy.read(input_path)
            points = np.vstack((las.x, las.y, las.z)).transpose()

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Estimate normals
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd.normalize_normals()

            # Extract normals
            normals = np.asarray(pcd.normals)

            # Add normals to the LAS file
            las.add_extra_dim(laspy.ExtraBytesParams(name="normal_x", type=np.float32, description="Normal X component"))
            las.add_extra_dim(laspy.ExtraBytesParams(name="normal_y", type=np.float32, description="Normal Y component"))
            las.add_extra_dim(laspy.ExtraBytesParams(name="normal_z", type=np.float32, description="Normal Z component"))

            # Assign normals to the LAS file
            las.normal_x = normals[:, 0]
            las.normal_y = normals[:, 1]
            las.normal_z = normals[:, 2]

            # Write the LAS file
            las.write(output_path)

    print("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description="Process LAZ files to add normals.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input LAZ files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output LAZ files with normals')
    
    args = parser.parse_args()
    
    process_laz_files(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
