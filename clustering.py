import argparse
import os

import laspy
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def read_and_filter_points(filepath, label_value=2):
    with laspy.open(filepath) as file:
        lidar = file.read()
        try:
            labels = lidar['label']  # Replace 'label' with the actual field name if different
        except KeyError:
            raise KeyError("The specified field 'label' does not exist in the LAS file.")
        # Applying filter
        mask = labels == label_value
        points = np.vstack((lidar.x, lidar.y, lidar.z)).transpose()[mask]

        # Extracting scale and offset
        scale = (file.header.x_scale, file.header.y_scale, file.header.z_scale)
        offset = (file.header.x_offset, file.header.y_offset, file.header.z_offset)

    return points, scale, offset


def apply_dbscan(points, eps=0.3, min_samples=5):
    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return clustering.labels_


def calculate_centroids(points, labels):
    # Calculate centroids of clusters
    centroids = []
    for label in np.unique(labels):
        if label != -1:  # Ignore noise points which are labeled as -1
            cluster_points = points[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
    return centroids


def denormalize_coordinates(centroids, scale, offset):
    denormalized_centroids = []
    for centroid in centroids:
        denorm_centroid = (centroid * np.array(scale)) + np.array(offset)
        denormalized_centroids.append(denorm_centroid)
    return denormalized_centroids


def process_directory(directory_path, label_value=2):
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.laz'):
            filepath = os.path.join(directory_path, filename)
            print(f"Processing file: {filename}")
            points, scale, offset = read_and_filter_points(filepath, label_value)
            labels = apply_dbscan(points)
            centroids = calculate_centroids(points, labels)
            denormalized_centroids = denormalize_coordinates(
                centroids, scale, offset
            )

            row_data = {'File': filename}
            for i, centroid in enumerate(denormalized_centroids):
                row_data[f'Raw_Cluster_{i+1}_X'] = centroid[0]
                row_data[f'Raw_Cluster_{i+1}_Y'] = centroid[1]
                row_data[f'Raw_Cluster_{i+1}_Z'] = centroid[2]
            data.append(row_data)

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(
        description='Process LiDAR data and extract clusters.'
    )
    parser.add_argument(
        '--directory', type=str, required=True,
        help='Path to the directory containing LiDAR files'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output CSV file to save clusters data'
    )
    parser.add_argument(
        '--label', type=int, default=2,
        help='Label value to filter points (default: 2)'
    )
    args = parser.parse_args()

    # Process the directory and save the results
    result_df = process_directory(args.directory, label_value=args.label)
    result_df.to_csv(args.output, index=False)
    print(f'Results saved to {args.output}')


if __name__ == "__main__":
    main()
