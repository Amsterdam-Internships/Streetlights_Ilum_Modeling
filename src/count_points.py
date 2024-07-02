import os
import laspy

def count_points_in_laz_file(file_path):
    """
    Count the number of points in a .laz file.
    """
    with laspy.open(file_path) as f:
        laz_data = f.read()
    return len(laz_data.points)

def count_total_points_in_folder(input_folder):
    """
    Count the total number of points in all .laz files within a given folder.
    """
    total_points = 0

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.laz'):
            file_path = os.path.join(input_folder, file_name)
            num_points = count_points_in_laz_file(file_path)
            total_points += num_points

    print(f'Total number of points in folder: {total_points}')
    return total_points

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

# Example usage
input_folder = 'data/train_val'
b = 'data/test'  # Replace with your input folder path
print(count_points_per_class(b))
print(count_points_per_class(input_folder))