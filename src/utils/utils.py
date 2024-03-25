import os
import laspy

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
    
directory = "data/bb_extracted_data_labelled"
average = average_points_in_laz_files(directory)
print(f"Average number of points per point cloud: {average}")