import laspy
import re
import pandas as pd
import os
import argparse


def extract_numbers_from_string(s):
    """
    Extracts two numbers from a string that follows a specific pattern and multiplies them by 50.

    The function uses a regular expression to search for a pattern
    'final_<number>_<number>.laz' in the given string. If the pattern is found,
    it extracts the two numbers, multiplies them each by 50, and returns them.

    Parameters:
    s (str): The string from which to extract the numbers.

    Returns:
    tuple of int: A tuple containing the two extracted numbers multiplied by 50.
                   Returns None if the pattern does not match.
    """
    match = re.search(r'final_(\d+)_(\d+).laz', s)
    if match:
        # Extracting the two numbers
        num1, num2 = match.groups()
        return int(num1) * 50, int(num2) * 50
    else:
        # Return a default value or raise an error if the format doesn't match
        return None


def filter_pole_coordinates(filename, base_x, base_y):
    """
    Filters pole coordinates from an Excel file based on specified
    base values for 'Grond X', 'Top X', 'Grond Y', and 'Top Y'.

    Parameters:
    filename (str): Path to the Excel file.
    base_x (float): The base value for filtering X coordinates.
    base_y (float): The base value for filtering Y coordinates.

    Returns:
    pandas.DataFrame: Filtered DataFrame with only the rows where
                      'Grond X', 'Top X', 'Grond Y', and 'Top Y'
                      fall within the specified range of base_x/base_y
                      to base_x/base_y + 50.
    """
    # Read the Excel file
    df = pd.read_csv(filename)

    # Assuming that the decimal separator is a comma, replace it with a dot and convert to float
    df['Grond X'] = df['Grond X'].apply(lambda x: float(str(x).replace(',', '.')))
    df['Top X'] = df['Top X'].apply(lambda x: float(str(x).replace(',', '.')))
    df['Grond Y'] = df['Grond Y'].apply(lambda x: float(str(x).replace(',', '.')))
    df['Top Y'] = df['Top Y'].apply(lambda x: float(str(x).replace(',', '.')))

    # Filter the data
    filtered_df = df[(df['Grond X'] >= base_x) & (df['Grond X'] <= base_x + 50) &
                     (df['Top X'] >= base_x) & (df['Top X'] <= base_x + 50) &
                     (df['Grond Y'] >= base_y) & (df['Grond Y'] <= base_y + 50) &
                     (df['Top Y'] >= base_y) & (df['Top Y'] <= base_y + 50)]

    return filtered_df[['Grond X', 'Top X', 'Grond Y', 'Top Y', 'Grond Z', 'Top Z']]


def create_bounding_box_laz(input_laz_file, base_filename, poles_df, box_size=5):
    """
    Processes a given LAZ file, creating individual bounding boxes around
    each point defined in the provided DataFrame. It then extracts points
    within each bounding box and writes them to new LAZ files.

    The function iterates over each row in the DataFrame, defines a
    bounding box around the point coordinates (expanded by a specified box size),
    filters the points from the original LAZ file within this bounding box,
    and writes these points to a new LAZ file.

    Parameters:
    input_laz_file (str): Path to the input LAZ file containing the
                          original point cloud data.
    base_filename (str): Base path and filename for the output LAZ files.
                         Each file will be appended with an index number.
    poles_df (pandas.DataFrame): DataFrame containing the coordinates
                                ('Grond X', 'Top X', 'Grond Y', 'Top Y', 'Grond Z',
                                'Top Z') for each point.
    box_size (float, optional): Size of the bounding box to be created around each
                                point. Defaults to 5.

    Each output file is named using the base_filename followed by an underscore and
    the index of the row from the DataFrame.
    """
    # Read the original LAZ file
    with laspy.open(input_laz_file) as file:
        las = file.read()

        for index, row in poles_df.iterrows():
            # Define the bounding box for each point
            bounding_box = {
                'min_x': row['Grond X'] - box_size,
                'max_x': row['Top X'] + box_size,
                'min_y': row['Grond Y'] - box_size,
                'max_y': row['Top Y'] + box_size,
                'min_z': float(str(row['Grond Z']).replace(',', '.')) - box_size + 6,
                'max_z': float(str(row['Top Z']).replace(',', '.')) + box_size
            }

            # Apply the bounding box filter
            mask = (
                (las.x >= bounding_box['min_x']) & (las.x <= bounding_box['max_x']) &
                (las.y >= bounding_box['min_y']) & (las.y <= bounding_box['max_y']) &
                (las.z >= bounding_box['min_z']) & (las.z <= bounding_box['max_z'])
            )

            # Extract points within the bounding box
            filtered_points = las.points[mask]

            # Write the filtered points to a new LAZ file
            output_laz_file = f"{base_filename}_{index}.laz"

            # Create a new LasData object and assign the filtered points
            new_las = laspy.LasData(las.header)
            new_las.points = filtered_points

            new_las.write(output_laz_file)


def main(directory, output_base_filename, sheet):
    """
    Processes all .laz files in the specified directory, creating bounding boxes
    around points defined in an external Excel file.

    For each .laz file in the directory, this function extracts the base X and Y
    coordinates from the filename, filters relevant points from a provided Excel
    file, and then creates a new .laz file in the output directory with points
    within the bounding box.

    Parameters:
    directory (str): Directory containing the original .laz files to process.
    output_base_filename (str): Base path and filename for the output .laz files.
    sheet (str): Path to the Excel sheet containing coordinates of interest.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".laz"):
            input_laz_file = os.path.join(directory, filename)
            base_x, base_y = extract_numbers_from_string(filename)
            filtered_data = filter_pole_coordinates(sheet, base_x, base_y)

            # Process each file
            create_bounding_box_laz(
                input_laz_file, output_base_filename, filtered_data
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bounding Box Extraction from LiDAR Data.')
    parser.add_argument('--in_folder', metavar='path', action='store',
                        type=str, default='data/laz_pc_data', required=True)
    parser.add_argument('--out_folder', metavar='path', action='store',
                        type=str, default='data/laz_bb')
    parser.add_argument('--coordinates_csv', metavar='path', action='store', type=str,
                        default='data/sheets/processed_sheets/clustered_amsterdam.csv')
    args = parser.parse_args()
    if not os.path.isdir(args.out_folder):
        os.mkdir(args.out_folder)
    main(args.in_folder, args.out_folder, args.coordinates_csv)
