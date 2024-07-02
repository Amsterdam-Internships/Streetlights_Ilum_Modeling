import bpy
import os
import laspy 
import re 
import pandas as pd 
import numpy as np
import argparse

def import_laz(filepath):
    # Import the LAS file
    bpy.ops.import_scene.las_data(filepath=filepath)

def select_object_by_name(obj_name):
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    # Select the specified object
    obj = bpy.data.objects.get(obj_name)
    if obj:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    else:
        print(f"Object named {obj_name} not found.")
        return None
    return obj

def update_geo_node_tree(node_tree):
    out_node = node_tree.nodes["Group Output"]
    in_node = node_tree.nodes["Group Input"]

    node_x_location = 0
    node_location_step_x = 300

    mesh_node = node_tree.nodes.new("GeometryNodeMeshToPoints")
    mesh_node.inputs[3].default_value = 0.02
    mesh_node.location.x = node_x_location
    node_x_location += node_location_step_x

    set_atr_node = node_tree.nodes.new("GeometryNodeSetMaterial")
    set_atr_node.inputs[2].default_value = bpy.data.materials["NewMaterial"]
    set_atr_node.location.x = node_x_location
    node_x_location += node_location_step_x

    out_node.location.x = node_x_location

    from_node = in_node
    to_node = mesh_node
    node_tree.links.new(from_node.outputs["Geometry"], to_node.inputs["Mesh"])

    from_node = mesh_node
    to_node = set_atr_node
    node_tree.links.new(from_node.outputs["Points"], to_node.inputs["Geometry"])

    from_node = set_atr_node
    to_node = out_node
    node_tree.links.new(from_node.outputs["Geometry"], to_node.inputs["Geometry"])
    pass

def update_shading_material(material):
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    shader_node = nodes['Principled BSDF']

    attribute_node = nodes.new(type='ShaderNodeAttribute')
    attribute_node.attribute_name = 'Col'  # Assuming you want to use 'Col' vertex colors
    attribute_node.location = (-600, 0)

    links.new(attribute_node.outputs['Color'], shader_node.inputs['Base Color'])
    
def add_modifiers(obj):
    if obj is None:
        print("No object is active or found.")
        return None
    
    # Ensure the object is the active object
    bpy.context.view_layer.objects.active = obj

    material = bpy.data.materials.new(name="NewMaterial")
    material.use_nodes = True
    obj.data.materials.append(material)
    
    update_shading_material(material)

    # Use the specific Blender operator to add a Geometry Nodes modifier
    bpy.ops.node.new_geometry_nodes_modifier()
    
    node_tree = bpy.data.node_groups["Geometry Nodes"]

    update_geo_node_tree(node_tree)

def save_blender_file(directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    full_path = os.path.join(directory, filename)
    bpy.ops.wm.save_as_mainfile(filepath=full_path)

    
def add_lights(df):
    """
    Adds point lights and a sun light to the current scene based on the provided DataFrame.

    :param df: DataFrame containing 'Grond X' and 'Grond Y' columns for point light positions.
    """
    for index, row in df.iterrows():
        # Assuming each row could have multiple clusters, and their indices might not be sequential
        clusters = [(row[f'Raw_Cluster_{i}_X'], row[f'Raw_Cluster_{i}_Y'], row[f'Raw_Cluster_{i}_Z'])  # Default Z to 6 if not present
                    for i in range(1, 17)  # Assuming up to 16 clusters; adjust as needed
                    if f'Raw_Cluster_{i}_X' in row and pd.notna(row[f'Raw_Cluster_{i}_X'])]  # Check if X coordinate exists and is not NaN
        for idx, (x, y, z) in enumerate(clusters):
            # Create a new point light datablock
            light_data = bpy.data.lights.new(name=f"PointLight_{index}_{idx}", type='POINT')
            light_data.energy = 1000  # Set the point light power (energy) to 1000 watts

            # Create a new object with the point light data
            light_object = bpy.data.objects.new(name=f"PointLight_{index}_{idx}", object_data=light_data)
            light_object.location = (x, y, z)

            # Link point light object to the current collection
            bpy.context.collection.objects.link(light_object)

    # Add a sun light
    sun_data = bpy.data.lights.new(name="SunLight", type='SUN')
    sun_data.energy = 1.5  # Strength of the sun light

    hue = 0.648
    saturation = 0.622
    value = 1.0

    # Blender uses RGB, so convert HSV to RGB
    import colorsys
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    sun_data.color = rgb  # Set the color of the sun light

    sun_object = bpy.data.objects.new(name="SunLight", object_data=sun_data)
    sun_object.location = (0, 0, 50)  # Sun light coordinates
    bpy.context.collection.objects.link(sun_object)

def delete_objects(obj_names):
    # Delete specified objects from the scene
    for obj_name in obj_names:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)

def setup_and_save(filepath, save_directory, save_filename, cluster_filepath):
    match = re.search(r'(\d+)_(\d+).laz', filepath)
    if match:
        x_tile, y_tile = int(match.group(1)) * 50, int(match.group(2)) * 50
    min_coords = (x_tile, y_tile, 0)
    max_coords = (x_tile + 50, y_tile + 50, 0)
    center = tuple((np.array(min_coords) + np.array(max_coords)) / 2)

    print("Laz file begins importation")
    import_laz(filepath)
    print("Laz file imported")

    df = pd.read_csv(cluster_filepath)
    filtered_df = adjust_cluster_coordinates(df[df.apply(lambda row: filter_clusters(row, x_tile, y_tile), axis=1)], center)

    print(filtered_df)
    
    delete_objects(['Cube', 'Light'])
    print("Objects deleted")
    # Select the imported object
    obj = select_object_by_name("LAS Data")
    if obj is None:
        return
    print("Object selected")

    # Add the Geometry Nodes modifier and retrieve the node group
    add_modifiers(obj)
    print("Modifiers added")

    add_lights(filtered_df)
    print("Lights added")
    
    bpy.context.scene.render.engine = 'CYCLES'

    print("Cycles Engine set and Rendering")
    # Save the Blender file
    save_blender_file(save_directory, save_filename)

def filter_clusters(row, x_tile, y_tile):
    # Checking each cluster X and Y in the row to see if it falls within the tile
    for key in row.keys():
        if 'Raw_Cluster' in key and '_X' in key:
            index = key.split('_X')[0]  # Get the cluster index (e.g., 'Cluster_1')
            x = row[index + '_X']
            y = row[index + '_Y']
            # Check if the cluster is within the bounds of the tile
            if x_tile <= x < x_tile + 50 and y_tile <= y < y_tile + 50:
                return True
    return False

def adjust_cluster_coordinates(df, center):
    """
    Adjusts the cluster coordinates in the DataFrame by subtracting the given center coordinates.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the cluster coordinates.
    - center (tuple): A tuple of (x, y, z) representing the center coordinates to subtract.
    
    Returns:
    - pd.DataFrame: The adjusted DataFrame.
    """
    x_center, y_center, z_center = center

    # Iterate over each column in the DataFrame
    for col in df.columns:
        if 'Cluster' in col or 'Raw_Cluster' in col:
            if '_X' in col:
                df[col] = (df[col]) - x_center
            elif '_Y' in col:
                df[col] = (df[col]) - y_center
            elif '_Z' in col:
                df[col] = (df[col]) - z_center

    return df

def process_laz_files(laz_folder_path, save_directory, clusters_file_path):
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Iterate over all `.laz` files in the specified directory
    for laz_file in os.listdir(laz_folder_path):
        if laz_file.endswith('.laz'):
            laz_file_path = os.path.join(laz_folder_path, laz_file)
            save_filename = f"{os.path.splitext(laz_file)[0]}.blend"
            print(f"Processing {laz_file_path}, saving as {save_filename}")
            setup_and_save(laz_file_path, save_directory, save_filename, clusters_file_path)

def main():
    parser = argparse.ArgumentParser(description="Process LAZ files and save results.")
    parser.add_argument('--laz_folder_path', type=str, required=True, help='Directory containing LAZ files')
    parser.add_argument('--save_directory', type=str, required=True, help='Directory to save processed files')
    parser.add_argument('--clusters_file_path', type=str, required=True, help='Path to clusters CSV file')

    args = parser.parse_args()

    process_laz_files(args.laz_folder_path, args.save_directory, args.clusters_file_path)

if __name__ == "__main__":
    main()

