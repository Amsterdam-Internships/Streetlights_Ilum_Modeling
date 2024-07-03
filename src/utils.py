import os
import shutil


def copy_non_listed_laz_files(input_folder, output_folder, exclude_list):
    """
    Copy all .laz files that are not in the exclude_list from input_folder to output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".laz"):
            base_name = os.path.splitext(file_name)[0]
            if base_name not in exclude_list:
                input_path = os.path.join(input_folder, file_name)
                output_path = os.path.join(output_folder, file_name)
                shutil.copy2(input_path, output_path)
                print(f"Copied: {input_path} -> {output_path}")


# Example usage
exclude_list = [
    "bb_100",
    "bb_111",
    "bb_119",
    "bb_130",
    "bb_144",
    "bb_148",
    "bb_149",
    "bb_160",
    "bb_166",
    "bb_167",
    "bb_183",
    "bb_184",
    "bb_186",
    "bb_193",
    "bb_196",
    "bb_202",
    "bb_221",
    "bb_231",
    "bb_249",
    "bb_25",
    "bb_254",
    "bb_256",
    "bb_26",
    "bb_268",
    "bb_273",
    "bb_286",
    "bb_290",
    "bb_31",
    "bb_313",
    "bb_325",
    "bb_343",
    "bb_363",
    "bb_367",
    "bb_370",
    "bb_389",
    "bb_403",
    "bb_407",
    "bb_411",
    "bb_474",
    "bb_484",
    "bb_495",
    "bb_501",
    "bb_504",
    "bb_507",
    "bb_509",
    "bb_51",
    "bb_542",
    "bb_551",
    "bb_553",
    "bb_59",
    "bb_65",
    "bb_75",
    "bb_82",
    "bb_91",
    "bb_98",
]

input_folder = "data/train_val"  # Replace with your input folder path
output_folder = "data/train"  # Replace with your output folder path

copy_non_listed_laz_files(input_folder, output_folder, exclude_list)
