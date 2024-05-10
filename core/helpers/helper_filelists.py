import pathlib
import re
import os
import laspy
from tqdm import tqdm
import numpy as np


def get_tilecode_from_filename(filename, prefix):
    """Extract the tile code from a file name."""
    return filename.split(prefix)[1]


def get_tilecodes_from_folder_nested(folder, prefix='', extension=''):
    """Get a set of unique tilecodes for the LAS files in a given folder."""
    files = pathlib.Path(folder).rglob(f'{prefix}*{extension}')
    tilecodes = set([get_tilecode_from_filename(file.name, prefix) for file in files])
    return tilecodes


def get_tilecodes_from_folder(folder, prefix='', extension=''):
    """Get a set of unique tilecodes for the LAS files in a given folder."""
    files = pathlib.Path(folder).glob(f'{prefix}*{extension}')
    tilecodes = set([get_tilecode_from_filename(file.name, prefix) for file in files])
    return tilecodes


def merge_cloud_pred(cloud_file, pred_file, out_file, inference_on_labels, label_dict=None):
    """Merge predicted labels into a point cloud LAS file."""
    cloud = laspy.read(cloud_file)
    pred = laspy.read(pred_file)

    # TODO Add option to not overwrite labels that are already in cloud file
    labels = cloud.label # TODO This assumes that inference files are already labeled, is this always the case?
    if inference_on_labels:
        mask = np.zeros((len(labels),), dtype=bool)
        for inference_label in inference_on_labels:
            mask = mask | (labels == inference_label)
    else:
        # Numpy array with all True
        mask = np.ones((len(labels),), dtype=bool)

    if len(pred.label) != len(labels[mask]):
        print('Dimension mismatch between cloud and prediction '
              + f'for tile {cloud_file} and {pred_file}.')
        return

    if 'label' not in cloud.point_format.extra_dimension_names:
        cloud.add_extra_dim(laspy.ExtraBytesParams(
                            name="label", type="uint8", description="Labels"))

    cloud.label[mask] = pred.label.astype('uint8')
    if label_dict is not None:
        for key, value in label_dict.items():
            cloud.label[mask][cloud.label[mask] == key] = value

    if 'probability' in pred.point_format.extra_dimension_names:
        if 'probability' not in cloud.point_format.extra_dimension_names:
            cloud.add_extra_dim(laspy.ExtraBytesParams(
                                name="probability", type="float32",
                                description="Probabilities"))
        cloud.probability[mask] = pred.probability

    cloud.write(out_file)


def merge_cloud_pred_folder(cloud_folder, pred_folder, cloud_prefix, 
                            pred_prefix, inference_on_labels, out_folder='',
                            out_prefix='merged', label_dict=None,
                            resume=False, hide_progress=False):
    """
    Merge the labels of all predicted tiles in a folder into the corresponding
    point clouds and save the result.
    Parameters
    ----------
    cloud_folder : str
        Folder containing the unlabelled .laz files.
    pred_folder : str
        Folder containing corresponding .laz files with predicted labels.
    out_folder : str (default: '')
        Folder in which to save the merged clouds.
    cloud_prefix : str (default: 'filtered')
        Prefix of unlabelled .laz files.
    pred_prefix : str (default: 'pred')
        Prefix of predicted .laz files.
    out_prefix : str (default: 'merged')
        Prefix of output files.
    label_dict : dict (optional)
        Mapping from predicted labels to saved labels.
    resume : bool (default: False)
        Skip merge when output file already exists.
    hide_progress : bool (default: False)
        Whether to hide the progress bar.
    """
    cloud_codes = get_tilecodes_from_folder(cloud_folder, prefix=cloud_prefix)
    pred_codes = get_tilecodes_from_folder(pred_folder, prefix=pred_prefix)
    in_codes = cloud_codes.intersection(pred_codes)
    if resume:
        done_codes = get_tilecodes_from_folder(out_folder, prefix=out_prefix)
        todo_codes = {c for c in in_codes if c not in done_codes}
    else:
        todo_codes = in_codes
    files_tqdm = tqdm(todo_codes, unit="file", disable=hide_progress,
                      smoothing=0)
    print(f'{len(todo_codes)} files found.')

    for tilecode in files_tqdm:
        files_tqdm.set_postfix_str(tilecode)
        print(f'Processing tile {tilecode}...')
        cloud_file = os.path.join(
                        cloud_folder, cloud_prefix + tilecode)
        pred_file = os.path.join(
                        pred_folder, pred_prefix + tilecode)
        out_file = os.path.join(
                        out_folder, out_prefix + tilecode)
        merge_cloud_pred(cloud_file, pred_file, out_file, inference_on_labels, label_dict)