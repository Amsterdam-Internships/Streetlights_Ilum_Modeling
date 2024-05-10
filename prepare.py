from sklearn.neighbors import KDTree
from os.path import join
import numpy as np
import os
import sys
import glob
import pickle
import argparse
import laspy
from tqdm import tqdm

from core.helpers.helper_ply import write_ply
from core.helpers.helper_tool import DataProcessing as DP
import core.helpers.helper_filelists as utils

def list_to_dict(features_content, sub_features):
    # TODO do this part in the grid_sub_sampling CPP code directly
    if args.use_rgb:
        features_content['red'] = sub_features[:, 0]
        features_content['green'] = sub_features[:, 1]
        features_content['blue'] = sub_features[:, 2]
    if args.use_intensity:
        start_index = 3 * args.use_rgb
        features_content['intensity'] = sub_features[:, start_index]
    if args.use_normals:
        start_index = 3 * args.use_rgb + args.use_intensity
        features_content['normal_x'] = sub_features[:, start_index]
        features_content['normal_y'] = sub_features[:, start_index+1]
        features_content['normal_z'] = sub_features[:, start_index+2]
    return features_content


def get_features(data):
    features = []
    if args.use_rgb:
        # Load RGB
        rgb = np.vstack((data.red, data.green, data.blue)).T
        # Check color depth since this somehow seems to differ
        max_c = np.max(rgb)
        if max_c <= 1.:
            pass
        elif max_c < 2**8:
            rgb = rgb / 255.
        elif max_c < 2**16:
            rgb = rgb / 65535.
        else:
            print('RGB more than 16 bits, not implemented. Aborting...')
            sys.exit(1)
        rgb = rgb.astype(np.float16)

        features.append(rgb)
    if args.use_intensity:
        # Load intensity
        intensity = ((data.intensity / 65535.).reshape((-1, 1)).astype(np.float16))

        features.append(intensity)
    if args.use_normals:
        # Load normals
        normals = np.vstack((data.normal_x, data.normal_y, data.normal_z)).T
        normals = normals.astype(np.float16)

        features.append(normals)
    return np.hstack(features)

def get_ply_content():
    ply_content = ['x', 'y', 'z']
    if args.use_rgb:
        ply_content.extend(['red', 'green', 'blue'])
    if args.use_intensity:
        ply_content.extend(['intensity'])
    if args.use_normals:
        ply_content.extend(['normal_x', 'normal_y', 'normal_z'])
    return ply_content

def run():
    in_files = glob.glob(join(args.in_folder, '*.laz'))

    if args.resume:
        if args.save_ply:
            save_extension = '.ply'
        else:
            save_extension = '.npz'

        done_tiles = utils.get_tilecodes_from_folder(
                                args.out_folder, extension=save_extension)
        print('Resuming previous job. The following files are already done:')
        print(done_tiles)
        files = [f for f in in_files
                 if utils.get_tilecode_from_filename(f) not in done_tiles]
    else:
        files = in_files

    # Create the sub folder
    out_folder = join(args.out_folder,
                      f'input_{cfg.sub_grid_size:.3f}')
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # Initialize
    size_of_files = 0

    files_tqdm = tqdm(files, unit='file', smoothing=0)
    for f in files_tqdm:
        base_filename = os.path.splitext(os.path.basename(f))[0]

        data = laspy.read(f)
        # Load points
        xyz = (np.vstack((data.x - cfg.x_offset, data.y - cfg.y_offset,
                         data.z)).T.astype(np.float32))

        features = get_features(data)

        if args.save_ply:
            ply_content = get_ply_content()

        if args.mode == 'train':
            labels = data.label
            if args.config_file == 'AHNTrees': # TODO use config file
                # Numpy array with all True
                mask = np.ones((len(labels),), dtype=bool)
            else:
                for ignore_label in cfg.ignore_labels:
                    mask = ((labels == ignore_label))
                    labels[mask] = 255

                # Move label indexes 
                for label_to_move, idx in cfg.label_to_idx.items():
                    labels[labels == label_to_move] = idx

                mask = (labels != 255)

            labels_prepared = labels[mask].astype(np.uint8).reshape((-1,))
            if args.save_ply:
                ply_content.extend(['label'])

            sub_xyz, sub_features, sub_labels =\
                DP.grid_sub_sampling(xyz[mask], features[mask],
                                     labels_prepared,
                                     grid_size=cfg.sub_grid_size)

            features_content = list_to_dict({'label': sub_labels}, sub_features)

            if args.save_ply:
                # NOTE: Save the <x,y,z> to view the result in CloudCompare.
                sub_ply_file = join(out_folder, base_filename + '.ply')
                write_ply(sub_ply_file, [sub_xyz, sub_features, sub_labels],
                          ply_content)
            else:
                sub_npz_file = join(out_folder, base_filename + '.npz')
                np.savez_compressed(sub_npz_file, **features_content)

        elif args.mode == 'test':
            labels = data.label
            if cfg.inference_on_labels:
                mask = np.zeros((len(labels),), dtype=bool)
                for inference_label in cfg.inference_on_labels:
                    mask = mask | (labels == inference_label)
            else:
                # Numpy array with all True
                mask = np.ones((len(labels),), dtype=bool)

            sub_xyz, sub_features = DP.grid_sub_sampling(
                                        xyz[mask], features[mask],
                                        grid_size=cfg.sub_grid_size)

            features_content = list_to_dict({}, sub_features)

            if args.save_ply:
                sub_ply_file = join(out_folder, base_filename + '.ply')
                write_ply(sub_ply_file, [sub_xyz, sub_features], ply_content)
            else:
                sub_npz_file = join(out_folder, base_filename + '.npz')
                np.savez_compressed(sub_npz_file, **features_content)

        # save sub_cloud and KDTree file
        search_tree = KDTree(sub_xyz)
        kd_tree_file = join(out_folder, base_filename + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

        proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        proj_save = join(out_folder, base_filename + '_proj.pkl')
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_idx, labels], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for RandLA-Net.')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--in_folder', metavar='path', type=str, required=True)
    parser.add_argument('--out_folder', metavar='path', type=str,
                        required=True)
    parser.add_argument('--resume', action='store_true')
    # save_ply -> Useful for debugging sub Point Cloud in CloudCompare.
    parser.add_argument('--save_ply', action='store_true')
    parser.add_argument('--use_rgb', action='store_true')
    parser.add_argument('--use_intensity', action='store_true')
    parser.add_argument('--use_normals', action='store_true')
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()

    if not os.path.isdir(args.in_folder):
        print('The input path does not exist')
        sys.exit(1)

    if not args.use_rgb:
        print('RGB values are not used in this configuration.')
    if not args.use_intensity:
        print('Intensity values are not used in this configuration.')
    if not args.use_normals:
        print('Normal values are not used in this configuration.')

    if not args.use_rgb and not args.use_intensity and not args.use_normals:
        print('At least select one feature: --use_rgb --use_intensity --use_normals ')
        sys.exit(1)

    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)

    if args.config_file == 'AHNTrees':
        from configs.config_AHNTrees import ConfigAHNTrees as cfg
    else:
        from configs.config_Amsterdam3D import ConfigAmsterdam3D as cfg

    run()
