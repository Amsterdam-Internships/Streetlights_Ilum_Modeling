from os.path import join
import tensorflow as tf
# tf.set_random_seed(42)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
np.random.seed(42)

import time
import pickle
import argparse
import glob
import os
import sys
from sklearn.neighbors import KDTree
import laspy
import pandas as pd
import os 
from core.models.RandLANet import Network
from core.tester import ModelTester
from core.helpers.helper_tool import DataProcessing as DP

class Amsterdam3D:
    def __init__(self, in_folder, in_files):
        self.name = cfg.name
        self.label_to_names = cfg.labels
        self.label_values = np.sort(
            [k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.mode = args.mode

        self.val_files = cfg.val_files
        self.all_files = in_files

        # Initiate containers
        self.test_proj = []
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_features = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(in_folder)
        self.features_used = ''

        if args.use_rgb:
            self.features_used += 'RGB'
        if args.use_intensity:
            self.features_used += 'I'
        if args.use_normals:
            self.features_used += 'N'
        
    def get_features(self, data):
            sub_features = []
            if args.use_rgb:
                try:
                    sub_features.extend([data['red'], data['green'], data['blue']])
                except:
                    print('An exception occurred, no RGB values. Please run prepare.py first with the --use_rgb flag.')
                    raise
            if args.use_intensity:
                try:
                    sub_features.extend([data['intensity']])
                except:
                    print('An exception occurred, no intensity values. Please run prepare.py first with the --use_intensity flag.')
                    raise
            if args.use_normals:
                try:
                    sub_features.extend([data['normal_x'], data['normal_y'], data['normal_z']])
                except:
                    print('An exception occurred, no normal values. Please run prepare.py first with the --use_normals flag.')
                    raise
            if sub_features:
                return np.vstack(sub_features).T
            else:
                return None

    def calculate_normals(points_xyz):
        object_pcd = o3d.geometry.PointCloud()
        points = np.stack((points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]),
                          axis=-1)
        object_pcd.points = o3d.utility.Vector3dVector(points)
        object_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                                                  max_nn=30))

        normals = np.matrix.round(np.array(object_pcd.normals), 2)

        return normals

    def get_raw_features(self, data):
        # TODO Copy from prepare.py
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
            try:
                normals = np.vstack((data.normal_x, data.normal_y, data.normal_z)).T
            except:
                print('No normals found in the point cloud. Generating them now...')
                points_xyz = np.vstack((data.x, data.y, data.z)).T
                normals = self.calculate_normals(points_xyz)
            normals = normals.astype(np.float16)
            features.append(normals)
            
        return np.hstack(features)

    def load_sub_sampled_clouds(self, in_folder):
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.mode == 'test':
                cloud_split = 'test'
            elif cloud_name in self.val_files:
                cloud_split = 'validation'
                print("ok :", cloud_name)
            else:
                cloud_split = 'training'

            # Check if the mode is set to 'test' and no data preparation is requested.
            if self.mode == 'test' and args.no_prepare:
                # Read the .las or .laz file containing the point cloud data.
                data = laspy.read(file_path)
                number_of_points = len(data.x)
                # Load points from the file and apply offset adjustments, then convert to a floating-point format.
                # This structure, xyz, contains the coordinates of each point in the point cloud.
                xyz = (np.vstack((data.x - cfg.x_offset, data.y - cfg.y_offset, data.z)).T.astype(np.float32))

                # Extract raw features from the point cloud data, which could include intensity, color, or other attributes.
                features = self.get_raw_features(data)

                # Load labels from the point cloud data, which typically represent the classification of each point.
                
                if cfg.inference_on_labels:
                    labels = data.label
                    # Initialize a boolean mask for filtering points based on specified labels for inference.
                    mask = np.zeros((len(labels),), dtype=bool)
                    # Update the mask to include points whose labels are in the specified inference labels.
                    for inference_label in cfg.inference_on_labels:
                        mask = mask | (labels == inference_label)
                else:
                    # If no specific labels are targeted for inference, use all points.
                    mask = np.ones((number_of_points,), dtype=bool)

                # Subsample the point cloud based on the mask and configuration grid size,
                # reducing the number of points to process and focusing on areas of interest.
                sub_xyz, sub_features = DP.grid_sub_sampling(xyz[mask], features[mask], grid_size=cfg.sub_grid_size)

                # Create a KDTree for the subsampled points, which enables efficient spatial queries.
                search_tree = KDTree(sub_xyz)

                # Perform a query on the KDTree to get the nearest subsampled point index for each point in the masked set.
                # This is used to relate each original point to its corresponding point in the subsampled set.
                proj_idx = np.squeeze(search_tree.query(xyz[mask], return_distance=False))
                proj_idx = proj_idx.astype(np.int32)

                # Collect the projection indices for further processing or evaluation.
                self.test_proj += [proj_idx]

            else:
                npz_file = join(in_folder, '{:s}.npz'.format(cloud_name))

                data = np.load(npz_file)

                sub_features = self.get_features(data)

                # Name of the input files
                kd_tree_file = join(
                    in_folder, '{:s}_KDTree.pkl'.format(cloud_name))
                # Read pkl with search tree
                with open(kd_tree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            if self.mode == 'train':
                sub_labels = data['label'].squeeze()
                self.input_labels[cloud_split] += [sub_labels]

            if self.mode == 'evaluate':
                sub_labels = data['label'].squeeze()
                self.input_labels[cloud_split] += [sub_labels]

            # Test projection
            if self.mode == 'test' and args.no_prepare == False:
                # Validation projection and labels
                proj_file = join(in_folder, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]

            self.input_trees[cloud_split] += [search_tree]
            self.input_features[cloud_split] += [sub_features]
            if sub_features is not None:
                size = sub_features.shape[0] * 4 * 7 * 1e-6
                print(f'{file_path} {size:.1f} MB loaded in {time.time() - t0:.1f}s')
            else :
                size = None
                print(f'{file_path} {size} MB loaded in {time.time() - t0:.1f}s')

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        elif split == 'test':
            num_per_epoch = cfg.val_steps * cfg.test_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []

        # Random initialize
        for i, tree in enumerate(self.input_trees[split]):
            self.possibility[split] +=\
                                [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] +=\
                                [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud
                # as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(
                        self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10,
                                         size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less
                # than the predefined num_points
                if len(points) < cfg.num_points:
                    # Query all points within the cloud
                    queried_idx = (self.input_trees[split][cloud_idx]
                                   .query(pick_point, k=len(points))[1][0])
                else:
                    # Query the predefined number of points
                    queried_idx = (self.input_trees[split][cloud_idx]
                                   .query(pick_point, k=cfg.num_points)[1][0])

                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point

                # Check if self.input_features[split][cloud_idx] is None and handle accordingly
                if self.input_features[split][cloud_idx] is not None:
                    queried_pc_colors = self.input_features[split][cloud_idx][queried_idx]
                else:
                    # If there are no additional features, initialize queried_pc_colors as an empty array
                    queried_pc_colors = np.zeros((queried_pc_xyz.shape[0], 0))
                if split == 'test':
                    queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
                else:
                    queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]

                # Update the possibility of the selected points
                dists = np.sum(np.square(
                                    (points[queried_idx] - pick_point)
                                    .astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] =\
                    float(np.min(self.possibility[split][cloud_idx]))

                # up_sampled with replacement
                if len(points) < cfg.num_points:
                    (queried_pc_xyz, queried_pc_colors,
                     queried_idx, queried_pc_labels) =\
                        DP.data_aug(queried_pc_xyz, queried_pc_colors,
                                    queried_pc_labels, queried_idx,
                                    cfg.num_points)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)

        num_features = (3 * args.use_rgb) + (3 * args.use_normals) + args.use_intensity
        gen_shapes = ([None, 3], [None, num_features], [None], [None], [None])

        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels,
                   batch_pc_idx, batch_cloud_idx):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search,
                                           [batch_xyz, batch_xyz, cfg.k_n],
                                           tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1]
                                       // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1]
                                       // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(
                        DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = (input_points + input_neighbors
                          + input_pools + input_up_samples)
            input_list += [batch_features, batch_labels,
                           batch_pc_idx, batch_cloud_idx]

            return input_list

        return tf_map

    def init_input_pipeline_train(self):
        print('Initiating input pipelines for train and validation')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label]
                                  for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        self.train_data = tf.data.Dataset.from_generator(
                                    gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(
                                    gen_function_val, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(
                                        self.batch_train_data.output_types,
                                        self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)

    def init_input_pipeline_eval(self):
        print('Initiating input pipelines for train and validation')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label]
                                  for ign_label in self.ignored_labels]
        # gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, gen_types, gen_shapes = self.get_batch_gen('validation')
        # self.train_data = tf.data.Dataset.from_generator(
        #                             gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(
                                    gen_function_val, gen_types, gen_shapes)

        # self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        # self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        # self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(
                                        self.batch_val_data.output_types,
                                        self.batch_val_data.output_shapes)
        self.flat_inputs = iter.get_next()
        # self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)

    def init_input_pipeline_test(self):
            """
            Initializes the input pipeline for testing the model.

            This method sets up the TensorFlow data pipeline for the test dataset by creating a generator-based
            TensorFlow Dataset object. It batches the data, applies a mapping function for preprocessing,
            and prefetches batches to optimize performance during testing.

            The method leverages configuration settings to adjust batch sizes and uses specific labels
            to be ignored during testing as defined in the configuration.
            """

            # Log the initiation of the test data pipeline
            print('Initiating input pipelines for testing')

            # Filter out ignored labels based on configuration
            cfg.ignored_label_inds = [self.label_to_idx[ign_label]
                                    for ign_label in self.ignored_labels]

            # Generate the TensorFlow dataset for testing
            gen_function_test, gen_types, gen_shapes = self.get_batch_gen('test')
            self.test_data = tf.data.Dataset.from_generator(
                                        gen_function_test, gen_types, gen_shapes)

            # Batch the dataset according to the test batch size specified in the configuration
            self.batch_test_data = self.test_data.batch(cfg.test_batch_size)

            # Apply a mapping function to the data for any required preprocessing
            map_func = self.get_tf_mapping2()  # Assuming this function handles data normalization, augmentation etc.
            self.batch_test_data = self.batch_test_data.map(map_func=map_func)

            # Prefetch data to improve pipeline performance by overlapping the preprocessing of data with model execution
            self.batch_test_data = self.batch_test_data.prefetch(cfg.test_batch_size)

            # Create an iterator over the batched dataset
            iter = tf.data.Iterator.from_structure(
                                            self.batch_test_data.output_types,
                                            self.batch_test_data.output_shapes)
            self.flat_inputs = iter.get_next()  # Retrieve the next batch of data from the iterator

            # Create an operation to initialize the test data iterator
            # This operation will be run in the session to restart the iterator when needed
            self.test_init_op = iter.make_initializer(self.batch_test_data)



def train(in_folder, in_files):
    dataset = Amsterdam3D(in_folder, in_files)
    dataset.init_input_pipeline_train()

    model = Network(dataset, cfg, args.resume, args.resume_path)
    model.train(dataset)

def evaluate(in_folder, in_files):
    dataset = Amsterdam3D(in_folder, in_files)
    dataset.init_input_pipeline_train()

    model = Network(dataset, cfg)

    if args.snap_folder is not None:
        snap_steps = [int(f[:-5].split('-')[-1])
                      for f in os.listdir(args.snap_folder) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(args.snap_folder, 'snap-{:d}'.format(chosen_step))
    else:
        logs = np.sort([os.path.join('results', f)
                        for f in os.listdir('results') if f.startswith('Log')])
        chosen_folder = logs[-1]
        snap_path = join(chosen_folder, 'snapshots')
        snap_steps = [int(f[:-5].split('-')[-1])
                      for f in os.listdir(snap_path) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
    # print(chosen_snap)
    with tf.Session() as sess:
        # If `my_vars` is not specified, Saver will restore all variables in the graph
        saver = tf.train.Saver()

        # Restore the model from the chosen snapshot
        saver.restore(sess, chosen_snap)

        # Assign the session to your model if it's required
        model.sess = sess

        # Now you can evaluate your model
        model.evaluate(dataset)


def test(in_folder, in_files):
    dataset = Amsterdam3D(in_folder, in_files)
    dataset.init_input_pipeline_test()
    cfg.saving = False
    model = Network(dataset, cfg)
    if args.snap_folder is not None:
        snap_steps = [int(f[:-5].split('-')[-1])
                      for f in os.listdir(args.snap_folder) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(args.snap_folder, 'snap-{:d}'.format(chosen_step))
        print(chosen_snap)
    else:
        logs = np.sort([os.path.join('results', f)
                        for f in os.listdir('results') if f.startswith('Log')])
        chosen_folder = logs[-1]
        snap_path = join(chosen_folder, 'snapshots')
        snap_steps = [int(f[:-5].split('-')[-1])
                      for f in os.listdir(snap_path) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))

    tester = ModelTester(model, dataset, restore_snap=chosen_snap)
    tester.test(model, dataset, args.out_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RandLA-Net implementation.')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--in_folder', metavar='path', action='store',
                        type=str, required=True)
    parser.add_argument('--out_folder', metavar='path', action='store',
                        type=str)
    parser.add_argument('--snap_folder', metavar='path', action='store',
                        type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_path', metavar='path', action='store',
                        type=str)
    parser.add_argument('--use_rgb', action='store_true')
    parser.add_argument('--use_intensity', action='store_true')
    parser.add_argument('--use_normals', action='store_true')
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--no_prepare', action='store_true')
    args = parser.parse_args()

    if args.config_file == 'AHNTrees':
        from configs.config_AHNTrees import ConfigAHNTrees as cfg
    elif args.config_file == 'Amsterdam3D':
        from configs.config_Amsterdam3D import ConfigAmsterdam3D as cfg
    elif args.config_file == 'Streetlights3D':
        from configs.config_Streetlights3D import ConfigStreetlights3D as cfg
    elif args.config_file == 'Streetlights3D_val':
        from configs.config_Streetlights3D_val import ConfigStreetlights3D_val as cfg

    if not args.use_rgb:
        print('RGB values are not used in this configuration.')
    if not args.use_intensity:
        print('Intensity values are not used in this configuration.')
    if not args.use_normals:
        print('Normal values are not used in this configuration.')

    # if not args.use_rgb and not args.use_intensity and not args.use_normals:
    #     # TODO: this is a temp fix
    #     print('At least select one feature: --use_rgb --use_intensity --use_normals')
    #     sys.exit(1)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if args.mode == 'train':
        print("mode train")
        in_folder = join(args.in_folder, f'input_{cfg.sub_grid_size:.3f}')
        if not os.path.isdir(in_folder):
            print(f'The input folder {in_folder} does not exist. Aborting...')
            sys.exit(1)

        in_files = glob.glob(join(in_folder, '*.npz'))
        train_run_id = time.strftime('%Y-%m-%d_%H-%M-%S')
        train_sum_dir = os.path.join('tf_train_logs', train_run_id)
        os.makedirs(train_sum_dir, exist_ok=True)
        cfg.train_sum_dir = train_sum_dir
        train(in_folder, in_files)

    if args.mode == 'evaluate':
        print("mode evaluate")
        in_folder = join(args.in_folder, f'input_{cfg.sub_grid_size:.3f}')
        if not os.path.isdir(in_folder):
            print(f'The input folder {in_folder} does not exist. Aborting...')
            sys.exit(1)
        in_files = glob.glob(join(in_folder, '*.npz'))
        evaluate(in_folder, in_files)

    elif args.mode == 'test':
        inf_path = 'predict_res.csv'
        in_folder = args.in_folder
        if not os.path.isdir(in_folder):
            print(f'The input folder {in_folder} does not exist. Aborting...')
            sys.exit(1)

        if args.out_folder:
            if not os.path.isdir(args.out_folder):
                os.makedirs(args.out_folder)
        else:
            print('Please provide the output folder. Aborting...')
            sys.exit(1)

        inference_files = 'inference_files.csv'
        if args.resume:
            if not os.path.isfile(inference_files):
                print(f'The input file {inference_files} does not exist. Aborting...')
                sys.exit(1)

            df = pd.read_csv(inference_files)
            df = df[df['processed'] == False]
        else:
            # Get all test files in a folder
            if args.no_prepare:
                all_files = glob.glob(join(in_folder, '*.laz'))
            else:
                all_files = glob.glob(join(in_folder, '*.npz'))
            # Create a dataframe
            df = pd.DataFrame({'in_files':all_files})
            df['processed'] = False
            df.to_csv(inference_files, index=False)
        # Split the dataframe every 10 rows
        grps = df.groupby(df.index // 5)
        for _, dfg in grps:
            in_files = dfg['in_files'].tolist()
            # Init and start the inference
            print(f'Starting inference for {len(in_files)} files...')
            test(in_folder, in_files)
            tf.reset_default_graph() # added
            # Save processed once for the resume option
            df.loc[dfg.index, 'processed'] = True
        df.to_csv(inf_path, index=False)

    else:
        print('Mode not implemented. Aborting...')
        sys.exit(1)