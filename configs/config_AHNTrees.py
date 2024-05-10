"""RandLA-Net configuration for AHN point clouds."""


class ConfigAHNTrees:
    name = 'AHNTrees'

    # Dataset configuration.
    labels = {0: 'unlabelled',
              1: 'tree'}
    ignore_labels = {}
    num_classes = len(labels) - len(ignore_labels) # Number of valid classes
    idx_to_label = {0: 0, 1: 1}  # Original order
    inference_on_labels = [0]

    # Use an offset to subtract from the raw coordinates. Otherwise, use 0.
    x_offset = 100000
    y_offset = 400000

    # Number of points per class label.
    class_weights = [859795, 17151240]

    # Approximately max size Bytes to load for 16GB GPU
    max_size_bytes = 1000000000

    # RandLA-Net configuration.
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 2  # batch_size during training
    val_batch_size = 16  # batch_size during validation and test
    test_batch_size = 16  # batch_size during inference
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # Sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter

    # Training configuration.
    max_epoch = 200  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    # Validation files, used in training step
    val_files = ['tree_25DN2_13_rel_i_norm'] # TODO check if this file exists
