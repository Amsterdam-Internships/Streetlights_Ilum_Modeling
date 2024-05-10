"""RandLA-Net configuration for Amsterdam point clouds."""


class ConfigAmsterdam3D:
    name = 'Amsterdam3D'

    # Dataset configuration.
    labels = {0: 'Unlabelled',
              1: 'Ground',
              2: 'Building',
              3: 'Tree',
              4: 'Street light',
              5: 'Traffic sign',
              6: 'Traffic light',
              7: 'Car',
              99: 'Noise'}
    ignore_labels = [0, 6, 99]
    num_classes = len(labels) - len(ignore_labels) # Number of valid classes
    label_to_idx = {7:0}
    idx_to_label = {0: 7}  # Original order
    inference_on_labels = []

    # Use an offset to subtract from the raw coordinates. Otherwise, use 0.
    x_offset = 129500
    y_offset = 476500

    # Number of points per class. Order of main.py
    class_weights = [14806719, 183930533, 100581666, 17691733, 584752, 182871]

    # Approximately max size Bytes to load for 16GB GPU
    max_size_bytes = 12000000000

    # RandLA-Net configuration.
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 3  # batch_size during training
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

    train_sum_dir = 'tf_train_log'
    saving = True
    saving_path = None

    # Validation files, used in training step
    val_files = ['processed_2645_9582', 'processed_2641_9590',
                 'processed_2641_9583', 'processed_2615_9617',
                 'processed_2633_9595', 'processed_2615_9610',
                 'processed_2628_9615', 'processed_2628_9604',
                 'processed_2640_9583', 'processed_2624_9590']