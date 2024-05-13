"""RandLA-Net configuration for Amsterdam point clouds."""


class ConfigStreetlights3D:
    name = 'Streetlights3D'

    # Dataset configuration.
    labels = {1: 'Unlabelled',
              2: 'Light source'}
    ignore_labels = {}
    num_classes = len(labels) - len(ignore_labels) # Number of valid classes
    idx_to_label = {0: 1, 1: 2}  # Original order
    label_to_idx = {1:0, 2:1}
    inference_on_labels = []

    # Use an offset to subtract from the raw coordinates. Otherwise, use 0.
    x_offset = 113000
    y_offset = 479000

    # Number of points per class. Order of main.py
    class_weights = [2260784, 293859]

    # Approximately max size Bytes to load for 16GB GPU
    max_size_bytes = 12000000000

    # RandLA-Net configuration.
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 3  # batch_size during training
    val_batch_size = 16  # batch_size during validation and test
    test_batch_size = 3  # batch_size during inference
    train_steps = 148   # Number of steps per epochs
    val_steps = 4  # Number of validation steps per epoch

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
    val_files = ['bb_100', 'bb_111', 'bb_119', 'bb_130', 'bb_144', 'bb_148', 'bb_149', 'bb_160', 'bb_166', 'bb_167', 'bb_183', 'bb_184', 'bb_186', 'bb_193', 'bb_196', 'bb_202', 'bb_221', 'bb_231', 'bb_249', 'bb_25', 'bb_254', 'bb_256', 'bb_26', 'bb_268', 'bb_273', 'bb_286', 'bb_290', 'bb_31', 'bb_313', 'bb_325', 'bb_343', 'bb_363', 'bb_367', 'bb_370', 'bb_389', 'bb_403', 'bb_407', 'bb_411', 'bb_474', 'bb_484', 'bb_495', 'bb_501', 'bb_504', 'bb_507', 'bb_509', 'bb_51', 'bb_542', 'bb_551', 'bb_553', 'bb_59', 'bb_65', 'bb_75', 'bb_82', 'bb_91', 'bb_98']
    #val_files for old data
    # val_files = ['bb_119', 'bb_132', 'bb_136', 'bb_147', 'bb_152', 'bb_159', 'bb_161', 'bb_166', 'bb_170', 'bb_176', 'bb_177', 'bb_186', 'bb_196', 'bb_209', 'bb_212', 'bb_220', 'bb_227', 'bb_229', 'bb_230', 'bb_231', 'bb_236', 'bb_239', 'bb_270', 'bb_291', 'bb_315', 'bb_4', 'bb_44', 'bb_51', 'bb_52', 'bb_6', 'bb_62', 'bb_65', 'bb_68', 'bb_74']