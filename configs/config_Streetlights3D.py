"""RandLA-Net configuration for Amsterdam point clouds."""


class ConfigStreetlights3D:
    name = 'Streetlights3D'

    # Dataset configuration.
    labels = {
        1: 'Unlabelled',
        2: 'Light source'
    }
    ignore_labels = {}
    num_classes = len(labels) - len(ignore_labels)  # Number of valid classes
    idx_to_label = {0: 1, 1: 2}  # Original order
    label_to_idx = {1: 0, 2: 1}
    inference_on_labels = []

    # Use an offset to subtract from the raw coordinates. Otherwise, use 0.
    x_offset = 113000
    y_offset = 479000

    # Number of points per class. Order of main.py
    # class_weights = [2260784, 293859]
    # class weight 0.028 train
    class_weights = [1478291, 222869]
    # class weights 0.05 train
    # class_weights = [744270, 124190]

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
    train_steps = 148  # Number of steps per epochs
    val_steps = 4  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # Sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter

    # Training configuration.
    max_epoch = 70  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'tf_train_log'
    saving = True
    saving_path = None

    # Validation files, used in training step
    # val_files = ['bb_100', 'bb_111', 'bb_119', 'bb_130', 'bb_144', 'bb_148', 'bb_149', 'bb_160', 'bb_166', 'bb_167', 'bb_183', 'bb_184', 'bb_186', 'bb_193', 'bb_196', 'bb_202', 'bb_221', 'bb_231', 'bb_249', 'bb_25', 'bb_254', 'bb_256', 'bb_26', 'bb_268', 'bb_273', 'bb_286', 'bb_290', 'bb_31', 'bb_313', 'bb_325', 'bb_343', 'bb_363', 'bb_367', 'bb_370', 'bb_389', 'bb_403', 'bb_407', 'bb_411', 'bb_474', 'bb_484', 'bb_495', 'bb_501', 'bb_504', 'bb_507', 'bb_509', 'bb_51', 'bb_542', 'bb_551', 'bb_553', 'bb_59', 'bb_65', 'bb_75', 'bb_82', 'bb_91', 'bb_98']

    # val_files for old data
    # val_files = ['bb_119', 'bb_132', 'bb_136', 'bb_147', 'bb_152', 'bb_159', 'bb_161', 'bb_166', 'bb_170', 'bb_176', 'bb_177', 'bb_186', 'bb_196', 'bb_209', 'bb_212', 'bb_220', 'bb_227', 'bb_229', 'bb_230', 'bb_231', 'bb_236', 'bb_239', 'bb_270', 'bb_291', 'bb_315', 'bb_4', 'bb_44', 'bb_51', 'bb_52', 'bb_6', 'bb_62', 'bb_65', 'bb_68', 'bb_74']

    # testing for new data
    val_files = [
        'bb_114', 'bb_122', 'bb_135', 'bb_15', 'bb_152', 'bb_16', 'bb_176', 'bb_204',
        'bb_211', 'bb_23', 'bb_239', 'bb_244', 'bb_264', 'bb_269', 'bb_285', 'bb_291',
        'bb_3', 'bb_319', 'bb_32', 'bb_320', 'bb_326', 'bb_328', 'bb_33', 'bb_334',
        'bb_35', 'bb_352', 'bb_366', 'bb_371', 'bb_391', 'bb_395', 'bb_400', 'bb_401',
        'bb_422', 'bb_43', 'bb_433', 'bb_443', 'bb_461', 'bb_467', 'bb_476', 'bb_479',
        'bb_482', 'bb_487', 'bb_515', 'bb_516', 'bb_521', 'bb_523', 'bb_547', 'bb_559',
        'bb_67', 'bb_69', 'bb_7', 'bb_77', 'bb_80', 'bb_88', 'bb_89', 'bb_99'
    ]

    # val_files = ['bb_106', 'bb_11', 'bb_13', 'bb_137', 'bb_14', 'bb_142', 'bb_143', 'bb_144', 'bb_153', 'bb_189', 'bb_198', 'bb_208', 'bb_224', 'bb_235', 'bb_241', 'bb_244', 'bb_27', 'bb_272', 'bb_278', 'bb_281', 'bb_289', 'bb_290', 'bb_296', 'bb_310', 'bb_43', 'bb_48', 'bb_49', 'bb_57', 'bb_71', 'bb_83', 'bb_90', 'bb_93', 'bb_96']
