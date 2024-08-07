import argparse
import os
import sys

import core.helpers.helper_filelists as helper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine predictions with " + "original clouds."
    )
    parser.add_argument("--cloud_folder", metavar="path", type=str, required=True)
    parser.add_argument("--pred_folder", metavar="path", type=str, required=True)
    parser.add_argument("--out_folder", metavar="path", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="")
    args = parser.parse_args()

    if not os.path.isdir(args.cloud_folder) or not os.path.isdir(args.pred_folder):
        print("The input folder(s) does not exist")
        sys.exit(1)

    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)

    if args.config_file == "AHNTrees":
        from configs.config_AHNTrees import ConfigAHNTrees as cfg

        cloud_prefix = "tree"
        pred_prefix = "predtree"
    elif args.config_file == "Amsterdam3D":
        from configs.config_Amsterdam3D import ConfigAmsterdam3D as cfg

        cloud_prefix = "filtered"
        pred_prefix = "predfiltered"
    elif args.config_file == "Streetlights3D":
        from configs.config_Streetlights3D import ConfigStreetlights3D as cfg

        cloud_prefix = "bb"
        pred_prefix = "predbb"
    helper.merge_cloud_pred_folder(
        args.cloud_folder,
        args.pred_folder,
        cloud_prefix,
        pred_prefix,
        cfg.inference_on_labels,
        out_folder=args.out_folder,
    )
