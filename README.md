# Streetlight Illumination Modelling on 3D-Derived Lidar Point Cloud

This repository contains a Tensorflow implementation of [RandLA-Net](http://arxiv.org/abs/1911.11236) with small improvements to the [original implementation](https://github.com/QingyongHu/RandLA-Net). This repository only supports the 3D Point Cloud licensed to City of Amsterdam.

## Getting Started with Python and Blender API

### Installation of Blender Software
1. **Download Blender**: Visit [Blender's official website](https://www.blender.org/) and download Blender 4.0.

### Configuring Blender in System Path
2. **Set Blender as a PATH variable**:
   - Navigate to the Blender Foundation folder, typically located in "Program Files" or your chosen installation directory.
   - Find the `blender.exe` executable, usually found at a path similar to `C://...//Blender Foundation\Blender 4.0`.
   - Add this path to your system's PATH environment variable for easy access.

### Processing of Laz Point Cloud
3. **Configure the addon**

First, you'll need to install laspy[lazrs]. Open Blender, go to the Scripting workspace, and in the Interactive Console type:
```
import sys; sys.executable
```

This will tell you where the python interpreter is that Blender is using.  For example: '/usr/bin/python3.10'

Now, open a Terminal and install laspy by replacing the path to the interpreter to the one you just discovered above:

Linux:
```
/replace/with/path/to/python -m pip install laspy[lazrs]
```
Note: You may need to use sudo on Linux if your OS requires it.

Windows (replace the path with the correct version):
```
"C:\Program Files\Blender Foundation\Blender x.x\x.x\python\bin\python.exe" -m pip install laspy[lazrs]
```

Now, in Blender, navigate to Edit > Preferences, and click on Add-ons tab. Click "Install", and select this file. 
```
utils\addons_blender\__init__.py
```

Finally, activate the addon. You can now import laz/las files to Blender ! 


### Installing Pandas in Blender
4. **Integrate Pandas with Blender**:
   - Launch Blender and navigate to the Scripting menu.
   - Run the following script in the Python Console:
     ```python
     import sys
     print(sys.executable)
     ```
   - Open a command prompt or terminal.
   - Use the Blender Python executable path obtained from the previous step to install Pandas:
     ```bash
     "C:\path\to\blender\python.exe" -m ensurepip
     "C:\path\to\blender\python.exe" -m pip install pandas
     "C:\path\to\blender\python.exe" -m pip install openpyxl
     ```

### Running the Processing Script
5. **Execute Processing Script**:
   - Now you are able to run the `blender_functions/processing_blender.py` script, but first process your point cloud ! We'll get to that later.
   - Open a command prompt and run : 
     ```bash
     blender --background --python blender_functions/processing_blender.py
     ```
   - Ensure all dependencies and paths are correctly set for a smooth execution.

## Environment set-up
This code has been tested with Python 3.7, Tensorflow 1.15.5, CUDA 11.2 on Ubuntu 18.04.

1. Clone this repository

  ```sh
  git clone https://github.com/Amsterdam-Internships/IID_Ilum_Mod.git
  ```

2. Install all Python dependencies

  ```sh
 conda env create -f environment.yaml
  ```

3. Build RandLA-Net

  ```sh
  cd utils
  sh compile_op.sh
  mv nearest_neighbors/lib/python/KNN_NanoFLANN-*.egg/* nearest_neighbors/lib/python/.
  ```

## Point Cloud Processing
### Street-level dataset by Cyclomedia
The City of Amsterdam acquired a street-level 3D Point Cloud encompassing the entire municipal area. The point cloud, provided by Cyclomedia Technology, is recorded using both a panoramic image capturing device together with a Velodyne HDL-32 LiDAR sensor, resulting in a point cloud that includes not only <x, y, z> coordinates, but also RGB and intensity data. The 3D Point Cloud is licensed, but it may become open in the future. An example tile is available [here](https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing/tree/main/datasets/pointcloud).

### Usage

This repository is specifically designed for processing 50x50 tiles from the Cyclomedia street-level 3D Point Cloud Dataset. A crucial step in preprocessing involves downsampling the original point clouds, which can contain over 1 million points. This reduction is necessary to make the data manageable for subsequent 3D Semantic Segmentation.

#### Preliminary Data Preparation

Before further processing, we utilize existing research that has identified the coordinates of streetlights in the City of Amsterdam. These coordinates are available in the following CSV file:

- `data/sheets/processed_sheets/clustered_amsterdam.csv`

#### Running the Preprocessing Script

To perform the preprocessing operations, execute the `bb_extraction.py` script. This script extracts and processes bounding boxes from the point cloud data. Use the following command to run the script:

```sh
python bb_extraction.py --in_folder 'demo_data/pc' --out_folder 'demo_data/bb'
```

These steps next step are not mandatory if you only want to run inference. However, it will need to be done to train a model.

```sh
python src/compute_normals.py --input_dir 'demo_data/bb' --output_dir 'demo_data/bb_nm' 
```

```sh
python prepare.py --mode train --in_folder 'demo_data/bb_nm' --out_folder 'demo_data/processed' --use_rgb --use_intensity --use_normals --config_file 'Streetlights3D'
```


### Inference

Now that the bounding boxes are ready, we can utilize a pre-trained model to predict the light source coordinates for the specific streetlights. Follow these steps to complete the inference process:

#### Binary Segmentation

Proceed with binary segmentation to identify the Light Source component of each streetlight using the pre-trained model. Run the command below to conduct the segmentation:

```sh
python3 main.py --mode 'test' --in_folder 'demo_data/bb' --out_folder 'demo_data/predicted' --snap_folder 'model/RGBIN/snapshots' --no_prepare --use_rgb --use_intensity --use_color
```

#### Train a new model

The flags use_intensity, use_normals and use_rgb can be dropped to train models with less parameters.

```sh
python3 main.py --mode 'train' --in_folder 'demo_data/processed' --use_rgb --use_intensity --use_normals --config_file Streetlights3D
```

This command runs in test mode, processes files from processed/laz_bb, saves the predictions in predicted/, and utilizes snapshots from trained_model/RGB_I/snapshots.

#### Merging Predictions with Input Point Cloud

Finally, merge the prediction results with the original point cloud data. This step integrates the segmentation results back into the spatial context of the original data, providing a comprehensive view of the outcomes:

```sh
python3 merge.py --cloud_folder 'demo_data/bb' --pred_folder 'demo_data/predicted' --out_folder 'merged'
```

This command merges prediction files from predicted/laz_bb with the point cloud data from data/laz_pc_data, outputting the merged data to merged.

### Arguments
To enable the use of RGB, intensity or normals during training or inference, you can use the following flags: `--use_rgb`, `--use_intensity`. You can define the configuration for a specific dataset using the flag `--config_file`. For example we can use `--config_file 'Streetligths3D'`. To directly perform inference on the laz files, you need to use the `--no_prepare` flag when running `main.py`.


## Clustering
Now, you should run the clustering.py script to create a csv files with the positions of the light sources listed for each streetlight processed.

```sh
python clustering.py --directory merged --output demo_data/sheets/clustered_amsterdam.csv --label 2
```

## Visual Representation in Blender
Now that the light sources have been identified and their coordinates extracted, it is possible to model them in Blender.
This script creates a blend file for every laz point cloud present in the indicated folder under the name 'laz_folder_path'. It will add artificial lights thanks to blender to the point cloud in the positions of the light sources detected with the segmentation and further clustered. 

As it is a script run through blender, the variable path need to be changed directly in processing_blender.py
Once it is done run in command line : 

```sh
blender --background --python .\processing_blender.py --laz_folder_path 'demo_data/pc' --save_directory 'demo_data/blend_files' --clusters_file_path demo_data/sheets/clustered_amsterdam.csv
```

In the folder blend_files, you should have now blender file containing your visual representation ! 




