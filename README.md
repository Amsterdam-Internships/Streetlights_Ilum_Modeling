# PC_BB: Pole Detection in Point Clouds
Welcome to the PC_BB repository! This project is part of an internship with the City of Amsterdam, with a primary goal of modelling light illumination in a 3D model of Amsterdam!

## Getting Started with Python and Blender API

### Installation of Blender Software
1. **Download Blender**: Visit [Blender's official website](https://www.blender.org/) and download Blender 4.0.

### Configuring Blender in System Path
2. **Set Blender as a PATH variable**:
   - Navigate to the Blender Foundation folder, typically located in "Program Files" or your chosen installation directory.
   - Find the `blender.exe` executable, usually found at a path similar to `C://...//Blender Foundation\Blender 4.0`.
   - Add this path to your system's PATH environment variable for easy access.

### Collada File Preparation
3. **Download Collada Cube**:
   - Visit [3D Amsterdam](https://3d.amsterdam.nl/) and download a Collada cube.
   - Move the downloaded cube into the `dataset/dae` folder in your project directory.

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
   - Now you can run the `blender_functions/processing_blender.py` script.
   - Open a command prompt and run : 
     ```bash
     blender --background --python blender_functions/processing_blender.py
     ```
   - Ensure all dependencies and paths are correctly set for a smooth execution.

## Getting Started with Intrinsic Image Decomposition on Point Cloud

1. **Model Selection for Fine-Tuning**
   - Begin by choosing a model to fine-tune from the `pretrained_models` folder.
   - You can then fine-tune this model using three distinct loss functions, each controlled by its coefficient.
   - Configure the training settings using the `train.py` script. Adjust the loss coefficients and other parameters according to your needs. Here's an example command to get started:

     ```bash
     python train.py --include_loss_alb_smoothness True \
                     --loss_alb_smoothness_coeff $loss_alb_coeff \
                     --include_chroma_weights True \
                     --include_loss_shading True --loss_shading_coeff $loss_shd_coeff \
                     --include_loss_lid True --loss_lid_coeff $loss_lid_coeff \
                     --lr 0.00000001 --wandb True --epochs 300
     ```

2. **Testing the Trained Model**
   - Once training is complete, the model will be saved in the `pre_trained_models` directory.
   - To test the model, use the `test.py` script. Make sure to specify the correct paths for the test point cloud and normal maps.
   - Below is an example command for testing:

     ```bash
     python test.py --path_to_model "$model_file" \
                    --path_to_test_pc './Data/pcd/pcd_split_clean_0.5_test/' \
                    --path_to_test_nm './Data/gts/nm_split_clean_0.5_test/'
     ```


## Acknowledgements
Special thanks to the City of Amsterdam for their support and collaboration in this internship project.
