# Auto-identification of Supraglacial Lakes on the Greenland ice sheet from Satellite Imagery

> The GitHub repository should have a clearly-readable README file that explains clearly how to run the code contained in the repository.

## File Organization

This repository includes two original gpkg, codes, model, conda environment and the final testing gpkg.

The original tif files are not in this repository, as they are larger than the repository capacity limits. 

Here is the file tree of the repository:


## How to run the code

1. Copy the 8 original tif files (4 tif and 4 xml) into the data folder, put them together with two original gpkg files.

2. Setup conda environment: `conda env create -f conda_env.yaml`

3. Open task.ipynb and run the code.
   
Note:

- Running all code in task.ipynb to re-create the testing gpkg.
- All the cell can run for more than one time, even though it includes file operation.
- The U-Net model is trained on Ubuntu20.04 with 80GB RAM and RTX3090, but it actually uses about 10GB GPU memory and 40GB RAM.
- You can change BATCH_SIZE and IS_TRAIN to decide whether to train the whole model starting from scratch. This will change the model file, as the results of the neural network are not fixed, we use loss to control the consistency of the model here. 
- All processes exclude model training takes about two hours, and training model takes about more than one day.
    