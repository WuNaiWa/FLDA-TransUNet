# FLDA-TransUNet

## 1. Prepare Data

Please use the **preprocessed Synapse dataset** for research purposes:

- Preprocessed data (Google Drive): https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd

3.Environment
Please prepare an environment with python=3.7(conda create -n envir python=3.7.12), and then use the command "pip install -r requirements.txt" for the dependencies.

4.Train/Test
Run the train script on synapse dataset. The batch size can be reduced to 12 or 16 to save memory(please also decrease the base_lr linearly), and both can reach similar performance.

CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.
python test.py --dataset Synapse --vit_name R50-ViT-B_16
