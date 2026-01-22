# FLDA-TransUNet

## 1. Prepare Data

Please use the **preprocessed Synapse dataset** for research purposes:

- Preprocessed data: [Google Drive](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd)

## 2.Environment
Please prepare an environment with python=3.8(conda create -n envir python=3.8), and then use the command "pip install -r requirements.txt" for the dependencies.

## 3.Train/Test
Run the train script on synapse dataset. 
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16


Run the test script on synapse dataset.
```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
