# DISC Research
Code for the 2023-2024 Tufts DISC Resarch project regarding an analysis of chemotherapy resistance and classification of lesion images.

## Repository Contents
* `README.md` - this file
* `.gitignore` - specifies intentionally untracked files to ignore
* `.vscode/` - contains VS Code settings and configurations
* `gen/` - contains generated files like `training_test_splits.csv`
* `given_files/` - contains Python scripts for data preparation, model evaluation, and model training
  * `Data_prep.py` - script for preparing the data
  * `EvaluateCNN.py` - script for evaluating the Convolutional Neural Network
  * `TrainDenseNET.py` - script for training the DenseNET model
* `gpu_run.sh` - shell script for running the project on a GPU
* `run.sh` - shell script for running the project
* `trained_models/` - contains trained models and their logs
  * `DenseNET_train_loss.hdf5` - the trained DenseNET model
  * `DenseNET_train_loss_train_log.csv` - the training log for the DenseNET model