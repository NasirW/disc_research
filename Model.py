import os
from os.path import join as path_join
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import pickle as pk
from argparse import ArgumentParser as arg_parser
from tqdm import tqdm
from glob import glob

from colorama import Fore as F
from colorama import Style as S
from colorama import init

from ImageLoader import Path
from DenseNet import DenseNet

init(autoreset=True)

class Model:
    BEN = 0
    MAL = 1

    def __init__(self, data_dir, save_dir, verbose):
        self.save_dir = save_dir
        self.random_state = 42
        self.verbose = verbose
        
        npy_files = glob(path_join(data_dir, "*.npy"))
        self.data = {}

        for npy_file in tqdm(npy_files, desc="Loading Data"):
            name = Path.get_filename(npy_file)
            self.data[name] = np.load(npy_file, allow_pickle=True)
            
        self.dense_net = DenseNet(X_tr=self.data["X_train"],
                                  Y_tr=self.data["y_train"],
                                  X_ts=self.data["X_val"],
                                  Y_ts=self.data["y_val"])


    def train(self, history_save_path):
        self.history = self.dense_net.train(history_save_path)
        self.dense_net.plot_loss(self.history)
        self.dense_net.plot_loss_from_csv(history_save_path)
        
    def predict(self, X=None):
        ...
        # if X is None:
        #     X = self.x_test
        
        # return self.model.predict(X=X)
    
    def predict_proba(self, X=None):
        ...
        # if X is None:
        #     X = self.x_test
        
        # try:
        #     return self.model.predict_proba(X=X)
        # except AttributeError:
        #     return None

    def evaluate(self, X=None, y_true=None) -> None:
        ...
        # if X is None and y_true is None:
        #     X, y_true = self.x_test, self.y_test
            
        # # # #  COMPUTING PREDICTIONS  # # #
        # y_pred, y_pred_prob = self.predict(X=X), self.predict_proba(X=X)

        # self.metrics = {
        #     "Overall Score":    accuracy_score(y_true, y_pred),
        #     "Balanced Score":   balanced_accuracy_score(y_true, y_pred),
        #     "confusion_matrix": confusion_matrix(y_true, y_pred, normalize="true")
        # }

        # self.metrics["AUC"] = roc_auc_score(y_true, y_pred_prob[:, 1]) \
        #                       if y_pred_prob is not None else None


        # # # #  PRINTING RESULTS  # # #
        # if self.verbose:
        #     print()
            
        # print(F.RED + f"{self.model_name} ~ Model Evaluation:")

        # for name, score in self.metrics.items():
        #     if name == "confusion_matrix":
        #         continue
        #     if score is None:
        #         print(S.DIM + f"{name} Metric Not Computed")
        #     else:
        #         print(F.BLUE + f"{name}:\t{score:.3f}")

        # cmd = ConfusionMatrixDisplay(self.metrics["confusion_matrix"],
        #                              display_labels=[f"Benign", f"Malignant"])
        # cmd.plot(cmap="Purples", im_kw={"vmin": 0})

        # plt.title(f"{self.model_name} - Model Confusion Matrix")
        # plt.yticks(rotation=45)

        # cfn_mtx_dir = path_join(self.save_dir, "confusion_matrices")
        # cfn_mtx_path = path_join(cfn_mtx_dir, f"{self.method['abv']}_confusion_matrix.png")

        # if os.path.exists(cfn_mtx_path):
        #     os.remove(cfn_mtx_path)

        # os.makedirs(cfn_mtx_dir, exist_ok=True)
        # plt.savefig(cfn_mtx_path)
        # print(F.GREEN + f"\nSaved {self.method['abv']} Confusion Matrix: `{cfn_mtx_path}`", end="\n\n")


if __name__ == '__main__':
    parser = arg_parser()
    
    parser.add_argument("--verbose", default=1,
                        help="Controls the verbosity of the model fitting",
                        choices=[0, 1, 2, 3], type=int)
    
    parser.add_argument("--data_dir", required=True,
                        help="The directory containing the `.npy` data files", type=str)

    args = parser.parse_args()

    model = Model(data_dir=args.data_dir,
                  save_dir=path_join(args.data_dir, "model"),
                  verbose=args.verbose)
    
    model.train(path_join(args.data_dir, "DenseNET_train_loss_meta"))