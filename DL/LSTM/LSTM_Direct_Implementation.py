import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau


class ForecastLSTM:
    def __init__(self, random_seed: int = 1234):
        self.random_seed = random_seed

    def reshape_dataset(self, df: pd.DataFrame) -> np.array:
        # y 컬럼을 df의 맨 마지막 위치로 이동
        if "y" in df.columns:
            df = df.drop(columns=["y"]).assign(y=df["y"])
        else:
            raise KeyError("Not found target column 'y' in dataset.")

        dataset = df.values.reshape(df.shape)  # shape 변경
        return dataset

    def split_sequences(
            self, dataset: np.array, seq_len: int, steps: int, single_output: bool
    ) -> tuple:

        # feature와 y 각각 sequential dataset을 반환할 리스트 생성
        X, y = list(), list()
        # sequence length와 step에 따라 sequential dataset 생성
        for i, _ in enumerate(dataset):
            idx_in = i + seq_len
            idx_out = idx_in + steps
            if idx_out > len(dataset):
                break
            seq_x = dataset[i:idx_in, :-1]
            if single_output:
                seq_y = dataset[idx_out - 1: idx_out, -1]
            else:
                seq_y = dataset[idx_in:idx_out, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

# ForecastLSTM.reshape_dataset = reshape_dataset

