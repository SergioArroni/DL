import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import History


class ModelTrainer():

    def __init__(self):
        self._histories: list = []
        self.trained: bool = False

    def __reset__(self) -> None:
        self._histories = []
        self.trained = False

    def compute_ovefitting(self, history: History = None, range=0.2) -> float:
        if history == None:
            if self.trained == True:
                history = self._histories[-1]
            else:
                raise Exception(
                    "Argument history empty and training not completed.")

        res_df = pd.DataFrame(history.history)
        n_size = int(res_df.shape[0] *
                     range) if int(res_df.shape[0] * range) > 0 else 1

        bin_ac = res_df['binary_accuracy'][-n_size:]
        val_bin_ac = res_df['val_binary_accuracy'][-n_size:]
        overfitting = bin_ac - val_bin_ac
        total_ov = 0
        for item in overfitting:
            total_ov += item
        return total_ov/range

    def compute_ac_increase(self, history: History = None, range=0.1) -> list:
        if history == None:
            if self.trained == True:
                history = self._histories[-1]
            else:
                raise Exception(
                    "Argument history empty and training not completed.")

        res_df = pd.DataFrame(history.history)
        n_size = int(res_df.shape[0] *
                     range) if int(res_df.shape[0] * range) > 0 else 1

        results = []

        for col in [res_df['binary_accuracy'], res_df['val_binary_accuracy']]:
            bin_ac1 = col[:n_size]
            bin_ac2 = col[-n_size:]
            avg_bin_ac1 = 0
            avg_bin_ac2 = 0
            for elem in bin_ac1:
                avg_bin_ac1 += elem
            avg_bin_ac1 = avg_bin_ac1/n_size

            for elem in bin_ac2:
                avg_bin_ac2 += elem
            avg_bin_ac2 = avg_bin_ac2/n_size
            results.append(avg_bin_ac2 - avg_bin_ac1)

        return results

    def auto_fit(self, model: Sequential, X_train: list, t_train: list, X_val: list, t_val: list, auto_plot=False, epochs_per_step=20, initial_lr=0.001, limit_lr=0.001/1000, ac_increase_threshold=0.005, ac_increase_range=0.2) -> list:
        self.__reset__()
        lr = initial_lr
        histories = []
        div = 2
        ep_step = 1

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.legacy.Adadelta(
                          learning_rate=lr),
                      metrics=keras.metrics.binary_accuracy)

        while (lr > limit_lr):

            history = model.fit(X_train, t_train, batch_size=32, epochs=epochs_per_step, verbose=0, validation_data=(
                X_val, t_val))

            histories.append(history)
            self.trained = True
            self._histories.append(history)

            print('Epoch_step : {_ep_step} ; learning_rate : {_lr}'.format(
                _ep_step=ep_step, _lr=lr))
            logs = pd.DataFrame(history.history)
            print(logs)

            if auto_plot == True:
                self.plot_histories()

            ep_step += 1

            if self.compute_ac_increase(history, range=ac_increase_range)[1] < ac_increase_threshold:
                lr = lr/div
                div = (div + 3) % 6

                model.compile(loss=keras.losses.binary_crossentropy,
                              optimizer=keras.optimizers.legacy.Adam(
                                  learning_rate=lr),
                              metrics=keras.metrics.binary_accuracy)

        return histories

    def plot_histories(self, histories: list = None) -> None:
        if histories == None:
            if self.trained == True:
                histories = self._histories
            else:
                raise Exception(
                    "Argument history empty and training not completed.")

        histories = histories.copy()
        results = pd.DataFrame(histories[0].history)
        del histories[0]
        for history in histories:
            results = pd.concat([results, pd.DataFrame(
                history.history)], ignore_index=True)
        results.plot(figsize=(8, 5))
        plt.grid(True)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy - Mean Log Loss")
        plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
        plt.show()
        return
