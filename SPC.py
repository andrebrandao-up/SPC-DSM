import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import ConceptDriftStream
from skmultiflow.core import clone

class GeneratorFromFile():
    """Generator from file

    Creates a datastream generator from a file.

    Parameters
    ----------

    filename: string.
        Path to the dataset file.
    """

    def __init__(self, filename):
        self.data = pd.read_csv(filename)

    def prepare_for_use(self):
        self.pointer = -1

    def has_more_samples(self):
        return self.pointer+1 < self.data.shape[0]

    def next_sample(self):
        self.pointer += 1
        if self.pointer >= self.data.shape[0]:
            return None, None
        return np.array(self.data.iloc[self.pointer, :-1]).reshape(1 ,-1), np.array([self.data.iloc[self.pointer, -1]])


class SPC():
    """Statistical Process Control

    This algorithm adds a concept drift detection to existing machine
    learning models.

    Parameters
    ----------
    model: Scikit-multiflow model.
        The machine learning model to add the concept drift detection.

    stream: Scikit-multoflow stream object.
        The reference to the stream of data.

    alpha: int, default=3.
        The SPC parameter.

    beta: int, default=2.
        The SPC parameter.

    with_buffer: bool, default=True.
        Whether to use the buffer to store the observations in the warning
        stage or not.
    """

    def __init__(self, model, stream, alpha=3, beta=2, with_buffer=True):
        self.model0 = model
        self.model  = clone(model)
        self.stream = stream
        self.alpha = alpha
        self.beta = beta
        self.with_buffer = with_buffer

    def start(self):
        """Initial setup.
        """
        self.stream.prepare_for_use()
        self.errors0 = []
        self.errors = []
        self.pointer = 0
        self.finished = False
        self.log = {}
        self.log_key = 0
        self.__spc_thread = threading.Thread(target=self.__SPC)
        self.__spc_thread.start()

    def __SPC(self):
        p_min, s_min = None, None
        n_errors, n = 0, 0
        n_errors0, n0 = 0, 0
        warning = False
        buffer = []
        while self.stream.has_more_samples():
            # Collect sample from stream.
            X, y = self.stream.next_sample()
            # Create predictions.
            pred0 = self.model0.predict(X)
            pred  = self.model.predict(X)
            # Compute loss function.
            L0 = pred0[0] != y[0]
            L  = pred[0] != y[0]
            # Update the errors counting.
            n_errors0 += L0
            n_errors  += L
            # Update the samples counting.
            n0 += 1
            n  += 1
            # Compute the mean p_j and variance s_j of the errors
            # for SPC model.
            p_j = n_errors/n
            s_j = np.sqrt(p_j*(1-p_j)/n)
            # Save mean p_j for both models.
            self.errors0.append(n_errors0/n0)
            self.errors.append(p_j)
            # CONTROL BLOCK
            if p_min == None or (p_j + s_j) < (p_min + s_min):
                p_min = p_j
                s_min = s_j
            if (p_j + s_j) <= (p_min + self.beta*s_min):
                # In-Control
                if warning or (self.log_key-1 in self.log and self.log[self.log_key-1] == "Drift"):
                    self.log[self.log_key] = "In Control"
                warning = False
                self.model.partial_fit(X, y)
            else:
                if (p_j+s_j) <= (p_min + self.alpha*s_min):
                    # Warning Zone
                    if not warning:
                        buffer = [(X, y)]
                        if not warning:
                            self.log[self.log_key] = "Warning"
                        warning = True
                    else:
                        buffer.append((X, y))
                else:
                    # Out-Control
                    self.log[self.log_key] = "Drift"
                    # Re-learn a new decision model using the examples in the buffer
                    self.model.reset()
                    warning = False
                    temp_n_errors = 0
                    temp_n = 0
                    if self.with_buffer and buffer != []:
                        for X, y in buffer:
                            pred = self.model.predict(X)
                            temp_n_errors += (pred[0] != y[0])
                            temp_n += 1
                            self.model.partial_fit(X, y)
                        n_errors, n = temp_n_errors, temp_n
                        p_min = n_errors/n
                        s_min = np.sqrt(p_min*(1-p_min)/n)
                    else:
                        p_min, s_min = None, None
                        n_errors, n = 0, 0
            # Always update the baseline model without control.
            self.model0.partial_fit(X, y)
            self.log_key += 1
        self.finished = True

    def get_next_errors(self, n):
        """Obtain the next n errors.

        Parameters
        ----------

        n: int.
            The number of errors to return.

        Returns
        -------
        (errors0, errors): tupple of two lists.
            The n errors for the model with and without the SPC.
        """
        if self.pointer >= len(self.errors):
            return ([], [])
        if n > len(self.errors):
            n = len(self.errors)
        errors0 = self.errors0[self.pointer:self.pointer+n]
        errors = self.errors[self.pointer:self.pointer+n]
        self.pointer += n
        return (errors0, errors)

    def get_errors(self):
        """Obtain all the errors.

        Returns
        -------
        (errors0, errors): tupple of two lists.
            All errors for the model with and without the SPC.
        """
        return (self.errors0, self.errors)

    def has_messages(self):
        """Check if there is messages in the log.

        Returns
        -------
        log: bool.
            True if there is messages to read in the log.
        """
        return self.log

    def get_messages(self):
        """Obtain the messages in the log.

        Returns
        -------
        log: list.
            The messages to read in the log.
        """
        return self.log

    def has_finished(self):
        """Check if the algorithm has finished.

        Returns
        -------
        log: bool.
            True if the algorithm has finished.
        """
        return self.finished

    def dataset_size(self):
        """Obtain the dataset size.

        Returns
        -------
        size: int.
            The dataset size.
        """
        return self.data.shape[0]
