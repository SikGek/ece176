"""
K Nearest Neighbours Model
"""
import numpy as np


class KNN(object):
    def __init__(
        self,
        num_class: int
    ):
        self.num_class = num_class

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        k: int
    ):
        """
        Train KNN Classifier

        KNN only need to remember training set during training

        Parameters:
            x_train: Training samples ; np.ndarray with shape (N, D)
            y_train: Training labels  ; snp.ndarray with shape (N,)
        """
        self._x_train = x_train
        self._y_train = y_train
        self.k = k

    def predict(
        self,
        x_test: np.ndarray,
        k: int = None,
        loop_count: int = 1
    ):
        """
        Use the contained training set to predict labels for test samples

        Parameters:
            x_test    : Test samples                                     ; np.ndarray with shape (N, D)
            k         : k to overwrite the one specificed during training; int
            loop_count: parameter to choose different knn implementation ; int

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        k_test = k if k is not None else self.k

        if loop_count == 1:
            _, idc = self.calc_dis_one_loop(x_test)
        elif loop_count == 2:
            _, idc = self.calc_dis_two_loop(x_test)
        #(500, 5000) distance value with each of the images in train
        #get the k smallest values and their indices
        #array of shape (500, k)
        #using the indices get the classes of the images
        # Your Code Here
        out = self._y_train[idc]
        most_f = np.array([np.bincount(row).argmax() for row in out])
        return most_f

    def calc_dis_one_loop(
        self,
        x_test: np.ndarray
    ):
        """
        Calculate distance between training samples and test samples

        This function could one for loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
        distances = []
        idc = []
        for i in range(len(x_test)) :
            distances.append(np.linalg.norm(x_test[i]-self._x_train, axis = 1))
            res = sorted(range(distances[i].shape[0]), key = lambda sub: distances[i][sub])[:self.k]
            idc.append(res)
        return np.array(distances), np.array(idc)

    def calc_dis_two_loop(
        self,
        x_test: np.ndarray
    ):
        """
        Calculate distance between training samples and test samples

        This function could contain two loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
        distances = []
        idc = []
        for i in range(len(x_test)) :
            distance = []
            for j in range(len(self._x_train)) :
                distance.append(np.linalg.norm(x_test[i]-self._x_train[j]))
            res = sorted(range(len(distance)), key = lambda sub: distance[sub])[:self.k]
            idc.append(res)
            distances.append(np.array(distance))
        return np.array(distances), np.array(idc)
