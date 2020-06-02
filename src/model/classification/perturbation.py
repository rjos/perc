from math import fabs, sqrt, ceil, floor
from abc import abstractmethod
import numpy as np


class Base:

    def __init__(self):
        self.__unique_classes = None
        self.__count_classes = None

    @property
    def unique_classes(self):
        """ Return unique classes in trainig data

        Return
        ------
        unique_classes: ordered array
                        Unique classes
        """
        return self.__unique_classes

    @property
    def count_classes(self):
        """ Return the number of instance in each class

        Return
        ------
        count_classes: array_like shape (n_classes, 1)
                       The instance number in each class
        """
        return self.__count_classes

    @abstractmethod
    def _fit(self, X, y):
        """ Fit model based of pertubation for each classifier

        Parameters
        ----------
        X: array_like shape (n_samples, n_features)
           Training data
        y: numpy array shape (n_samples)
           Target values
        """
        ...

    def fit(self, X, y):
        """ Fit model based of pertubation classifier

        Parameters
        ----------
        X: array_like shape (n_samples, n_features)
           Training data
        y: numpy array shape (n_samples)
           Target values
        """
        unique_classes, count_classes = np.unique(y, return_counts=True)
        unique_classes = unique_classes.astype(int)
        count_classes = count_classes.reshape(unique_classes.shape[0], 1)

        self.__unique_classes = unique_classes
        self.__count_classes = count_classes

        self._fit(X, y)

    @abstractmethod
    def _predict(self, X):
        """ Predict class labels for samples in X for each classifier

        Parameters
        ----------
        X: array_like shape (n_samples, n_features)
           Samples

        Return
        ------
        y: array_like shape (n_samples)
           Predicted class label per sample
        """
        ...

    def predict(self, X):
        """ Predict class labels for samples in X

        Parameters
        ----------
        X: array_like shape (n_samples, n_features)
           Samples

        Return
        ------
        y: array_like shape (n_samples)
           Predicted class label per sample
        """
        return self._predict(X)

    @abstractmethod
    def _pertubations(self, X):
        """
        """
        ...

    def pertubations(self, X):
        return self._pertubations(X)


class PerC_Mean(Base):

    def __init__(self):
        super(PerC_Mean, self).__init__()
        self.__means = None

    @property
    def means(self):
        """ Return means of the classes

        Return
        ------
        means: array_like shape (n_classes, 1)
               The means of each class

        """
        return self.__means

    def _fit(self, X, y):
        __X = X
        __y = y

        # Get the instances belongs to each class
        instance_classes = list(map(lambda w: __X[__y == w], self.unique_classes))

        # Compute the mean by eq. 6
        means = list(map(lambda x, c: x.sum(axis=0) / c, instance_classes, self.count_classes))
        self.__means = np.array(means)

    def _compute_perturbations_means(self, X):
        """ Compute perturbation mean for each class

        Parameters
        ----------
        X: array_like shape (n_samples, n_features)
           Samples

        Return
        ------
        perturbations: array_like shape (n_samples, n_classes, n_features)
                       Generated perturbations to means
        """
        perturbations_means = list(map(lambda x: (x - self.means) / (self.count_classes + 1), X))
        return np.array(perturbations_means)

    def _predict(self, X):
        __X = X

        # Compute pertubations
        pertubations = self._pertubations(__X)

        # Get class labels for each class
        predicted = list(map(lambda pertubation: self.unique_classes[pertubation.argmin()], pertubations))

        return predicted

    def _pertubations(self, X):
        __X = X

        # Compute perturbations means for each class from eq. 13
        perturbations_means = self._compute_perturbations_means(__X)

        # Compute euclidean norm
        norms = np.array(list(map(lambda per: list(map(np.linalg.norm, per)), perturbations_means)))

        return norms


class PerC_Covariance(PerC_Mean):

    def __init__(self):
        super(PerC_Covariance, self).__init__()
        self.__covariances = None

    @property
    def covariances(self):
        """ Return coraviances matrix for each class

        Return
        ------
        covariances: list size (n_classes)
                     Covariances matrix
        """
        return self.__covariances

    @covariances.setter
    def _covariances(self, value):
        """ Set covariances matrix for each class

        Parameters
        ----------
        value: list size (n_classes)
               Covariances matrix
        """
        self.__covariances = value

    def _fit(self, X, y):
        __X = X
        __y = y

        # Compute means of each class
        super()._fit(X, y)

        # Get the instances belongs to each class
        instance_classes = np.array(list(map(lambda w: __X[__y == w], self.unique_classes)))

        # Compute covariances matrix for each class
        covariances = list(
            map(lambda x, m, c: np.dot((x - m).T, (x - m)) / c, instance_classes, self.means, self.count_classes))
        self._covariances = np.array(covariances)

    def _compute_perturbations_covariances(self, X):
        """ Compute perturbation covariances for each class

        Parameters
        ----------
        X: array_like shape (n_samples, n_features)
           Samples

        Return
        ------
        perturbations: array_like shape (n_samples, n_classes, n_features)
                       Generated perturbations to covariances
        """
        # Compute difference between each test instance and each mean of the class
        perturbations_covariances = list(map(lambda x: list(map(lambda m: (x - m).reshape(len(x), 1), self.means)), X))

        # Compute the covariance matrix between each test instance and each mean of the class
        perturbations_covariances = list(
            map(lambda perts: list(map(lambda per, n: np.dot(per, per.T) / (n + 1), perts, self.count_classes)),
                perturbations_covariances))

        # Compute the perturbations covariances
        perturbations_covariances = list(map(lambda perts: list(
            map(lambda cov, per, n: ((-(cov / n)) + per), self.covariances, perts, self.count_classes)),
                                             perturbations_covariances))
        return np.array(perturbations_covariances)

    def _predict(self, X):
        __X = X

        # Compute perturbations
        pertubations = self._pertubations(__X)

        # Get class labels for each class
        predicted = list(map(lambda pertubation: self.unique_classes[pertubation.argmin()], pertubations))

        return predicted

    def _pertubations(self, X):
        __X = X

        # Compute perturbations means for each class from eq. 22
        perturbations_covariances = self._compute_perturbations_covariances(__X)

        # Compute Frobenius norm
        norms = np.array(list(
            map(lambda perts: list(map(lambda per: np.linalg.norm(per, 'fro'), perts)), perturbations_covariances)))

        return norms


class PerC(PerC_Covariance):

    def __init__(self):
        super(PerC, self).__init__()
        self.__inverse_covariances = None

    @property
    def _inverse_covariances(self):
        """ """
        return self.__inverse_covariances

    @_inverse_covariances.setter
    def _inverse_covariances(self, value):
        """ """
        self.__inverse_covariances = value

    def _fit(self, X, y):
        super()._fit(X, y)

        # Compute pseudo-inverse matrix for each covariances
        inverse_covariances = list(map(np.linalg.pinv, self._covariances))
        self._inverse_covariances = np.array(inverse_covariances)

    def _predict(self, X):
        __X = X

        # Compute perturbations
        pertubations = self._pertubations(__X)

        # Get class labels for each class
        predicted = list(map(lambda pertubation: self.unique_classes[pertubation.argmin()], pertubations))

        return predicted

    def _pertubations(self, X):
        __X = X

        # Compute the perturbations means
        perturbations_means = super()._compute_perturbations_means(__X)

        # Compute the perturbations covariances
        perturbations_covariances = super()._compute_perturbations_covariances(__X)

        # Compute the difference between each test instance and each mean of the class
        difference_means = list(map(lambda x: list(map(lambda m: (x - m).reshape(len(x), 1), self.means)), __X))

        # Compute the expression for the means
        combinations_perturbations_means = list(map(lambda x, d_means: list(
            map(lambda inv, d_mean: np.dot((-2 * inv), d_mean).T, self._inverse_covariances, d_means)), __X,
                                                    difference_means))

        # Compute the multiplication between perturbation mean of each class and its values calculate before
        combinations_perturbations_means = list(
            map(lambda perts, means: list(map(lambda per, m: np.dot(m, per), perts, means)), perturbations_means,
                combinations_perturbations_means))
        combinations_perturbations_means = np.array(list(map(np.concatenate, combinations_perturbations_means)))

        # Compute the expression for the covariance matrix.
        combinations_perturbations_covariances = list(map(lambda d_means: list(
            map(lambda d_mean, inv: (inv - np.dot(np.dot(np.dot(inv, d_mean), d_mean.T), inv)), d_means,
                self._inverse_covariances)), difference_means))

        # Compute the multiplication between the value calculate before and its perturbation covariance.
        combinations_perturbations_covariances = list(
            map(lambda perts, covs: list(map(lambda per, cov: np.dot(per, cov), perts, covs)),
                perturbations_covariances, combinations_perturbations_covariances))

        # Compute trace of each result matrix calculate before
        combinations_perturbations_covariances = list(
            map(lambda perts: list(map(np.matrix.trace, perts)), combinations_perturbations_covariances))
        combinations_perturbations_covariances = np.array(combinations_perturbations_covariances)

        # Compute the combination perturbation for each class
        combinations_perturbations = list(map(lambda means, covs: means + covs, combinations_perturbations_means,
                                              combinations_perturbations_covariances))

        # Compute Absolute value
        norms = np.array(list(map(lambda perts: list(map(np.fabs, perts)), combinations_perturbations)))

        return norms