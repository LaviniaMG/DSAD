import numpy as np


def standardizare(X):  # asumam ca primim ca parametru un numpy.ndarray
    medii = np.mean(a=X, axis=0)  # medii pe coloane
    print(medii.shape)
    abateriStd = np.std(a=X, axis=0)  # avem variabilele pe coloane
    return (X - medii) / abateriStd
