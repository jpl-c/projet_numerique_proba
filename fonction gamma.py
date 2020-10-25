import autograd
import autograd.numpy as np
import matplotlib as mlt 
import matplotlib.pyplot as plt


def gamma(t, P1, P2, u1, u2):
    if not(0 <= t <= 1):
        return "t doit être compris entre 0 et 1"
    vect_12 = np.array([P2[0]-P1[0],P2[1]-P1[1]])
    if (2*vect_12[0] != (u1 + u2)[0]) and (2*vect_12[1] != (u1 + u2)[1]):
        print("pas d'interpolation : on trace le segment")
        X = P1 + t * vect_12
        return X
    else:
        print("On peut faire l'interpolation")
        X = P1 + t * u1 + t * t * (vect_12-u1) 
        return X