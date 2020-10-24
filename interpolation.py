<<<<<<< HEAD
import autograd
import autograd.numpy as np
import matplotlib as mlt 
import matplotlib.pyplot as plt 
'''fonction gamma qui marche

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
'''

def gamma(t, P1, P2, u1, u2):
        if not(0 <= t <= 1):
            return "t doit être compris entre 0 et 1"
        vect_12 = np.array([P2[0]-P1[0],P2[1]-P1[1]])
        if (2*vect_12[0] != (u1 + u2)[0]) or (2*vect_12[1] != (u1 + u2)[1]):
            print("pas d'interpolation : on trace le segment")
            X = P1 + t * vect_12
            return X
        else:
            print("On peut faire l'interpolation")
            X = P1 + t * u1 + t * t * (vect_12-u1) 
            return X


def gamma_1(t):
    P1, P2 = np.array([0, 1]), np.array([2, 3])
    u1, u2 = np.array([1, 2]), np.array([10, 2])
    return gamma(t, P1, P2, u1, u2 )

gamma_vect = np.vectorize(gamma_1)

'''
t1 = 0
t2 = 0.3
t3 = 0.6
t4 = 1
X1, X2, X3, X4 = gamma(t1, P1, P2, u1, u2), gamma(t2, P1, P2, u1, u2), gamma(t3, P1, P2, u1, u2), gamma(t4, P1, P2, u1, u2)
plt.plot ([X1[0], X2[0], X3[0], X4[0]],[X1[1], X2[1], X3[1], X4[1]])
plt.show()
print(X1, X2, X3, X4)

'''

t = np.linspace(0, 1, 5)
print(gamma_vect(np.array([0.5, 1])))








'''
def f(x):
    return 2*x

f_vect = np.vectorize(f)

print(f(2))
L = np.array([1,2,3,4])
print(f(L))
'''
=======
import autograd
import autograd.numpy as np
import matplotlib as mlt 
import matplotlib.pyplot as plt 
'''fonction gamma qui marche

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
'''

def gamma(t, P1, P2, u1, u2):
        if not(0 <= t <= 1):
            return "t doit être compris entre 0 et 1"
        vect_12 = np.array([P2[0]-P1[0],P2[1]-P1[1]])
        if (2*vect_12[0] != (u1 + u2)[0]) or (2*vect_12[1] != (u1 + u2)[1]):
            print("pas d'interpolation : on trace le segment")
            X = P1 + t * vect_12
            return X
        else:
            print("On peut faire l'interpolation")
            X = P1 + t * u1 + t * t * (vect_12-u1) 
            return X


def gamma_1(t):
    P1, P2 = np.array([0, 1]), np.array([2, 3])
    u1, u2 = np.array([1, 2]), np.array([10, 2])
    return gamma(t, P1, P2, u1, u2 )

gamma_vect = np.vectorize(gamma_1)

'''
t1 = 0
t2 = 0.3
t3 = 0.6
t4 = 1
X1, X2, X3, X4 = gamma(t1, P1, P2, u1, u2), gamma(t2, P1, P2, u1, u2), gamma(t3, P1, P2, u1, u2), gamma(t4, P1, P2, u1, u2)
plt.plot ([X1[0], X2[0], X3[0], X4[0]],[X1[1], X2[1], X3[1], X4[1]])
plt.show()
print(X1, X2, X3, X4)

'''

t = np.linspace(0, 1, 5)
print(gamma_vect(np.array([0.5, 1])))








'''
def f(x):
    return 2*x

f_vect = np.vectorize(f)

print(f(2))
L = np.array([1,2,3,4])
print(f(L))
'''
>>>>>>> 0d44c7e7b8ec2b944a4aed2732f595e2117363f9
