import autograd
import autograd.numpy as np
import matplotlib as mlt 
import matplotlib.pyplot as plt


def gamma(t, P1, P2, u1, u2):
    if not(0 <= t <= 1):
        return "t doit être compris entre 0 et 1"

    vect_12 = np.array([P2[0]-P1[0],P2[1]-P1[1]])
    P12_scal_u1 = u1[0]*vect_12[0] + u1[1]*vect_12[1]
    P12_scal_u2 = u2[0]*vect_12[0] + u2[1]*vect_12[1]
    u1_mixte_u2 = u1[0]*u2[1] - u2[0]*u1[1]	
    
    if P12_scal_u1 < 0 or P12_scal_u2 < 0 or u1_mixte_u2 == 0:
        print("pas d'interpolation : on trace le segment")
        X = P1 + t * vect_12
        return X
    else:
        ("On peut faire l'interpolation")
        M = np.array([[u1[0], u2[0]],
                      [u1[1], u2[1]]])
        M_inv = np.linalg.inv(M)
        vect_coeff = np.dot(M_inv, vect_12)
        l = vect_coeff[0]
        X = P1 + t * l*u1 + t * t * (vect_12-l*u1) 
        return X

f_gamma = lambda t : gamma(t, P1, P2, u1, u2)
P1, P2 = np.array([1,2]), np.array([4,5])
u1, u2 = np.array([0,5]), np.array([1,10])
tab_val = np.empty(shape = (2, 100))
tab_t = np.linspace(0, 1, 100)
for i in range(100):
    t = tab_t[i]
    tab_val[0][i] = f_gamma(t)[0]
    tab_val[1][i] = gamma(t, P1, P2, u1, u2)[1]

plt.plot(tab_val[0], tab_val[1], 'g')
plt.plot(P1[0], P1[1], 'bo')
plt.plot(P2[0], P2[1], 'ro')

'''
tab_t1 = np.linspace(-1, 1, 100)
tab_tan1 = np.empty(shape = (2, 100))
tab_tan2 = np.empty(shape = (2, 100))''' 


plt.axis('equal')
plt.show()
