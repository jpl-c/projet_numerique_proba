# level_curve3
import autograd
import autograd.numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['figure.figsize'] = [10, 10] # [width, height] (inches). 
from IPython.display import display


def grad(f):
    g = autograd.grad
    def grad_f(x, y):
        return np.array([g(f, 0)(x, y), g(f, 1)(x, y)])
    return grad_f
def J(f):
    j = autograd.jacobian
    def J_f(x, y):
        return np.array([j(f, 0)(x, y), j(f, 1)(x, y)]).T
    return J_f

def f1(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return 3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2 
def f2(x1, x2):
    return (x1 - 1)**2 + (x1 - x2**2)**2
def f3(x, y):
    return np.sin(x + y) - np.cos(x * y) - 1 + 0.001 * (x * x + y * y)
def F1(x,y):
    return np.array([f1(x,y)-0.8, x-y])

def Newton(F, x0, y0, eps, N):
    J_F = J(F)
    for i in range(N):
        X0 = np.array([x0, y0])
        J_F_inv = np.linalg.inv(J_F(X0[0], X0[1])) # calcul de l'inverse de la jacobienne en x0, y0
        X = X0 - np.dot(J_F_inv,F(X0[0], X0[1])) # calcul du nouveau point X =(x, y)
        x, y = X[0], X[1]
        if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return (x, y)
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")

def intersection(S1,S2):
    x11, y11 = S1[0][0], S1[0][1]
    x12, y12 = S1[1][0], S1[1][1]
    x21, y21 = S2[0][0], S2[0][1]
    x22, y22 = S2[1][0], S2[1][1]
    if (x22-x21)*(y12-y11) == (x12-x11)*(y22-y21) :
        #segments colinéaires
        if max(min(x11,x12),min(x21,x22)) <= min(max(x12,x11),max(x21,x22)) :
            #les segments se chevauchent
            return True
        else :
            return False
    else :
        uAB = ((x22-x21)*(y21-y11)-(x21-x11)*(y22-y21))/((x22-x21)*(y12-y11)-(x12-x11)*(y22-y21))
        uCD = ((x12-x11)*(y21-y11)-(x21-x11)*(y12-y11))/((x22-x21)*(y12-y11)-(x12-x11)*(y22-y21))
        if uAB > 0 and uAB < 1 and uCD > 0 and uCD < 1 :
            #il y a une intersection
            return True
        else :
            return False

def level_curve(f, x0, y0, delta=0.01, N=350, eps=0.001):
    c = f(x0, y0)
    def level_curve_1step(f, x2, y2, delta1 = delta, N=1000, eps=eps):
            X0 = np.array([x2, y2])
            grad_f = grad(f)
            Grad0 = grad_f(x2, y2)
            norme_grad_f = np.sqrt(Grad0[0]**2 + Grad0[1]**2)
            if not(norme_grad_f): 
                raise ValueError(f"Gradient nul au point {X0} !")
            M_quart_tour_droit = np.array([[0, 1],
                                        [-1, 0]])
            X = X0 + delta1*np.dot(M_quart_tour_droit, Grad0)/norme_grad_f # Calcul du point de départ de la méthode de Newton
            def F(x,y):
                return np.array([f(x,y)-c, (x-x2)**2 + (y-y2)**2 - delta1**2])
            X1 = Newton(F, X[0], X[1], eps, N)
            return np.array([X1[0], X[1]])

    tab_points = np.empty(shape=(2, N))
    tab_points[0][0], tab_points[1][0] = x0, y0

    for i in range(0,N-1):
        x, y = tab_points[0][i], tab_points[1][i]
        x1, y1 = level_curve_1step(f, x, y)[0], level_curve_1step(f, x, y)[1]
        tab_points[0][i+1], tab_points[1][i+1] = x1, y1 
    return tab_points

def level_curve2(f, x0, y0, delta=0.3, N=1000, eps=0.001):
    X = np.empty(shape = (2, N))
    X[0][0] = x0
    X[1][0] = y0
    P0 = [x0,y0]
    c = f(x0, y0)
    stop = False
    i = 0
    def g(x1, y1):          # on veut une fonction de R^2 dans R^2
        return np.array([f(x1, y1) - c, (x1 - x0)**2 + (y1 - y0)**2 - delta**2])
    while not(stop) and i <(N-1):
        gradF = grad(f)(x0, y0)
        grad_rot = np.array([gradF[1], -gradF[0]]) # rotation de 90° vers la droite du gradient
        point_int = grad_rot * delta/np.linalg.norm(grad_rot) 
        point_depart = np.array([point_int[0]+x0, point_int[1]+y0])  # point de départ pour Newton
        tab = Newton(g, point_depart[0], point_depart[1], eps, N)
        x0, y0 = tab[0], tab[1]
        X[0][i+1] = x0
        X[1][i+1] = y0
        if i == 0 :
            P1 = [x0,y0]
            S0 = [P0,P1]
            S = S0         # on enregistre le 1 er segment
        else :
            Pint = S[1]
            P = [x0,y0]
            S = [Pint,P] 
            if intersection(S0,S) :
                print('on a intersecté le 1er segment à i =',i)
                stop = True # on arrête le programme si il y a une intersection
        i +=1
    return X,stop

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

def level_curve3(f, x0, y0, oversampling = 20, delta=0.01, N=350, eps=0.001):
    Tab_0 = level_curve(f, x0, y0)
    grad_f = grad(f)
    if oversampling == 1:
        return Tab_0
    X = np.zeros(shape = (2, (N-1)*oversampling + 1))
 
    for i in range(N):
        X[0][i*oversampling] = Tab_0[0][i]
        X[1][i*oversampling] = Tab_0[1][i]    
        if i < N-1:
            P1 = np.array([Tab_0[0][i], Tab_0[1][i]])
            P2 = np.array([Tab_0[0][i+1], Tab_0[1][i+1]])
            #print(f"P1 = {P1}, P2 = {P2}")
            GradP1 = grad_f(P1[0], P1[1])
            GradP2 = grad_f(P2[0], P2[1])
            u1 = np.array([GradP1[1], -GradP1[0]])
            u2 = np.array([GradP2[1], -GradP2[0]])
            Tab_t = np.linspace(0, 1, oversampling)
            for j, t in enumerate(Tab_t):
                X_interpol = gamma(t, P1, P2, u1, u2)
                X[0][i*oversampling + j] = X_interpol[0]
                X[1][i*oversampling + j] = X_interpol[1]
    return X

X0 = Newton(F1, 0.8, 0.8, 0.001, 100)
x0, y0 = X0[0], X0[1] 

Tab_val2 = level_curve(f1, x0, y0)
Tab_val3 = level_curve3(f1, x0, y0)

plt.plot(Tab_val2[0], Tab_val2[1], 'r--')
plt.plot(Tab_val3[0], Tab_val3[1], 'g--')
plt.show()




                





# test avec N = 5 et oversampling = 3
Tab_0 = [1,2,3,4,5]
X = np.zeros(4*3+1)
for i in range(5):
    X[i*3] = Tab_0[i]

print(Tab_0)
print(X)



