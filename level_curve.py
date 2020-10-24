import autograd
import autograd.numpy as np
from autograd import grad


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

def Newton(F, x0, y0, eps, N):
    X0 = np.array([x0, y0])
    J_F = J(F)
    for i in range(N):
        J_F_inv = np.linalg.inv(J_F(X0[0], X0[1])) # calcul de l'inverse de la jacobienne en x0, y0
        X = X0 - np.dot(J_F_inv,F(X0[0], X0[1])) # calcul du nouveau point X =(x, y)
        x, y = X[0], X[1]
        if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return (x, y)
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")

M_quart_tour_droit = np.array([[0, 1],
                               [-1, 0]])

def level_curve(f, x0, y0, delta=0.1, N=1000, eps=0.001):
    X0 = np.array([x0, y0])
    grad_f = grad(f)
    Grad0 = grad_f(x0, y0)
    norme_grad_f = np.sqrt(Grad0[0]**2 + Grad0[1]**2)
    X = X0 + delta*np.dot(M_quart_tour_droit, Grad0)/norme_grad_f # Calcul du point de départ de la méthode de Newton
    
    def F(x,y):
        return np.array([f(x,y), x**2 + y**2 - delta**2])

    X1 = Newton(F, X[0], X[1], eps, N)
    return (X1[0], X[1])



def f1(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return 3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2

def F1(x,y):
    return np.array([f1(x,y)-0.8, x-y])


X0 = Newton(F1, 0.8, 0.8, 0.0001, 100)
x0, y0 = X0[0], X0[1]
X1 = level_curve(f1, x0, y0, delta=0.1, N=1000, eps=0.001)
x1, y1 = X1[0], X1[1]
dist = np.sqrt((x0-x1)**2 + (y0-y1)**2)
print(X0, X1, '\n')
print(f'dist = {dist}')
