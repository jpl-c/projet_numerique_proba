import autograd
import autograd.numpy as np
from autograd import grad


def J(f):
    j = autograd.jacobian
    def J_f(x, y):
        return np.array([j(f, 0)(x, y), j(f, 1)(x, y)]).T
    return J_f



def f1(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return 3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2

def F1(x,y):
    return np.array([f1(x,y)-0.8, x-y])

def F_test(x, y):
    return np.array([x+y, x-y])






def Newton(F, x0, y0, eps, N):
    X0 = np.array([x0, y0])
    J_F = J(F)
    for i in range(N):
        print ('n = ', i)
        print(x0, y0)
        J_F_inv = np.linalg.inv(J_F(X0[0], X0[1])) # calcul de l'inverse de la jacobienne en x0, y0
        X = X0 - np.dot(J_F_inv,F(X0[0], X0[1])) # calcul du nouveau point X =(x, y)
        x, y = X[0], X[1]
        print(f"(x, y) = {(x,y)}")
        if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            print(f"x0 = {x0}, y0 = {y0}, x  = {x}, y = {y}" )
            print(np.sqrt((x - x0)**2 + (y - y0)**2))
            return (x, y)
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")


X_sol = Newton(F_test, 0.8, 0.8, 0.0001, 100)
x1, x2 = X_sol[0], X_sol[1]
print(X_sol, F_test(x1, x2))






