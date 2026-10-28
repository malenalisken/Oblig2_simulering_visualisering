import numpy as np

g = 9.81


def F(Q):
    h, hu, hv = Q
    u = hu/h
    v = hv/v
    F = np.zeros(len(Q))
    F[0] = h*u
    F[1] = h*u**2 + 0.5 * g * h**2
    F[2] = h*u*v
    return F

def G(Q):
    h, hu, hv = Q
    u = hu/h
    v = hv/v
    G = np.zeros(len(Q))
    G[0] = h*v
    G[1] = h*u*v
    G[2] = h*v**2 + 0.5 * g * h**2
    return G

def lax_friedrichs(Q0,  ):

    Q = np.ndarray(Q0.size, n+2)
    F = np.zeros_like(Q)
    G= np. zeros_like(Q)
    Q[:,0] = Q0
    F[:,0] = F(Q0)
    G[:,0] = G(Q0)

    for i in range(1, n+2):
        Q[:,i] = 