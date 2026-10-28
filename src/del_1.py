import numpy as np

g = 9.81


def F(Q):
    h, hu, hv = Q
    u = hu/h
    v = hv/h
    return np.array([h*u, h*u**2 + 0.5*g*h**2, h*u*v])


def G(Q):
    h, hu, hv = Q
    u = hu/h
    v = hv/h
    return np.array([h*v, h*u*v, h*v**2 + 0.5 * g * h**2])

def lf_step(Q, dt, dx, dy ):
    FQ = np.zeros_like(Q)
    GQ= np. zeros_like(Q)
    F[:,0] = F(Q)
    G[:,0] = G(Q)
    Qn = Q.copy()
    Qn[:,1:-1, 1:-1] = (0.25 * (Q[:, 2:, 1:-1] + Q[:, :-2, 1:-1] + Q[:, 1:-1, 2:] + Q[:, 1:-1, :-2])
        - dt/(2*dx) * (FQ[:, 2:, 1:-1] - FQ[:, :-2, 1:-1])
        - dt/(2*dy) * (GQ[:, 1:-1, 2:] - GQ[:, 1:-1, :-2])
    )
    # TODO: randbetingelser for Qn[:, 0, :], Qn[:, -1, :], Qn[:, :, 0], Qn[:, :, -1]
    return Qn
