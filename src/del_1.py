import numpy as np
from mayavi import mlab

def grid(Lx, Ly, Nx, Ny):
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    return np.meshgrid(x, y, indexing = "ij")

def initial_function(X, Y, hD, hS, R):
    h0 = np.where(np.sqrt(X**2 + Y**2) < R, hD, hS)
    u0 = np.zeros_like(h0)
    v0 = np.zeros_like(h0)
    return h0, u0, v0

def plot(X, Y, h):
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0,0,0))
    s = mlab.surf(X, Y, h, colormap="Blues", warp_scale=1)
    mlab.axes(s, xlabel="x [m]", ylabel="y [m]", zlabel="h [m]")
    mlab.colorbar(s, title='h[m]')
    mlab.show()

if __name__ == '__main__':
    X, Y = grid(Lx=100, Ly=100, Nx=201, Ny=201)
    h0, u0, v0 = initial_function(X, Y, hD=10, hS=1, R=10)
    plot(X, Y, h0)

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
    
    FQ= F(Q)
    GQ =G(Q)
    Qn = Q.copy()
    Qn[:,1:-1, 1:-1] = (0.25 * (Q[:, 2:, 1:-1] + Q[:, :-2, 1:-1] + Q[:, 1:-1, 2:] + Q[:, 1:-1, :-2])
        - dt/(2*dx) * (FQ[:, 2:, 1:-1] - FQ[:, :-2, 1:-1])
        - dt/(2*dy) * (GQ[:, 1:-1, 2:] - GQ[:, 1:-1, :-2])
    )
    # TODO: randbetingelser for Qn[:, 0, :], Qn[:, -1, :], Qn[:, :, 0], Qn[:, :, -1]
    return Qn

