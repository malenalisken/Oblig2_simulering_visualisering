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
    """
    one lax-Freidrich step, inner nodes.
    """
    
    FQ= F(Q)
    GQ =G(Q)
    Qn = Q.copy()
    Qn[:,1:-1, 1:-1] = (0.25 * (Q[:, 2:, 1:-1] + Q[:, :-2, 1:-1] + Q[:, 1:-1, 2:] + Q[:, 1:-1, :-2])
        - dt/(2*dx) * (FQ[:, 2:, 1:-1] - FQ[:, :-2, 1:-1])
        - dt/(2*dy) * (GQ[:, 1:-1, 2:] - GQ[:, 1:-1, :-2])
    )
    
    return Qn

def lf_step(Q, dt, dx, dy):
    """
    one Lax–Friedrichs-step with periodiske BC (via np.roll).
    """
    FQ = F(Q)
    GQ = G(Q)
    ip = lambda A: np.roll(A, -1, axis=1)   # i+1
    im = lambda A: np.roll(A,  1, axis=1)   # i-1
    jp = lambda A: np.roll(A, -1, axis=2)   # j+1
    jm = lambda A: np.roll(A,  1, axis=2)   # j-1
    Qn =(0.25 * (ip(Q) + im(Q) + jp(Q) + jm(Q))
            - dt/(2*dx) * (ip(FQ) - im(FQ))
            - dt/(2*dy) * (jp(GQ) - jm(GQ)))
    return Qn

def lax_f_simulation(Q0, dx, dy, dt,T):
    """
    simulates hight of water using las friedrich using set dt. 
    retursn height of water
    """
    #TODO look into if we need to  return Q values 
    #TODO Look into if dt should be dynamic 
    Q = Q0.copy() # Q is updating to new value for every timestep
    n = int(T/dt) # number of inner nodes
    hs = [Q[0].copy()] #hs will be array of hight of water in grid, with appendend values
    n = int(T/dt)
    for i in range (1, n+2):
        Q = lf_step(Q,dt,dx,dy)
        hs.append(Q[0].copy())
    return hs
    

    

if __name__ == '__main__':
    X, Y = grid(Lx=100, Ly=100, Nx=201, Ny=201)
    h0, u0, v0 = initial_function(X, Y, hD=10, hS=1, R=10)
    Q0 = np.array([h0, u0, v0])
    dx, dy = 0.25, 0.25
    dt = 0.4 * dx / np.sqrt(g * 10)
    lax_f_simulation(Q0, dx, dy, dt, T =1 )

    #plot(X, Y, h0)



