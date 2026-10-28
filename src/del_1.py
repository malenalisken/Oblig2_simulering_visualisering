import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt

g = 9.81


def grid(Lx, Ly, Nx, Ny):
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    return np.meshgrid(x, y, indexing = "ij")

def initial_Q(X, Y, hD, hS, R):
    h0 = np.where(np.sqrt(X**2 + Y**2) < R, hD, hS)
    u0 = np.zeros_like(h0)
    v0 = np.zeros_like(h0)
    return np.array([h0, u0, v0])

def plot_instance(X, Y, h, timestamp, filename = None):
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0,0,0), size = (500, 300))

    s = mlab.surf(X, Y, h, colormap="Blues", warp_scale=5)

    # custom colormap: light blue (low h) -> dark blue (high h)
    light = np.array([173, 216, 230])
    dark = np.array([8, 48, 107])
    t_lut = np.linspace(0, 1, 256)[:, None]
    rgb = (1 - t_lut) * light + t_lut * dark
    lut = np.hstack([rgb, 255 * np.ones((256, 1))]).astype(np.uint8)
    s.module_manager.scalar_lut_manager.lut.table = lut

    # softer lighting so the surface colors match the colorbar in the saved png
    s.actor.property.ambient = 0.6
    s.actor.property.diffuse = 0.5
    s.actor.property.specular = 0.0

    ax = mlab.axes(s, xlabel="x [m]", ylabel="y [m]", zlabel="", nb_labels = 3)
    ax.axes.y_axis_visibility = False
    ax.axes.label_format = "%.0f"
    ax.axes.corner_offset = 0.05

    cb = mlab.colorbar(s, title='h[m]', orientation = 'vertical')
    cb.scalar_bar_representation.position = [0.85, 0.15]
    cb.scalar_bar_representation.position2 = [0.08, 0.7]

    mlab.view(azimuth=45, elevation= 60, distance=220, focalpoint=(0, 0, 3))
    t = mlab.title(f'System at timepoint t = {timestamp}', size=0.3, height=0.9)

    font = 'times'
    for prop in (ax.title_text_property, ax.label_text_property,
                 cb.title_text_property, cb.label_text_property,
                 t.property):
        prop.font_family = font
    
    if filename != None:
        mlab.savefig(filename)



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

def lf_step(Q, dt, dx, dy):
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

def stable_timestep(Q, dx, dy, safety = 0.95):
    h, hu, hv = Q
    u, v = hu/h, hv/h
    c = np.sqrt(g*h)
    dt_x = dx / (2 * np.max(np.abs(u) + c))
    dt_y = dy / (2 * np.max(np.abs(v) + c))
    return safety * min(dt_x, dt_y)


def lax_f_simulation(Q0, dx, dy, T, save_times, safety = 0.95):
    """
    Simulates height of water using las friedrich with dynamic dt.
    Returns arrays of timesteps and waterheights for grid
    """

    Q = Q0         
    t = 0.0

    ts, hs, = [], []

    save_times = sorted(save_times)
    next_save = 0               #index of next save
    if save_times[next_save] == 0:
        next_save += 1
        ts.append(0.0)
        hs.append(Q0[0].copy())   

    while t < T:
        dt = stable_timestep(Q, dx, dy, safety = safety)
        dt = min(dt, T-t)            #so it doesnt step past T
        dt = min(dt, save_times[next_save] - t)      #not past wanted timestamp

        Q = lf_step(Q, dt, dx, dy)
        t += dt

        if np.isclose(t, save_times[next_save]):
            t = save_times[next_save]
            ts.append(t)
            hs.append(Q[0].copy())
            next_save += 1

    return np.array(ts), np.array(hs)
    

    

if __name__ == '__main__':
    Lx, Ly , Nx, Ny = [100, 100, 401, 401]
    hD, hS, R = [10, 1, 10]
    X, Y = grid(Lx, Ly, Nx, Ny)
    Q0 = initial_Q(X, Y, hD, hS, R)
    #plot_instance(X, Y, h0, timestamp= 0)

    dx, dy = [Lx/(Nx-1), Ly/(Ny-1)]
    save_times = np.linspace(0,20,4)
    T = save_times[-1]
    ts, hs = lax_f_simulation(Q0, dx, dy, T, save_times = save_times)

    #for i in range(len(ts)):
    #    plot_instance(X, Y, hs[i], timestamp= ts[i])
    
    # 1c) periodic boundry conditions should maintain total water volume
    V = np.array([np.sum(h) * dx * dy for h in hs])
    print('Volumn :', V)
    print('Relative change in volume:', (V - V[0]) / V[0])

    # 1c) check that finer grid does not improve solution significantly
    h_grid_variation = []
    for N in [201, 401, 801]:
        X,Y = grid(Lx, Ly, N, N)
        dx, dy = [Lx/(N-1), Ly/(N-1)]
        Q0= initial_Q(X, Y, hD, hS, R)

        ts, hs = lax_f_simulation(Q0, dx, dy, T = 5, save_times= [5])
        h_end = hs[0]
        
        j = np.argmin(np.abs(Y[0, :])) 
        x_line = X[:, j]
        h_line = h_end[:, j]

        plt.plot(x_line, h_line, label=f'N = {N}')

    plt.xlabel('x [m]')
    plt.ylabel('h [m]')
    plt.title('Cross-section at y = 0, t = 5 s')
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()  



    # 1c) constant initial conditions should preserve waterdepth over time
    h0 = np.ones_like(X)
    u0 = np.ones_like(X)
    v0 = np.zeros_like(X)
    Q0 = np.array([h0, u0, v0])
    ts, hs = lax_f_simulation(Q0, dx, dy, T, save_times= save_times)
    print('The absolute maximum diviation between t0 and T is :', np.max(np.abs(hs[-1] - h0)))

    
    #mlab.show()
    

    



