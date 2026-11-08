'''
MAMO3200 Oblig 2 part 1: shallow water equation

Simulates the 2D shallow water equation with Lax-Friedrich scheme, 
adaptive time step and different boundary conditions

Animations are saved in figures/animation_SW

Contents:
1a) initial condition
1b) simulate with provided initial values and periodic boundary conditions
1c) test of volume conservation, effect of grid refinement and constant intial cond
1d) animation of water height h and quiver plot of velocity with periodic BC
1e) animation with dirichlet, neumann, slip and no_slip BC
1f) large scale simulation 
'''

import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
import imageio_ffmpeg
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

#constant
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

def initial_Q_large_scale(X, Y, Lx, Ly,H = 100, A =1, sigma = 50_000):
    x_c = Lx/2.7
    y_c = Ly/4
    
    h0 = A*np.exp(-((X-x_c)**2 + (Y-y_c)**2)/(2*sigma**2)) + H
    hu0 = np.zeros_like(h0)
    hv0 = np.zeros_like(h0)
    return np.array([h0, hu0, hv0])


def style_surface(s, gamma = 0.7):
    """
    Function for custom styles for plots and animation
    of shallow water simulation
    gamma : bends the color gradient, gamma < 1 gives more color change
            for low h values (gamma = 1 is a straight gradient)
    """
    # custom colormap: light blue (low h) -> dark blue (high h)
    light = np.array([173, 216, 230])
    dark = np.array([8, 48, 107])
    t_lut = np.linspace(0, 1, 256)[:, None] ** gamma
    rgb = (1 - t_lut) * light + t_lut * dark
    lut = np.hstack([rgb, 255 * np.ones((256, 1))]).astype(np.uint8)
    s.module_manager.scalar_lut_manager.lut.table = lut

    # softer lighting so the surface colors match the colorbar in the saved png
    s.actor.property.ambient = 0.6
    s.actor.property.diffuse = 0.5
    s.actor.property.specular = 0.0


def plot_instance(X, Y, h, timestamp, filename = None):
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0,0,0), size = (500, 300))

    s = mlab.surf(X, Y, h, colormap="Blues", warp_scale=5)
    style_surface(s)
    

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
    
    if filename is not None:
        mlab.savefig(filename)

def s_w_animation(X, Y, ts, hs, delay=50, frame_dir = None, custom_style = True, colormap = "Blues", 
                  warp_scale = 1, time_unit= "s", time_scale =1, length_unit = "m", length_scale = 1, 
                  vmin = None,vmax = None, cbar_title="h [m]", label = None):
    """
    animates water height h over time

    ts: array of time points,
    hs: array of h-grids, shape (len(ts), Nx, Ny)
    frame_dir : if not none, each frame is saved in frame_dir_folder
    label : text shown in upper left corner (describing boundary cond)
    """

    #same style settings as 3d plot
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0,0,0), size=(1280, 720))
    vmin = hs.min() if vmin is None else vmin
    vmax = hs.max() if vmax is None else vmax
    
    s = mlab.surf(X, Y, hs[0], colormap=colormap, warp_scale= warp_scale,
              vmin=vmin, vmax=vmax)
    if custom_style: style_surface(s)

    ax = mlab.axes(s, xlabel=f"x [{length_unit}]", ylabel=f"y [{length_unit}]", zlabel="", nb_labels=3,
                   ranges = [X.min()/length_scale, X.max()/length_scale,
                           Y.min()/length_scale, Y.max()/length_scale,
                           hs.min(), hs.max()])
    ax.axes.y_axis_visibility = False               #hide vertical axis and show hight with colorbar
    ax.axes.label_format = "%.0f"
    ax.axes.corner_offset = 0.05        #so ax labels do not collide
    
    cb = mlab.colorbar(s, title=cbar_title, orientation = 'vertical')
    cb.scalar_bar_representation.position = [0.85, 0.15]
    cb.scalar_bar_representation.position2 = [0.08, 0.7]
    
    mlab.view(azimuth=45, elevation=60, focalpoint=(0, 0, 3))
    dist= mlab.view()[2]         # using mayavi auto sistance 
    mlab.view(distance=1.3 * dist)          # zooms out 30 %
   
    title = mlab.title(f"t = {ts[0]/time_scale:.2f} {time_unit}", size=0.3, height=0.9)
    if frame_dir is not None:
        os.makedirs(frame_dir, exist_ok=True)
    if label is not None:
        txt = mlab.text(0.02, 0.92, label, width=0.3, color=(0, 0, 0))
        txt.property.font_family = 'times'


    @mlab.animate(delay=delay)
    def anim():
        first_loop = True                   #needed to stop fig save after after first loop
        while True:                         # loop until window is closed
            for i, (t, h) in enumerate(zip(ts, hs)):
                s.mlab_source.scalars = h   # update height, X and Y stay the same
                title.text = f"t = {t/time_scale:.2f} {time_unit}"
                if frame_dir is not None and first_loop:
                    mlab.savefig(os.path.join(frame_dir, f"frame_{i:04d}.png"), figure = fig)
                yield                       # draw this frame
            first_loop = False              # stops save fig


    return anim()

def quiver_animation(X, Y, ts, us, vs, step =20, interval = 50, filename = None, fps = 10, bitrate = 200, time_unit = "s", time_scale = 1, length_unit = "m", length_scale = 1, vmax = None):
    """
    2d quiver animation of velocity u and v
    step : determines wivh nth step vil be plotted with arrow
    interval: how long in ms between new frames
    """
    Xs, Ys= X[::step,::step]/length_scale, Y[::step, ::step]/length_scale
    speeds = np.sqrt(us**2 + vs**2)
    vmax = speeds.max() if vmax is None else vmax
    fig, ax = plt.subplots(figsize = (7,6))
    q = ax.quiver(Xs, Ys, us[0][::step,::step], vs[0][::step, ::step],
                  speeds[0][::step, ::step], cmap="viridis", clim=(0, vmax),
                  scale=vmax * 6, scale_units="width", width = 0.005)
    fig.colorbar(q, ax=ax, label=r"speed $\sqrt{u^2 + v^2}$ [m/s]")
    ax.set_xlabel(f"x [{length_unit}]")
    ax.set_ylabel(f"y [{length_unit}]")
    ax.set_aspect("equal")
    title = ax.set_title(f"t = {ts[0]/time_scale:.2f} {time_unit}")

    def update(k):
        q.set_UVC(us[k][::step, ::step], vs[k][::step, ::step], speeds[k][::step, ::step])
        title.set_text(f"t = {ts[k]/time_scale:.2f} {time_unit}")
        return q, title

    anim = FuncAnimation(fig, update, frames=len(ts), interval=interval)

     
    if filename is not None:
        anim.save(filename, writer=FFMpegWriter(fps=fps, bitrate=bitrate))
        print(f"Saved {filename}: {os.path.getsize(filename) / 1e6:.1f} MB")

    
    return fig, anim

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
    one Lax–Friedrichs-step with periodic BC (via np.roll).
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

def dirichlet_bc(Q):
    '''Sets the momentum and egde height equal to 1'''
    Q[:, [0, -1], :] = 1        # h, hu and hv on the left and right edges
    Q[:, :, [0, -1]] = 1       # h, hu and hv on the bottom and top edges

    return Q

def neumann_bc(Q):
    '''Sets momentum and height for boundry nodes equal to the neighbouring inner node'''
    Q[:, [0, -1], :] = Q[:, [1, -2], :] 
    Q[:, :, [0, -1]] = Q[:, :, [1, -2]]
    return Q

def slip_cond_bc(Q, dt, dx, dy, no_slip = False):
    '''Fictionless wall (slip): ghost cells mirror h, 
        flipped sign of the normal velocity, tangent vel stays the same if no_slip false
        if noslip true the tangent velcoity is also flipped
    '''
    Qp = np.pad(Q, ((0, 0), (1, 1), (1, 1)), mode="edge")       #filling ghost nodes with nearest real values
    Qp[1, [0, -1], :] *= -1
    Qp[2, :, [0, -1]] *= -1
    if no_slip:                    #change coeff for tangial velocity too
        Qp[2, [0, -1], :] *= -1     
        Qp[1, :, [0, -1]] *= -1
    Qp = lf_step(Qp, dt, dx, dy)
    return Qp[:, 1: -1, 1: -1]


def stable_timestep(Q, dx, dy, safety = 0.95):
    h, hu, hv = Q
    u, v = hu/h, hv/h
    c = np.sqrt(g*h)
    dt_x = dx / (2 * np.max(np.abs(u) + c))
    dt_y = dy / (2 * np.max(np.abs(v) + c))
    return safety * min(dt_x, dt_y)


def lax_f_simulation(Q0, dx, dy, T, save_times,  safety = 0.95, return_velocity = False, 
                     boundry_condition = None, ghost_cells = None, no_slip = False):
    """
    Simulates height of water using lax friedrich with dynamic dt.
    Returns arrays of timesteps and waterheights for grid, and velocity if return_velocity = True
    """

    Q = Q0         
    t = 0.0

    ts, hs, us, vs = [], [], [], []

    def save(t, Q):
        ts.append(t)
        hs.append(Q[0].copy()) 
        if return_velocity:
            us.append(Q[1] / Q[0])
            vs.append(Q[2] / Q[0])

    save_times = sorted(save_times)
    next_save = 0               #index of next save
    if save_times[next_save] == 0:
        next_save += 1
        save(0.0, Q0)  

    while t < T:
        dt = stable_timestep(Q, dx, dy, safety = safety)
        dt = min(dt, T-t)            #so it doesnt step past T
        dt = min(dt, save_times[next_save] - t)      #not past wanted timestamp

        if ghost_cells is None:
            Q = lf_step(Q, dt, dx, dy)
        else: 
            Q = slip_cond_bc(Q, dt, dx, dy, no_slip)

        if boundry_condition is not None:
            Q = boundry_condition(Q)
        t += dt

        if np.isclose(t, save_times[next_save]):
            t = save_times[next_save]
            save(t, Q)
            
            next_save += 1
    if return_velocity:
        return np.array(ts), np.array(hs), np.array(us), np.array(vs)
    return np.array(ts), np.array(hs)


def finer_grid_check(Lx, Ly, hD, hS, R):
    '''plots graph of cross section for varying distance between nodes in grid'''
    for N in [201, 401, 801]:
        X,Y = grid(Lx, Ly, N, N)
        dx, dy = [Lx/(N-1), Ly/(N-1)]
        Q0= initial_Q(X, Y, hD, hS, R)

        _, hs = lax_f_simulation(Q0, dx, dy, T = 5, save_times= [5])
        h_end = hs[0]

        j = np.argmin(np.abs(Y[0, :])) 
        x_line = X[:, j]
        h_line = h_end[:, j]

        plt.plot(x_line, h_line, label=f'N = {N}')

    plt.xlabel('x [m]')
    plt.ylabel('h [m]')
    plt.title('Cross-section at y = 0, t = 5 s with changing grid')
    plt.grid(alpha = 0.3)
    plt.legend() 
        

def main():
    # 1a) visualization of initial condition

    Lx, Ly , Nx, Ny = [100, 100, 401, 401]
    hD, hS, R = [10, 1, 10]
    X, Y = grid(Lx, Ly, Nx, Ny)
    Q0 = initial_Q(X, Y, hD, hS, R)
    
    plot_instance(X, Y, Q0[0], timestamp= 0)
    mlab.show()

    # 1b) simulation of system with suggested initial conditions

    dx, dy = [Lx/(Nx-1), Ly/(Ny-1)]
    save_times = np.linspace(0,20,4)
    T = save_times[-1]
    ts, hs = lax_f_simulation(Q0, dx, dy, T, save_times = save_times)
    

    # 1c) periodic boundry conditions should maintain total water volume
    
    V = np.array([np.sum(h) * dx * dy for h in hs])
    print('Volumn :', V)
    print('Relative change in volume:', (V - V[0]) / V[0], '\n \n')
    
    # check that finer grid does not improve solution significantly
    
    finer_grid_check(Lx, Ly, hD, hS, R)
    plt.show()
    
    # constant initial conditions should preserve waterdepth over time
    h0 = np.ones_like(X)
    u0 = np.ones_like(X)
    v0 = np.zeros_like(X)
    Q0 = np.array([h0, u0, v0])
    ts, hs = lax_f_simulation(Q0, dx, dy, T, save_times= save_times)
    print('The absolute maximum diviation of water depth between t0 and T \n for constant initial conditions is :', np.max(np.abs(hs[-1] - h0)))


    # 1d) animations and 1e) animations with different boundry conditions

    Q0 = initial_Q(X, Y, hD, hS, R)

    anim_times = np.linspace(0, 20, 301)

    # animation of shallow water with periodic boundry conditions
    
    ts_anim, hs_anim = lax_f_simulation(Q0, dx, dy,anim_times[-1], save_times=anim_times)
    periodic_anim = s_w_animation(X, Y, ts_anim, hs_anim, delay = 10, warp_scale = 1, 
                                  label = 'Periodic boundary conditions' )
    mlab.show()
    
    #animation of shallow water with dirichlet boundry condition
    
    ts_anim, hs_anim = lax_f_simulation(Q0, dx, dy, anim_times[-1], save_times=anim_times, boundry_condition=dirichlet_bc)
    dirichlet_anim = s_w_animation(X, Y, ts_anim, hs_anim, delay= 10, warp_scale = 1, 
                                   label = 'Dirichlet boundary conditions')
    mlab.show()
    
    #animation of shallow water with neumann boundry condition
    
    ts_anim, hs_anim = lax_f_simulation(Q0, dx, dy, anim_times[-1], save_times=anim_times, boundry_condition=neumann_bc)
    neumann_anim = s_w_animation(X, Y, ts_anim, hs_anim, delay= 10, warp_scale=1, 
                                 label = 'Neumann boundary condition')
    mlab.show()

    #animation of shallow water with slip wall boundry condition

    ts_anim, hs_anim = lax_f_simulation(Q0, dx, dy, anim_times[-1], save_times=anim_times, ghost_cells=slip_cond_bc)
    slip_anim = s_w_animation(X, Y, ts_anim, hs_anim, delay= 10, warp_scale=1, 
                              label = 'Slip boundary condition')
    mlab.show()

    #animation of shallow water with no-slip wall boundry condition

    ts_anim, hs_anim = lax_f_simulation(Q0, dx, dy, anim_times[-1], save_times=anim_times, ghost_cells=slip_cond_bc, no_slip= True)
    no_slip_anim = s_w_animation(X, Y, ts_anim, hs_anim, delay= 10, warp_scale = 1, 
                                 label = 'No-slip boundary condition')
    mlab.show()

    #animation of quiverplot

    quiver_times = np.linspace(0, 20, 201)
    ts_q, hs_q, us_q, vs_q = lax_f_simulation(Q0, dx, dy, quiver_times[-1], save_times=quiver_times, return_velocity= True)
    fig_q, a_q = quiver_animation(X, Y, ts_q, us_q, vs_q)
    plt.show()
    
    # 1 f) simulation on larger scale with new initialfunc and new lenghts
    
    Lx, Ly, Nx, Ny = [3_000_000, 3_000_000, 401, 401]
    dx, dy = [Lx/(Nx-1), Ly/(Ny-1)]
    
    X, Y = grid(Lx, Ly, Nx, Ny)
    H = 100 
    Q0 = initial_Q_large_scale(X,Y, Lx,Ly, H = H)
    T = 30 * 3600 #30 hours
       
    anim_times = np.linspace(0, T, 301)
    ts_anim, hs_anim = lax_f_simulation(Q0, dx, dy, anim_times[-1], save_times=anim_times, boundry_condition=neumann_bc)
    eta_anim = hs_anim - H
    large_scale_anim = s_w_animation(X, Y, ts_anim, eta_anim, delay= 10, custom_style=False, colormap="seismic",
                       warp_scale =1, time_unit = "h",time_scale=3600, length_unit= "km", length_scale= 1000, 
                       vmin= -0.15, vmax =0.15, cbar_title= "eta[m]", label = 'Large scale animation')
    mlab.show()

    #quiver animation

    quiver_times = np.linspace(0, T, 101)
    ts_q, hs_q, us_q, vs_q = lax_f_simulation(Q0, dx, dy, quiver_times[-1], save_times=quiver_times, return_velocity= True, boundry_condition=neumann_bc)
    fig_q, a_q = quiver_animation(X, Y, ts_q, us_q, vs_q, step = 12, filename= None, time_unit="h", time_scale= 3600, length_unit="km", length_scale= 1000, vmax = 0.08)
    
    plt.show()



if __name__ == '__main__':
    main()

    