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

def plot(X, Y, h, filename = None):
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0,0,0), size = (500, 300))

    s = mlab.surf(X, Y, h, colormap="Blues", warp_scale=1)

    ax = mlab.axes(s, xlabel="x [m]", ylabel="y [m]", zlabel="", nb_labels = 3)
    ax.axes.y_axis_visibility = False
    ax.axes.label_format = "%.0f"
    ax.axes.corner_offset = 0.05

    cb = mlab.colorbar(s, title='h[m]', orientation = 'vertical')
    cb.scalar_bar_representation.position = [0.85, 0.15]
    cb.scalar_bar_representation.position2 = [0.08, 0.7]

    mlab.view(azimuth=45, elevation= 60, distance=220, focalpoint=(0, 0, 3))
    t = mlab.title('$h(x,y)$ at initial condition $h_0$', size=0.3, height=0.8)

    font = 'times'
    for prop in (ax.title_text_property, ax.label_text_property,
                 cb.title_text_property, cb.label_text_property,
                 t.property):
        prop.font_family = font
    
    if filename != None:
        mlab.savefig(filename)
    mlab.show()



if __name__ == '__main__':
    X, Y = grid(Lx=100, Ly=100, Nx=401, Ny=401)
    h0, u0, v0 = initial_function(X, Y, hD=10, hS=1, R=10)
    plot(X, Y, h0)