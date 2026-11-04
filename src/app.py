import panel as pn
import matplotlib.pyplot as plt
from del_2 import initial_state
from constants import *

pn.extension()

button = pn.widgets.Button(
    name = 'Reset',
    button_type = 'primary'
)

position, state, *_ = initial_state(N = N, I0=I0, R0 = R0, Lx = Lx, Ly = Ly)

fig, ax = plt.subplot(figsize = (8,8))
ax.plot(position[:, 0], position[:, 1])
plot = pn.pane.Matplotlib(fig)
                   
app = pn.Column(
    '# SIR model simulation',
    'Position of agents',
    'Sum of cases',
    button, 
    plot
)
app.servable()