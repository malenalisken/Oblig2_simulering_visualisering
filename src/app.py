import panel as pn
import numpy as np
import matplotlib.pyplot as plt

from constants import blue, red, gray
from del_2 import Simulation

pn.extension()

button = pn.widgets.Button(
    name = 'Reset',
    button_type = 'primary'
)


sim = Simulation()

colors = np.array([blue, red, gray])
agent_colors = colors[sim.state]

fig, ax = plt.subplots(figsize = (3,3))
ax.scatter(sim.pos[:,0], sim.pos[:,1], color = agent_colors, s = 0.7)

                   
app = pn.Column(
    '# SIR model simulation',
    'Position of agents',
    'Sum of cases',
    pn.pane.Matplotlib(fig),
    button
)
app.servable()