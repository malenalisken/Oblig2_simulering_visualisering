import panel as pn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import blue, red, gray
from del_2 import Simulation
pn.extension()


sim = Simulation()

colors = np.array([blue, red, gray])
agent_colors = colors[sim.state]

fig, ax = plt.subplots(figsize = (3,3))
scat = ax.scatter(sim.pos[:,0], sim.pos[:,1], color = agent_colors, s = 8)
ax.set_ylim(0, sim.Ly)
ax.set_xlim(0, sim.Lx)
ax.set_xticks([])     
ax.set_yticks([])       
ax.set_aspect('equal')
ax.set_title(f"t = {sim.t:.1f} (days)", size = 8)
fig.tight_layout()

position_plot = pn.pane.Matplotlib(fig)
plt.close(fig)

def update_postion():
    if sim.counts()[1] == 0:            #when nobody can transition to another state
        callback.stop()
        return
    sim.step()
    agent_colors = colors[sim.state]
    scat.set_offsets(sim.pos)
    scat.set_color(agent_colors)
    ax.set_title(f"t = {sim.t:.1f} (days)", size = 8)
    position_plot.param.trigger("object")

callback = pn.state.add_periodic_callback(update_postion, period=50)

legend = pn.pane.HTML(
    f'<span style="color:{blue}">●</span> Susceptible &nbsp;&nbsp; '
    f'<span style="color:{red}">●</span> Infected &nbsp;&nbsp; '
    f'<span style="color:{gray}">●</span> Recovered',
    styles={"border": "1px solid #ccc", "padding": "4px 10px", "border-radius": "4px", "font-size": "15px"},
    align = 'center'
)

plot_with_legend = pn.Column(
    position_plot,
    legend,
)

button = pn.widgets.Button(
    name = 'Reset',
    button_type = 'primary'
)
                
app = pn.Column(
    '# SIR model simulation',
    plot_with_legend,
    button
)
app.servable()