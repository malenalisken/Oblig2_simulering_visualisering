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

def redraw():
    """Draw the current state of sim in the figure."""
    scat.set_offsets(sim.pos)
    scat.set_color(colors[sim.state])
    ax.set_title(f"t = {sim.t:.1f}", size=8)
    position_plot.param.trigger("object")


def update_postion():
    if sim.counts()[1] == 0:            # outbreak over
        callback.stop()
        return
    sim.step()
    redraw()


def start(event):
    callback.start()


def pause(event):
    callback.stop()


def reset(event):
    global sim
    callback.stop()
    sim = Simulation(seed=None)         # new random outbreak each time
    redraw()

callback = pn.state.add_periodic_callback(update_postion, period=50, start = False)


legend = pn.pane.HTML(
    f'<span style="color:{blue}">●</span> Susceptible &nbsp;&nbsp; '
    f'<span style="color:{red}">●</span> Infected &nbsp;&nbsp; '
    f'<span style="color:{gray}">●</span> Recovered',
    styles={"border": "1px solid #ccc", "padding": "4px 10px", "border-radius": "4px", "font-size": "15px"},
    align = 'center'
)
start_button = pn.widgets.Button(
    name = 'Start',
    button_type = 'success',
    styles = { 'width' : '120px','height': '40px', 'opacity' : '0.8'},
    align = 'start'
)
pause_button = pn.widgets.Button(
    name = 'Pause',
    button_type = 'primary',
    styles = { 'width' : '120px','height': '40px', 'opacity' : '0.8'},
    align = 'center'
)
reset_button = pn.widgets.Button(
    name = 'Reset',
    button_type = 'danger',
    styles = { 'width' : '120px','height': '40px', 'opacity' : '0.8'},
    align = 'end'
)

start_button.on_click(start)
pause_button.on_click(pause)
reset_button.on_click(reset)

position_plot_buttons = pn.Row(
    start_button,
    pause_button, 
    reset_button,
    align = 'center'
)

plot_with_legend = pn.Column(
    position_plot,
    legend,
    position_plot_buttons
)


                
app = pn.Column(
    '# SIR model simulation',
    plot_with_legend
)
app.servable()