import panel as pn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import blue, red, gray
from del_2 import Simulation

pn.extension()

sim = Simulation(seed = None)

#scatterplot of position of agents
colors = np.array([blue, red, gray])
agent_colors = colors[sim.state]

fig1, ax1 = plt.subplots(figsize = (3,3))
scat = ax1.scatter(sim.pos[:,0], sim.pos[:,1], color = agent_colors, s = 8)
ax1.set_ylim(0, sim.Ly)
ax1.set_xlim(0, sim.Lx)
ax1.set_xticks([])     
ax1.set_yticks([])       
ax1.set_aspect('equal')
ax1.set_title(f"t = {sim.t:.1f} (days)", size = 8)
fig1.tight_layout()

position_plot = pn.pane.Matplotlib(fig1)
plt.close(fig1)


#historgram of current cases
fig2, ax2 = plt.subplots(figsize = (5,2))
pct = 100 * np.array(sim.history)/ sim.N

#colors need to match the order of stacks
stack = ax2.stackplot(sim.ts, pct[:, 1], pct[:, 0], pct[:, 2], colors=[red, blue, gray]) 

ax2.set_ylim(0, 100)
ax2.set_xlim(0, max(5,sim.t))
ax2.tick_params(labelsize=6)
ax2.set_xlabel('t (days)', size = 7)

label_i = ax2.text(1.02, 0, '', transform=ax2.get_yaxis_transform(),
                   color=red,  va='center', size=7)
label_s = ax2.text(1.02, 0, '', transform=ax2.get_yaxis_transform(),
                   color=blue, va='center', size=7)
label_r = ax2.text(1.02, 0, '', transform=ax2.get_yaxis_transform(),
                   color='#000000', va='center', size=7)

fig2.tight_layout()
fig2.subplots_adjust(right=0.72) #adjust space for labels

SIR_hist = pn.pane.Matplotlib(fig2)
plt.close(fig2)


#estimating r eff
R_T = 5.0     # length of the time window in days

def estimate_R_eff(sim, tau_R = R_T):
    """Average number infected by agents who recovered during the last R_T days."""
    recent = (sim.t_rec > sim.t - tau_R) 
    if not recent.any():
        return np.nan
    return sim.n_infected[recent].mean()



def redraw():
    """Draw the current state of sim in the figure."""
    scat.set_offsets(sim.pos)
    scat.set_color(colors[sim.state])
    ax1.set_title(f"t = {sim.t:.1f} (days)", size=8)
    position_plot.param.trigger("object")

    #updating R_eff for each step
    R = estimate_R_eff(sim)
    if np.isnan(R):
        R_eff_display.object = "<b>R<sub>eff</sub></b> = "
    else:
        R_eff_display.object = f"<b>R<sub>eff</sub></b> = {R:.2f}"


    global stack 

    pct = 100 * np.array(sim.history)/ sim.N
    for area in stack:
        area.remove()                     # remove the old areas
    stack = ax2.stackplot(sim.ts, pct[:, 1], pct[:, 0], pct[:, 2],colors=[red, blue, gray])
    ax2.set_xlim(0, max(5,sim.t))

    # create moving labels for stack
    S, I, R = pct[-1]                         # current percentages

    label_i.set_y(I / 2)                      # middle of the bottom band
    label_s.set_y(I + S / 2)                  # middle of the middle band
    label_r.set_y(I + S + R / 2)              # middle of the top band

    label_i.set_text(f'Infected {I:.0f}%')
    label_s.set_text(f'Susceptible {S:.0f}%')
    label_r.set_text(f'Recovered {R:.0f}%')  

    SIR_hist.param.trigger("object")



def update_postion():
    if sim.counts()[1] == 0:            # outbreak over
        callback.stop()
        return
    for i in range(4):
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
    align = 'end',
    margin = (0,0,15,0)
)

R_eff_display = pn.pane.HTML("", styles={"font-size": "16px"}, align = 'start')

legend_and_r_eff = pn.Row(
    R_eff_display,
    legend,
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

position_section = pn.Column(
    pn.pane.Markdown('## Positions and SIR state for agents over time', align='center'),
    position_plot,
    legend_and_r_eff,
    position_plot_buttons,
    align = 'end'
)



info_param_values_1 = pn.pane.Markdown(
    f'''
    - Agents: N = {sim.N}
    - Initially infected: I₀ = {sim.I0}
    - Box: {sim.Lx:.0f} × {sim.Ly:.0f}
    - Movement: D = {sim.D}
    - Infection radius: r_inf = {sim.r_inf}
    - Contact rate: λ = {sim.lam}
    ''', 
    width=250
)
info_param_values_2 = pn.pane.Markdown(
    f'''
    - Recovery rate: γ = {sim.gamma}
    - Time step: Δt = {sim.dt}
    - Simulation time: visible on plots, stops when number of infected = 0
    - R_eff window: τ_R = {R_T:.0f} days

    ''', 
    width=250,
)
info_params = pn.Row(
    info_param_values_1,
    info_param_values_2
)
info_section = pn.Column(
    pn.pane.Markdown('## The param values used for this simulation:', align = 'start'),
    info_params,
    align = 'start',
    margin = (0,0,0,40)
)

histogram_section = pn.Column(
    info_section,
    pn.pane.Markdown('## Percentage evolution of SIR distribution', align='start', margin = (0,0,0,50)),
    SIR_hist,
    align = 'start'
)

horisontal_layout = pn.Row(
    histogram_section,
    position_section
)

app = pn.Column(
    '# SIR model simulation',
    horisontal_layout
)
app.servable()