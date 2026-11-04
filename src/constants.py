import numpy as np

# Colors - same as assignment examples
blue = "#1E4DB7" #Susceptible
red = "#F03224" #Infected
gray = "#999999" #recovered

N = 400         #number of agents
I0 = 4          #number of infected at start
R0= 0           #number of recovered at start
Lx = Ly = 20
D = 2           #movement param
r_inf = 1       #infection radius
lam =0.1        #lambda: infection rate of S inside infectionradius f I 
gamma = 0.1     #gamma: recovery rate
    
T = 90          #total time of simulation
dt = 0.02       #size of timestep