import numpy as np

rng = np.random.default_rng(42)


# Colors - same as assignment examples
blue = "#1E4DB7" #Susceptible
red = "#F03224" #Infected
gray = "#999999" #recovered

# position and distance funtions

def random_positions(n, margin=0.2, Lx = 12.0, Ly = 8.0):
    """Generate n uniformly distributed points inside the box."""
    x = rng.uniform(margin, Lx - margin, n)
    y = rng.uniform(margin, Ly - margin, n)
    return np.column_stack((x, y))

def _reflect(position_coor, L):
    #position_coor is x og y coordinate
    #if position outside 0 wall, position = -position:
    position_coor = np.where(position_coor < 0, -position_coor, position_coor) 
    #if position is outside wall L, position = 2L - position
    position_coor = np.where(position_coor > L , 2*L - position_coor, position_coor)
    return position_coor

def update_position(position, step, N, Lx = 12.0, Ly = 8.0):
    new = position + step*rng.standard_normal((N,2))
    new[:,0] = _reflect(new[:,0], Lx)
    new[:,1] = _reflect(new[:,1], Ly)
    return new


def distance(X1, X2):
    return np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)

def P_infection(lamda, dt):
    #infection probability of a pair of agents (S, I)
    # where S is in infection radius of I
    return 1 - np.exp(-lamda*dt)

def P_tot_infection(k, lamda, dt):
    # gives total probability of infection in S
    #when S is in radius of k number of I,  
    return 1 - np.exp(-k*lamda*dt)

def P_recovery(gamma, dt):
    return 1-np.exp(-gamma*dt)

def initial_state(N, start_I_n=1, margin=0.2, Lx = 12.0, Ly = 8.0):
    position = random_positions(N, margin=margin, Lx=Lx, Ly=Ly)
    state = np.zeros(N, dtype = int) # 0 =S, 1 = I, 2 =R
    n_infected = np.zeros(N, dtype= int) #number of how many each agent have infected
    t_infected  = np.full(N, np.nan) #time each agent got infected
    t_recovered = np.full(N, np.nan) # time each agent recover

    for i in range(start_I_n):
        state[i] = 1
        t_infected[i] = 0.0

    return position, state, n_infected, t_infected, t_recovered

def inf_step(I, position, state, n_inf, t_inf, t, R, lamda, dt):
    S = np.where(state == 0)[0] # gives index of S from position array
    
    
    if len(S) == 0 or len(I) == 0:
        return
    in_radius = distance(position[S], position[I]) < R**2
    k = in_radius.sum(axis =1) # number of infected
    P = P_tot_infection(k,lamda, dt)
    new = rng.random(len(S)) < P

    for a in np.where(new)[0]:              
        inf_sorces = I[in_radius[a]]                  
        src = rng.choice(inf_sorces)   #since all probabilites have same lamda, we can choose infection source randomly        
        n_inf[src] += 1
        state[S[a]] = 1
        t_inf[S[a]] = t

def rec_step(I, state, t_rec, t, gamma, dt):
    P = P_recovery(gamma, dt)
    new = rng.random(len(I)) < P

    for a in np.where(new)[0]:
        state[I[a]] = 2
        t_rec[I[a]] = t

def simulate(N, dt, T):
    pos, state, n_inf, t_inf, t_rec = initial_state(N)
    ts = np.arange(0,T+0*dt, dt)
    for t in ts:
        I = np.where(state == 1)[0] # gives index of I from position array

def main():

    Lx = 12.0
    Ly = 8.0
    R = 0.8 
    N = 10 #number of agents
    D = 0.1
    dt = 0.1
    T = 20 # total time of simulation
    step = np.sqrt(2*D*dt) 
    lamda =0.3 #infection rate of S inside infectionradius f I 

    #initial state 
    pos, state, n_inf, t_inf, t_rec = initial_state(N)

    I = np.where(state == 1)[0] # gives index of I from position array

if __name__ == '__main__':
    main()