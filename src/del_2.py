import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator




# Colors - same as assignment examples
blue = "#1E4DB7" #Susceptible
red = "#F03224" #Infected
gray = "#999999" #recovered

# position and distance funtions

def random_positions(n,  Lx , Ly, margin=0.2):
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

def update_position(position, step, N, Lx = 20.0, Ly = 20.0):
    new = position + step*rng.standard_normal((N,2))
    new[:,0] = _reflect(new[:,0], Lx)
    new[:,1] = _reflect(new[:,1], Ly)
    if ((Lx < np.any(new[:,0] < 0)) or (Ly < np.any(new[:,0] < 0))):
        raise ValueError('Agent has ventured outside of the box')
    return new


def distance_2(X1, X2):
    return np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)


def P_tot_infection(k, lam, dt):
    # gives total probability of infection in S
    #when S is in radius of k number of I,  
    return 1 - np.exp(-k*lam*dt)

def P_recovery(gamma, dt):
    return 1-np.exp(-gamma*dt)

def initial_state(N, I0, R0, Lx, Ly, margin=0.2):
    position = random_positions(N, Lx=Lx, Ly=Ly,margin=margin,)
    state = np.zeros(N, dtype = int) # 0 =S, 1 = I, 2 =R
    n_infected = np.zeros(N, dtype= int) #number of how many each agent have infected
    t_infected  = np.full(N, np.nan) #time each agent got infected
    t_recovered = np.full(N, np.nan) # time each agent recover

    for i in range(I0):
        state[i] = 1
        t_infected[i] = 0.0
    for i in range(I0,R0+I0):
        state[i] = 2
        t_recovered[i] = 0.0

    return position, state, n_infected, t_infected, t_recovered

def inf_step(I, position, state, n_inf, t_inf, t, r_inf, lam, dt):
    S = np.where(state == 0)[0]         # gives index of S from position array
    
    
    if len(S) == 0 or len(I) == 0:
        return
    in_radius = distance_2(position[S], position[I]) < r_inf**2
    k = in_radius.sum(axis =1)          # number of infected
    P = P_tot_infection(k,lam, dt)
    new = rng.random(len(S)) < P

    for a in np.where(new)[0]:              
        inf_sorces = I[in_radius[a]]                  
        src = rng.choice(inf_sorces)    # since all probabilites have same l, we can choose infection source randomly        
        n_inf[src] += 1
        state[S[a]] = 1
        t_inf[S[a]] = t

def rec_step(I, state, t_rec, t, gamma, dt):
    P = P_recovery(gamma, dt)
    new = rng.random(len(I)) < P

    for a in np.where(new)[0]:
        state[I[a]] = 2
        t_rec[I[a]] = t

def simulate(N, T, dt, r_inf, lam, gamma, D, Lx, Ly, I0, R0, seed = 42):
    global rng 
    rng = np.random.default_rng(seed)

    step = np.sqrt(2*D*dt)
    pos, state, n_inf, t_inf, t_rec = initial_state(N, I0, R0, Lx=Lx, Ly=Ly)
    ts = np.arange(0, T + 0.5*dt, dt)
    sir_history = np.zeros((len(ts), 3), dtype=int)
    sir_history[0] = [(state==0).sum(), (state==1).sum(), (state==2).sum()]

    for n, t in enumerate(ts[1:]):
        pos = update_position(pos, step, N, Lx, Ly)
        I = np.where(state == 1)[0]

        prev_state = state.copy()

        inf_step(I, pos, state, n_inf, t_inf, t, r_inf, lam, dt)    #who gets infected
        rec_step(I, state, t_rec, t, gamma, dt)                     #who recovers
        
        transition = state - prev_state

        #test for illegal transitions
        if np.any((transition > 1 )| (transition < 0)):
            raise ValueError(f"Illegal transition at t={t:.2f}")
        
        sir_history[n + 1] = [(state==0).sum(), (state==1).sum(), (state==2).sum()]
       
        #check for maintaining number of agents
        if (np.sum(sir_history[n+1]) != N):
            raise ValueError(f'Total number of agents is not constant for {n+1}')

    return ts, sir_history, n_inf, t_inf, t_rec

def plot_sir(sir_history, ts, dt,  filename = None):
    fig, ax = plt.subplots()
    ax.plot(ts, sir_history[:,0],color = blue, label = 'Susceptible')
    ax.plot(ts, sir_history[:,1], color = red, label = 'Infected')
    ax.plot(ts, sir_history[:,2], color = gray, label = 'Recovered')
    ax.plot(ts, sir_history.sum(axis = 1), label = 'Total of agents')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True), )
    ax.legend()
    ax.set_title('SIR History')
    ax.set_ylabel('Agents')
    ax.set_xlabel(f'Time (dt = {dt})')
    ax.grid(alpha = 0.3)
    if filename is not None:
        plt.savefig(filename)

def timestep_study(N, T, r_inf, lam, gamma, D, Lx, Ly, I0, R0, dts, seeds):
    for dt in dts:
        peaks, t_peaks, totals = [], [], []
        for seed in seeds:
            ts, sir, *_ = simulate(N=N, T=T, dt=dt, r_inf=r_inf, lam=lam, gamma=gamma,
                                   D=D, Lx=Lx, Ly=Ly, I0=I0, R0=R0, seed=seed)
            I = sir[:, 1]
            peaks.append(I.max())
            t_peaks.append(ts[I.argmax()])
            totals.append(N - sir[-1, 0] - R0)

        peaks, t_peaks, totals = np.array(peaks), np.array(t_peaks), np.array(totals)
        print(f"dt = {dt}: peak I = {peaks.mean():.1f} ± {peaks.std():.1f}, "
              f"t_peak = {t_peaks.mean():.1f} ± {t_peaks.std():.1f}, "
              f"total infected = {totals.mean():.1f} ± {totals.std():.1f}")

def main():
    
    #TODO: deloppgave 2a punkt 4 og 5
    N = 400         #number of agents
    I0 = 4          #number of infected at start
    R0= 0           #number of recovered at start
    Lx = Ly = 20
    D = 2           #movement param
    r_inf = 1       #infection radius
    lam =0.1        #lambda: infection rate of S inside infectionradius f I 
    gamma = 0.1     #gamma: recovery rate
    
    dt = 0.02
    T = 90          #total time of simulation
    
    
    #Test with 
    ts, sir_history, *_ = simulate(N=N, T=T, dt=dt, r_inf=r_inf,
                                            lam=lam, gamma=gamma, D=D, 
                                            Lx=Lx, Ly=Ly, I0=I0, R0=R0)
    #plot_sir(sir_history, ts, dt = dt, filename="figures/SIR_hist_with_starting_values.png")
    #plt.show() 

    #check boundary conditions
    '''
    lam_bc = [0, lam]
    I0_bc = [I0, 0]
    for l, i0 in zip(lam_bc, I0_bc):
        ts, sir_his, n_inf, t_inf, t_rec = simulate(N=N, T=T, dt=dt, r_inf=r_inf,
                                                lam=l, gamma=gamma, D=D, 
                                                Lx=Lx, Ly=Ly, I0=i0, R0=R0)
        if ((sir_his[0,0] != sir_his[-1,0]) or (sir_his[-1,2] != R0 + i0) or (sir_his[-1,1] != 0)):
            raise ValueError(f'The boundary conditions for lambda = {l} and I0 = {i0} give unexpected values')
    '''
    #undersøke tidssteg
    dts = [dt, dt/2]
    random_seeds = range(10)
    timestep_study(N, T, r_inf, lam, gamma, D, Lx, Ly, I0, R0, dts=dts, seeds=random_seeds)


        


    

if __name__ == '__main__':
    main()