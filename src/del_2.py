import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from constants import *



def _reflect(p, L):
    #if p position outside 0 wall, position = -position:
    p = np.where(p < 0, -p, p) 
    #if position is outside wall L, position = 2L - position
    p = np.where(p > L , 2*L - p, p)
    return p

def distance_2(X1, X2):
    return np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)

def P_tot_infection(k, lam, dt):
    # gives total probability of infection in S
    #when S is in radius of k number of I,  
    return 1 - np.exp(-k*lam*dt)

def P_recovery(gamma, dt):
    return 1-np.exp(-gamma*dt)


class Simulation:
    '''Agent base SIR model'''

    def __init__(self, N=400, dt=0.02, r_inf=1.0, lam=0.1, gamma=0.1, D=2.0,
                 Lx=20.0, Ly=20.0, I0=4, R0=0, seed=42, margin=0.2):
        # parameters
        self.N, self.dt, self.r_inf = N, dt, r_inf
        self.lam, self.gamma, self.D = lam, gamma, D
        self.Lx, self.Ly, self.I0, self.R0 = Lx, Ly, I0, R0
        self.step_size = np.sqrt(2*D*dt)

        # random numb generator
        self.rng = np.random.default_rng(seed)

        #agents
        x = self.rng.uniform(margin, Lx - margin, N)
        y = self.rng.uniform(margin, Ly - margin, N)
        self.pos = np.column_stack((x, y))
        self.state = np.zeros(N, dtype=int)             # 0 = S, 1 = I, 2 = R
        self.n_infected = np.zeros(N, dtype=int)        # how many infected
        self.t_inf = np.full(N, np.nan)                 # time of infection
        self.t_rec = np.full(N, np.nan)                 # time of recovery

        #inital distrubution or SIR
        self.state[:I0] = 1                             
        self.t_inf[:I0] = 0                             # mark time for initial infected as 0
        self.state[I0:I0 + R0] = 2                      # if we want to start with some immunities
        self.t_rec[I0:I0 + R0] = 0.0

        #time and history
        self.n = 0                          #number of time steps
        self.t = 0.0
        self.ts = [0.0]                     #all timepoints so far
        self.history = [self.counts()]

    def counts(self):
        '''returns [S, I, R] at current point in time'''
        return [(self.state == 0).sum(), (self.state == 1).sum(), (self.state == 2).sum()]

    def _move(self):
        new = self.pos + self.step_size * self.rng.standard_normal((self.N, 2))
        new[:, 0] = _reflect(new[:, 0], self.Lx)
        new[:, 1] = _reflect(new[:, 1], self.Ly)
        # test to check that agents remain in box
        if np.any(new < 0) or np.any(new[:, 0] > self.Lx) or np.any(new[:, 1] > self.Ly):
            raise ValueError('Agent has ventured outside of the box')
        self.pos = new
        
def random_positions(n,  Lx , Ly, margin=0.2):
    """Generate n uniformly distributed points inside the box."""
    x = rng.uniform(margin, Lx - margin, n)
    y = rng.uniform(margin, Ly - margin, n)
    return np.column_stack((x, y))



def update_position(position, step, N, Lx = 20.0, Ly = 20.0):
    new = position + step*rng.standard_normal((N,2))
    new[:,0] = _reflect(new[:,0], Lx)
    new[:,1] = _reflect(new[:,1], Ly)
    if np.any(new < 0) or np.any(new[:,0] > Lx) or np.any(new[:,1] > Ly):
        raise ValueError('Agent has ventured outside of the box')
    return new






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