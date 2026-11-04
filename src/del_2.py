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

def P_vaccination(nu, dt):
    return 1 - np.exp(-nu*dt)
class Simulation:
    '''Agent base SIR model'''

    def __init__(self, N=400, dt=0.02, r_inf=1.0, lam=0.1, gamma=0.1, D=2.0,
                 Lx=20.0, Ly=20.0, I0=4, R0=0, t_vac = np.inf, nu = 0.0, seed=42, margin=0.2, ):
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

        #vaccination
        self.t_vac = t_vac                #time when vacinations can start, default infinite
        self.nu = nu
        self.vaccinated = np.zeros(N,dtype=bool)   #if vaccinated
        self.t_vac = np.full(N, np.nan)            #time of vaccination

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

    def _infect(self, I):
        S = np.where(self.state == 0)[0]    # gives index of S from position array
        if len(S) == 0 or len(I) == 0:
            return
        in_radius = distance_2(self.pos[S], self.pos[I]) < self.r_inf**2
        k = in_radius.sum(axis = 1)

        P = P_tot_infection(k, self.lam, self.dt)
        new = self.rng.random(len(S)) < P

        for a in np.where(new)[0]:
                    inf_sources = I[in_radius[a]]
                    src = self.rng.choice(inf_sources)   # since all probabilities have same lambda, we can choose infection source randomly
                    self.n_infected[src] += 1
                    self.state[S[a]] = 1
                    self.t_inf[S[a]] = self.t

    def _recover(self, I):
        P = P_recovery(self.gamma, self.dt)
        new = self.rng.random(len(I)) < P

        self.state[I[new]] = 2
        self.t_rec[I[new]] = self.t

    def step(self):
        self.n += 1
        self.t = self.n * self.dt
        self._move()

        I = np.where(self.state == 1)[0] #indexes of infected agents
        prev_state = self.state.copy()
        self._infect(I)
        self._recover(I)

        #test for illegar tranformations in state
        transition = self.state - prev_state
        if np.any((transition > 1) | (transition < 0)):
            raise ValueError(f"Illegal transition at t={self.t:.2f}")

        self.ts.append(self.t)
        self.history.append(self.counts())

        #check for constant total agents
        if sum(self.history[-1]) != self.N:
            raise ValueError(f"Total number of agents is not constant at t={self.t:.2f}")

def simulate(N, T, dt, r_inf, lam, gamma, D, Lx, Ly, I0, R0, t_vac, nu, seed=42):
    '''Instance of an simulation object and running it'''
    sim = Simulation(N=N, dt=dt, r_inf=r_inf, lam=lam, gamma=gamma, D=D,
                    Lx=Lx, Ly=Ly, I0=I0, R0=R0, seed=seed)
    n_steps = int(round(T / dt))
    for _ in range(n_steps):
        sim.step()
    return np.array(sim.ts), np.array(sim.history), sim.n_infected, sim.t_inf, sim.t_rec


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
        fig.savefig(filename)

def plot_seed_study(ts, sirs, peaks, t_peaks, seeds, dt, filename=None):
    mean_sir = sirs.mean(axis=0)

    fig, ax = plt.subplots()
    #plotting each sir: 
    for i, sir in enumerate(sirs):
        ax.plot(ts, sir[:, 0], color=blue, alpha=0.3, lw=1,)
        ax.plot(ts, sir[:, 1], color=red, alpha=0.3, lw=1)
        ax.plot(ts, sir[:, 2], color=gray, alpha=0.3, lw=1)
    ax.scatter(t_peaks, peaks, color=red, s=20, zorder=3, label='Peak of each run')
    #plotting the mean: 
    ax.plot(ts, mean_sir[:, 0], color=blue, lw=2.5, label='Susceptible (mean)')
    ax.plot(ts, mean_sir[:, 1], color=red, lw=2.5, label='Infected (mean)')
    ax.plot(ts, mean_sir[:, 2], color=gray, lw=2.5, label='Recovered (mean)')

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.set_title(f'SIR for {len(seeds)} seeds, thin lines are single runs')
    ax.set_ylabel('Agents')
    ax.set_xlabel(f'Time (dt = {dt})')
    ax.grid(alpha=0.3)
    if filename is not None:
        fig.savefig(filename)

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

def seed_study(N , T,dt, r_inf, lam, gamma, D, Lx, Ly, I0, R0, seeds):
    sirs, peaks, t_peaks, totals = [], [], [], []
    for seed in seeds:
        ts, sir, *_ = simulate(N=N, T=T, dt=dt, r_inf=r_inf, lam=lam, gamma=gamma,
                                   D=D, Lx=Lx, Ly=Ly, I0=I0, R0=R0, seed=seed)
        I = sir[:, 1]
        sirs.append(sir)
        peaks.append(I.max())
        t_peaks.append(ts[I.argmax()])
        totals.append(N - sir[-1, 0] - R0)

    sirs = np.array(sirs)
    peaks, t_peaks, totals = np.array(peaks), np.array(t_peaks), np.array(totals)
     # table of results
    print(f"{'seed':>6} {'peak I':>8} {'t_peak':>8} {'total':>8}")
    for seed, p, tp, tot in zip(seeds, peaks, t_peaks, totals):
        print(f"{seed:>6} {p:>8} {tp:>8.1f} {tot:>8}")
    print(f"{'mean':>6} {peaks.mean():>8.1f} {t_peaks.mean():>8.1f} {totals.mean():>8.1f}")
    print(f"{'std':>6} {peaks.std():>8.1f} {t_peaks.std():>8.1f} {totals.std():>8.1f}")
    print(f"{'min':>6} {peaks.min():>8} {t_peaks.min():>8.1f} {totals.min():>8}")
    print(f"{'max':>6} {peaks.max():>8} {t_peaks.max():>8.1f} {totals.max():>8}")

    return ts, sirs, peaks, t_peaks
def main():
    
    #Test with 
    ts, sir_history, *_ = simulate(N=N, T=T, dt=dt, r_inf=r_inf,
                                            lam=lam, gamma=gamma, D=D, 
                                            Lx=Lx, Ly=Ly, I0=I0, R0=R0)
    #plot_sir(sir_history, ts, dt = dt)
    #plt.show() 

    #check boundary cases
    
    lam_bc = [0, lam]
    I0_bc = [I0, 0]
    for l, i0 in zip(lam_bc, I0_bc):
        ts, sir_his, n_inf, t_inf, t_rec = simulate(N=N, T=T, dt=dt, r_inf=r_inf,
                                                lam=l, gamma=gamma, D=D, 
                                                Lx=Lx, Ly=Ly, I0=i0, R0=R0)
        if ((sir_his[0,0] != sir_his[-1,0]) or (sir_his[-1,2] != R0 + i0) or (sir_his[-1,1] != 0)):
            raise ValueError(f'The boundary conditions for lambda = {l} and I0 = {i0} give unexpected values')
    
    #undersøke tidssteg
    dts = [dt, dt/2]
    random_seeds = range(10)
    #timestep_study(N, T, r_inf, lam, gamma, D, Lx, Ly, I0, R0, dts=dts, seeds=random_seeds)

    seeds = range(5)
    #ts, sirs, peaks, t_peaks = seed_study(N, T, dt, r_inf, lam, gamma, D, Lx, Ly,
    #                                      I0, R0, seeds=seeds)
    #plot_seed_study(ts, sirs, peaks, t_peaks, seeds, dt, filename=None)

    #'/figures/seed_study.png'
    plt.show()
        


    

if __name__ == '__main__':
    main()