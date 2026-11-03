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

def reflect(position, L):
    #if position outside 0 wall, position = -position:
    position = np.where(position < 0, -position, position) 
    #if position is outside wall L, position = L - last step
    position = np.where(position > L , 2*L - position, position)
    return position

def update_position(position, dx, N, Lx = 12.0, Ly = 8.0):
    new = position + dx*rng.standard_normal((N,2))
    new[:,0] = reflect(new[:,0], Lx)
    new[:,1] = reflect(new[:,1], Ly)
    return new


def distance(X1, X2):
    return np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)

#infection    

if __name__ == '__main__':

    Lx = 12.0
    Ly = 8.0
    R = 0.8 
    N = 500
    D = 0.1
    dt = 0.1
    dx = np.sqrt(2*D*dt)
    #postition array
    pos = random_positions(N)
    state = np.zeros(N, dtype = int) # 0 =S, 1 = I, 2 =R
    n_infected = np.zeros(N, dtype= int) #how many each agent have infected