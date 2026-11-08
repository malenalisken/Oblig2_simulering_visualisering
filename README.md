# Oblig2_simulering_visualisering
## Contributors
Tora Larsen Bredal
Malena Amelie Lisken

## The files
- **src/**
    - **del_1.py** : simulation and animations of shallow water equations
    - **del_2.py** : simulation of SIR model
    - **app.py** : task 2b, SIR model vizualized with panel
- **figures/**
    - **animation_SW/** : shallow water animations
        - **small_scale_anim_BC/** : one animation per boundary condition with suggested constant values from task 1. One extra top view anim of dirichlet to show offset of centre
        - **quiver** : quiver plot of velocity for periodic BC
        - **quiver_large_scale** : as above for the larger scale animation
        - **shallow_water_blues and seismic** : large scale shallow water animation with neumann boundary condition in blue color scale and seismic color scale
    - **finer_grid_check** : plots a cross section of simulation at t = 5 with varying number of nodes in x and y dir
    - **plot_initial_conidtion** : plot of shallow water at initial condition
    - **seed_study** : plot with 5 single runs with different seed and mean of those for SIR distribution over time
    - **SIR_hist_with_starting_values** : single run of SIR dsitribution over time
- **requirements** : needed libraries with version number


## Setup
This project uses **Python 3.12**
```bash
#Check which version you have in bash. If it fails install python 3.12 from python.org
python3.12 --version   

#create and activate virtual enviroment (once)
python3.12 -m venv .venv
source .venv/bin/activate 

#install the required libraries
pip install -r requirements.txt
```
## How to run
Each figure/animation for del1 and del2 open in their own window, close to move to the next
From the root of the project run:
```bash
python src/del_1.py 
python src/del_2.py
panel serve src/app.py --show 
```
