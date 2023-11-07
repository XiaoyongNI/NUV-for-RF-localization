"""
This file contains the dataset parameters for the simulations with RF localization 
* state: [px,vx,py,vy]
* F: constant velocity model
* h: non-linear, phase detection on the ULA
* Q: discretization from SDE, function of dt
* R: random phase noise (mod 2pi)
"""

import torch
import math

state_dim = 4 # dim of state 
dt = 0.1  # time interval
RF_freq = 8.6e8 # RF frequency
wave_length = 3e8/RF_freq # wave length
Phase_0 = 0 # initial phase

#################################################################
### process noise variance Q and observation phase noise std  ###
#################################################################
# Q
q = 0.125 * wave_length # process noise std 
Q_CV = q**2 * torch.tensor([[1/3*dt**3, 1/2*dt**2],
                            [1/2*dt**2,  dt]]).float()  
Q = torch.block_diag(Q_CV,Q_CV)
# r
SNR_dB = 20  # SNR in dB
SNR = 10**(SNR_dB/10) # SNR in linear scale
phase_noise_std = torch.sqrt(torch.tensor(1 / SNR)) # phase noise std

#################################
### state evolution matrix F  ###
#################################
F_1 = torch.tensor([[1, dt], [0, 1]], dtype=torch.float32)
F = torch.block_diag(F_1, F_1)

###############################
### observation function h  ###
###############################
def h_phase(states, ULA_array):
    """
    states: [px, vx, py, vy]
    ULA_array: (N, 2) array of antenna locations (x, y)
    
    return: phases of all antennas (rad)
    """

    N = ULA_array.shape[0]
    phases = torch.zeros((N), dtype=torch.float32)
    
    position = states[[0, 2]]

    # Expand dimensions of position to (N, 2)
    position_expanded = position.repeat(N,1)

    # Now you can compute the distances in a vectorized way
    distances_to_antenna = torch.norm(position_expanded - ULA_array, dim=1)
        
    # Compute the phases for all antennas at once
    phases = (distances_to_antenna * 4 * math.pi / wave_length + Phase_0) % (2 * math.pi)
    
    return phases

