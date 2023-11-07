"""
Data generator for RF localization.

"""

import torch
import math

class DataGenerator:
    def __init__(self, args):
        self.args = args
        # Set device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # variance of complex AWGN
        self.r2 = torch.tensor(args.r2) 
    
    def ULA_array(self): # can be both (x,y) or (r,theta)
        ULA_gap = 0.5 * self.args.wave_length # ULA gap is half of wave length
        ULA_array = torch.zeros((self.args.n, 2), dtype=torch.float32) # y coordinates/theta coordinates are all 0
        ULA_array[:, 0] = torch.arange(self.args.n) * ULA_gap # x coordinates/ r coordinates

        return ULA_array
    
    # (x, y) coordinates
    def position_generator_xy(self): 
        """Generates random positions and DOA."""
        # Generate random positions
        gt_positions = torch.zeros(self.args.sample, self.args.k, 2, dtype=torch.float32, device=self.device)
        # x coordinates
        gt_positions[:, :, 0] = torch.rand(self.args.sample, self.args.k, dtype=torch.float32, device=self.device) * (self.args.position_gt_xright_bound - self.args.position_gt_xleft_bound) + self.args.position_gt_xleft_bound 
        # y coordinates
        gt_positions[:, :, 1] = torch.rand(self.args.sample, self.args.k, dtype=torch.float32, device=self.device) * (self.args.position_gt_yright_bound - self.args.position_gt_yleft_bound) + self.args.position_gt_yleft_bound
        
        return gt_positions

    # (r, theta) coordinates
    def position_generator_rtheta(self): 
        """Generates random positions and DOA."""
        # Generate random positions
        gt_positions = torch.zeros(self.args.sample, self.args.k, 2, dtype=torch.float32, device=self.device)
        # r coordinates
        gt_positions[:, :, 0] = torch.rand(self.args.sample, self.args.k, dtype=torch.float32, device=self.device) * (self.args.position_gt_rright_bound - self.args.position_gt_rleft_bound) + self.args.position_gt_rleft_bound
        # theta coordinates
        gt_positions[:, :, 1] = torch.rand(self.args.sample, self.args.k, dtype=torch.float32, device=self.device) * (self.args.position_gt_thetaright_bound - self.args.position_gt_thetaleft_bound) + self.args.position_gt_thetaleft_bound
        # convert deg to rad
        gt_positions[:, :, 1] = gt_positions[:, :, 1] * math.pi / 180
        return gt_positions

    def obs_steeringmatrix_rtheta(self, gt_positions, ULA_array):
        """
        Generates steering matrices for observations, given ground truth positions.
        """
        STM_A = torch.zeros(self.args.sample, self.args.n, self.args.k, dtype=torch.cfloat, device=self.device)
        
        # compute steering matrix
        for i in range(self.args.k):
            # broadcast source to (sample, n, 2), i.e. repeat n times
            source = gt_positions[:, i, :].unsqueeze(1).repeat(1, self.args.n, 1)
            # broadcast ULA array to (sample, n, 2), i.e. repeat sample times
            ULA_array_batch = ULA_array.unsqueeze(0).repeat(self.args.sample, 1, 1)
            # compute distances from each source to each ULA element (polar coordinates)
            distances = polar_distance(source[:, :, 0], source[:, :, 1], ULA_array_batch[:, :, 0], ULA_array_batch[:, :, 1])
            # compute phase shifts
            phase_shifts = distances * 2 * math.pi / self.args.wave_length
            # compute steering matrix
            STM_A[:, :, i] = torch.exp(-1j * phase_shifts)

        return STM_A

    def nonco_signal_generator(self):
        """Generates non-coherent source signals."""
        
        # Generate real and imaginary parts using torch.randn
        real_part = torch.randn(self.args.sample, self.args.l, self.args.k, 1, dtype=torch.float32, device=self.device)
        imag_part = torch.randn(self.args.sample, self.args.l, self.args.k, 1, dtype=torch.float32, device=self.device)
        
        # Scale by the factor derived from variance
        scale_factor = torch.sqrt(torch.tensor(self.args.x_var / 2, dtype=torch.float32))
        
        x = scale_factor * (real_part + 1j * imag_part)

        x = x + self.args.mean_c
        
        return x

    def generate_experiment_data_rtheta(self):
        """Experiment data generation."""
        gt_positions = self.position_generator_rtheta()
        
        STM_A = self.obs_steeringmatrix_rtheta(gt_positions, self.ULA_array())

        x_true = self.nonco_signal_generator()
        
        y_train = torch.zeros(self.args.sample, self.args.l, self.args.n, 1, dtype=torch.cfloat, device=self.device)
        
        for j in range(self.args.sample):
            for t in range(self.args.l):
                er1 = torch.normal(mean=0.0, std=torch.sqrt(self.r2 / 2), size=(self.args.n,)).to(self.device)
                er2 = torch.normal(mean=0.0, std=torch.sqrt(self.r2 / 2), size=(self.args.n,)).to(self.device)

                y_train[j, t, :, 0] = STM_A[j].matmul(x_true[j, t, :, 0]) + er1 + er2 * 1j
                
        return gt_positions, x_true, y_train
    
    def dictionary_matrix_rtheta(self):
        """Generate dictionary."""
        ## Generate hypothesis positions
        r_left = self.args.position_gt_rleft_bound
        r_right = self.args.position_gt_rright_bound
        theta_left = self.args.position_gt_thetaleft_bound
        theta_right = self.args.position_gt_thetaright_bound
        r_resol = (r_right - r_left) / (self.args.m_r-1)
        theta_resol = (theta_right - theta_left) / (self.args.m_theta-1)       
        r_positions = torch.arange(r_left, r_right + r_resol, r_resol)
        theta_positions = torch.arange(theta_left, theta_right + theta_resol, theta_resol)
        # convert deg to rad
        theta_positions = theta_positions * math.pi / 180 
        # Create a meshgrid of all r, theta pairs
        R, Theta = torch.meshgrid(r_positions, theta_positions)
        # Flatten the meshgrids to create vectors of r and theta
        r_vector = R.flatten()
        theta_vector = Theta.flatten()
        # Pair them up in a 2D tensor (each row is a pair [r, theta]), size (m_r*m_theta, 2)
        position_pairs = torch.stack((r_vector, theta_vector), dim=1)
        
        ## compute dictionary matrix
        # broadcast ULA array to (m_r*m_theta, n, 2)
        ULA_array_batch = self.ULA_array().unsqueeze(0).repeat(self.args.m_r*self.args.m_theta, 1, 1)
        # broadcast position pairs to (m_r*m_theta, n, 2)
        position_pairs_batch = position_pairs.unsqueeze(1).repeat(1, self.args.n, 1)
        # compute distances from each hypothesis source to each ULA element (polar coordinates)
        distances = polar_distance(position_pairs_batch[:, :, 0], position_pairs_batch[:, :, 1], ULA_array_batch[:, :, 0], ULA_array_batch[:, :, 1])
        # compute phase shifts
        phase_shifts = distances * 2 * math.pi / self.args.wave_length
        # compute steering matrix
        A_dic = torch.exp(-1j * phase_shifts)
        A_dic = A_dic.transpose(0, 1) # (m_r*m_theta, n) -> (n, m_r*m_theta)

        return A_dic, r_positions, theta_positions

def polar_distance(r1, theta1, r2, theta2): 
    """
    Use the law of cosines for polar coordinates
    r1, theta1, r2, theta2: (sample, n)
    distances: (sample, n)
    """    
    # compute distance
    # r1^2 + r2^2 - 2*r1*r2*cos(theta1 - theta2)
    distances = torch.sqrt(torch.pow(r1, 2) + torch.pow(r2, 2) - 2 * r1 * r2 * torch.cos(theta1 - theta2))  
            
    return distances

# Unit test 
