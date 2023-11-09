"""This file contains the settings for the simulation"""
import argparse

def general_settings():
    parser = argparse.ArgumentParser(prog = 'NUV-RF',\
                                     description = 'Dataset and tuning parameters')
    
    ### Dataset settings
    parser.add_argument('--sample', type=int, default=10, metavar='dataset-size',
                        help='input dataset size')
    parser.add_argument('--k', type=int, default=1, metavar='k',
                        help='number of sources')
    parser.add_argument('--n', type=int, default=10, metavar='n',
                        help='number of ULA elements')
    parser.add_argument('--m_r', type=int, default=11, metavar='m_r',
                        help='number of grids in r')
    parser.add_argument('--m_theta', type=int, default=31, metavar='m_theta',
                        help='number of grids in theta')


    parser.add_argument('--x_var', type=float, default=0.5, metavar='x_var',
                        help='variance of source signals')
    parser.add_argument('--mean_c', type=float, default=2, metavar='mean_c',
                        help='mean of source signals')
    parser.add_argument('--r2', type=float, default=1e-3, metavar='r2',
                        help='ground truth variance of observation noise')
    parser.add_argument('--l', type=int, default=100, metavar='l',
                        help='number of snapshots')

    parser.add_argument('--wave_length', type=float, default=1, metavar='wave_length',
                        help='wave length (meter)')





    ### Tuning settings
    parser.add_argument('--use_cuda', type=bool, default=False, metavar='CUDA',
                        help='if True, use CUDA')
    parser.add_argument('--q_init', type=float, default=0.01, metavar='q_init',
                        help='initial guess of q')
    parser.add_argument('--max_iterations', type=int, default=10000, metavar='max_iterations',
                        help='maximum number of iterations')
    parser.add_argument('--convergence_threshold', type=float, default=4e-4, metavar='convergence_threshold',
                        help='convergence threshold')
    parser.add_argument('--resol', type=float, default=0.05, metavar='resol',
                        help='resolution of spatial filter')
    # coordinate system
    parser.add_argument('--coordinate', type=str, default='rtheta', metavar='coordinate',
                        help='coordinate system of ground truth position, can be rtheta or xy')
    # xy coordinates
    parser.add_argument('--position_gt_xleft_bound', type=float, default=0, metavar='position_gt_xleft_bound',
                        help='left boundary of ground truth position x axis')
    parser.add_argument('--position_gt_xright_bound', type=float, default=10, metavar='position_gt_xright_bound',
                        help='right boundary of ground truth position x axis')
    parser.add_argument('--position_gt_yleft_bound', type=float, default=0, metavar='position_gt_yleft_bound',
                        help='left boundary of ground truth position y axis')   
    parser.add_argument('--position_gt_yright_bound', type=float, default=10, metavar='position_gt_yright_bound',     
                        help='right boundary of ground truth position y axis')
    # rtheta coordinates
    parser.add_argument('--position_gt_rleft_bound', type=float, default=500, metavar='position_gt_rleft_bound',
                        help='left boundary of ground truth position r axis')
    parser.add_argument('--position_gt_rright_bound', type=float, default=550, metavar='position_gt_rright_bound',
                        help='right boundary of ground truth position r axis')
    parser.add_argument('--position_gt_thetaleft_bound', type=float, default=30, metavar='position_gt_thetaleft_bound',
                        help='left boundary of ground truth position theta axis (degree)')
    parser.add_argument('--position_gt_thetaright_bound', type=float, default=45, metavar='position_gt_thetaright_bound',
                        help='right boundary of ground truth position theta axis (degree)')
    

    args = parser.parse_args()
    return args
