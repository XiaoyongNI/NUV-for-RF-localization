import torch
import math
import time
import scipy.io

from NUV import NUV_SSR
from simulations import utils
from simulations import config
from data.data_gen import DataGenerator

#### initialization ####
args = config.general_settings()
args.use_cuda = False
# GPU or CPU
if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")
# path names
plot_folder = 'simulations/plots/'
data_folder = 'data/'
data_file_name = 'data_polar_default.pt'
matlab_file_name = 'result_polar_2iters.mat'

# Tuning parameters
args.q_init = 0.01
r_tuning = 1
args.m_r = 10
args.m_theta = 91
m = args.m_r * args.m_theta # total num of hypotheses
args.convergence_threshold = 4e-4
next_iter_std_mult = 3

# dataset settings
args.sample = 2 # number of samples
samples_run = args.sample
args.on_grid = False # gt positions are on grid or not
args.plot_grid = True # plot grid or not
args.position_gt_rleft_bound = 500
args.position_gt_rright_bound = 550
args.position_gt_thetaleft_bound = 45
args.position_gt_thetaright_bound = 135

#### Generate data ####
generator_iter1 = DataGenerator(args)
gt_positions, x_true, y_train, y_noiseless = generator_iter1.generate_experiment_data_rtheta()
torch.save([gt_positions, x_true, y_train, y_noiseless], data_folder+data_file_name)
[gt_positions, x_true, y_train, y_noiseless] = torch.load(data_folder+data_file_name, map_location=device)
# generate dictionary matrix A_dic, and corresponding hypothesis positions (r, theta)
A_dic, r_positions, theta_positions = generator_iter1.dictionary_matrix_rtheta() 
y_mean = y_train.mean(dim=1) # generate y_mean by averaging l snapshots for each sample

#### estimation ####
print('======================================')
# Tuning parameter
print('Tuning parameter (iteration1):')
print('r_tuning = {}'.format(r_tuning))
print('convergence_threshold = {}'.format(args.convergence_threshold))
# Dataset
print('r range = [{}, {}]'.format(args.position_gt_rleft_bound, args.position_gt_rright_bound))
print('theta range = [{}, {}] deg'.format(args.position_gt_thetaleft_bound, args.position_gt_thetaright_bound))
print('# sample points of r = {}'.format(args.m_r))
print('# sample points of theta = {}'.format(args.m_theta))

start = time.time()
# initialize
x_pred = torch.zeros(samples_run, m, dtype=torch.cfloat, device=device)
EM_steps = torch.zeros(samples_run, dtype=torch.int, device=device)

# NUV-SSR 
for i in range(samples_run):
    x_pred[i], EM_steps[i] = NUV_SSR(args, A_dic, y_mean[i], r_tuning, m)   
    print ('EM steps = {}'.format(EM_steps[i]))


# de-flatten x_pred [sample, m_r*m_theta] -> [sample, m_r, m_theta]
x_pred_2D = utils.batch_de_flatten(x_pred, args.m_r, args.m_theta)

# find peaks [sample, k, 3]
peak_indices = utils.batch_peak_finding_2D(x_pred_2D, args.k)

# convert to positions [sample, k, 2]
pred_positions = utils.batch_convert_to_positions(peak_indices, r_positions, theta_positions)

end = time.time()
t_iter1 = end - start

# convert to xy coordinates
pred_positions_xy = utils.batch_polar_to_cartesian(pred_positions)
gt_positions_xy = utils.batch_polar_to_cartesian(gt_positions)

# compute RMSEs
squared_diffs_xy = utils.batched_permuted_SquareDiff_2D(pred_positions_xy, gt_positions_xy) 
RMSE_distance = utils.RMSE_distance_error(squared_diffs_xy)
squared_diffs_polar = utils.batched_permuted_SquareDiff_2D(pred_positions, gt_positions)
RMSE_r, RMSE_theta = utils.RMSE_AxisWise_error(squared_diffs_polar)
RMSE_theta = RMSE_theta * 180 / math.pi

print('Results (iteration1):')
print('average EM steps = {}'.format(torch.mean(EM_steps.float())))
print('averaged RMSE distance = {} [m]'.format(RMSE_distance))
print('averaged RMSE r = {} [m]'.format(RMSE_r))
print('averaged RMSE theta = {} [deg]'.format(RMSE_theta))
# Print Run Time
print("Total Run Time:", t_iter1)
SNR = 10*math.log10((args.x_var + args.mean_c) / args.r2)
print('SNR = {} [dB]'.format(SNR))


##########################################################################################
### iteration 2 ###

# Tuning parameters for iteration 2

print('======================================')
# Tuning parameter
print('Tuning parameter (iteration2):')
print('r_tuning = {}'.format(r_tuning))
print('convergence_threshold = {}'.format(args.convergence_threshold))
# Dataset
print('new search area: pred +/- {} * RMSE'.format(next_iter_std_mult))
print('# sample points of r = {}'.format(args.m_r))
print('# sample points of theta = {}'.format(args.m_theta))

start = time.time()
# Initialization
r_positions_iter2 = torch.zeros(samples_run, args.m_r, dtype=torch.float, device=device)
theta_positions_iter2 = torch.zeros(samples_run, args.m_theta, dtype=torch.float, device=device)
x_pred_iter2 = torch.zeros(samples_run, m, dtype=torch.cfloat, device=device)
EM_steps_iter2 = torch.zeros(samples_run, dtype=torch.int, device=device)
pred_positions_iter2 = torch.zeros(samples_run, args.k, 2, dtype=torch.float, device=device)

for i in range(samples_run):
    ### New dictionaries ###
    # New search area
    args.position_gt_rleft_bound = pred_positions[i,0,0] - next_iter_std_mult * RMSE_r
    args.position_gt_rright_bound = pred_positions[i,0,0] + next_iter_std_mult * RMSE_r
    args.position_gt_thetaleft_bound = pred_positions[i,0,1]*180/math.pi - next_iter_std_mult * RMSE_theta # degree
    args.position_gt_thetaright_bound = pred_positions[i,0,1]*180/math.pi + next_iter_std_mult * RMSE_theta # degree  
    # Generate new dictionary matrix A_dic, and corresponding hypothesis positions (r, theta)   
    generator_iter2 = DataGenerator(args)
    A_dic, r_positions_iter2[i], theta_positions_iter2[i] = generator_iter2.dictionary_matrix_rtheta()
    ### NUV-SSR ###
    x_pred_iter2[i], EM_steps_iter2[i] = NUV_SSR(args, A_dic, y_mean[i], r_tuning, m)
    print ('EM steps = {}'.format(EM_steps_iter2[i]))


# de-flatten x_pred [sample, m_r*m_theta] -> [sample, m_r, m_theta]
x_pred_2D_iter2 = utils.batch_de_flatten(x_pred_iter2, args.m_r, args.m_theta)

# find peaks [sample, k, 3]
peak_indices_iter2 = utils.batch_peak_finding_2D(x_pred_2D_iter2, args.k)

# convert to positions [sample, k, 2]
for i in range(samples_run):
    pred_positions_iter2[i] = utils.convert_to_positions(peak_indices_iter2[i], torch.squeeze(r_positions_iter2[i]), theta_positions_iter2[i])

end = time.time()
t_iter2 = end - start

# convert to xy coordinates
pred_positions_xy_iter2 = utils.batch_polar_to_cartesian(pred_positions_iter2)
# compute RMSEs
squared_diffs_xy_iter2 = utils.batched_permuted_SquareDiff_2D(pred_positions_xy_iter2, gt_positions_xy)
RMSE_distance_iter2 = utils.RMSE_distance_error(squared_diffs_xy_iter2)
squared_diffs_polar_iter2 = utils.batched_permuted_SquareDiff_2D(pred_positions_iter2, gt_positions)
RMSE_r_iter2, RMSE_theta_iter2 = utils.RMSE_AxisWise_error(squared_diffs_polar_iter2)
RMSE_theta_iter2 = RMSE_theta_iter2 * 180 / math.pi

print('Results (iteration2):')
print('average EM steps = {}'.format(torch.mean(EM_steps_iter2.float())))
print('averaged RMSE distance = {} [m]'.format(RMSE_distance_iter2))
print('averaged RMSE r = {} [m]'.format(RMSE_r_iter2))
print('averaged RMSE theta = {} [deg]'.format(RMSE_theta_iter2))
# Print Run Time
print("Total Run Time:", t_iter2)

#######################
### Save for MATLAB ###
#######################
# Save in a .mat file
y_train_np = y_train.cpu().numpy()
y_noiseless_np = y_noiseless.cpu().numpy()

gt_positions_rtheta_np = gt_positions.cpu().numpy()
pred_positions_rtheta_iter1_np = pred_positions.cpu().numpy()
pred_positions_rtheta_iter2_np = pred_positions_iter2.cpu().numpy()

r_positions_iter1 = r_positions.cpu().numpy()
theta_positions_iter1 = theta_positions.cpu().numpy()
r_positions_iter2 = r_positions_iter2.cpu().numpy()
theta_positions_iter2 = theta_positions_iter2.cpu().numpy()

spectrum_2D = torch.abs(x_pred_2D)
spectrum_2D_np = spectrum_2D.cpu().numpy()
spectrum_2D_iter2 = torch.abs(x_pred_2D_iter2)
spectrum_2D_np_iter2 = spectrum_2D_iter2.cpu().numpy()

RMSE_distance = RMSE_distance.cpu().numpy()
RMSE_distance_iter2 = RMSE_distance_iter2.cpu().numpy()
RMSE_r = RMSE_r.cpu().numpy()
RMSE_r_iter2 = RMSE_r_iter2.cpu().numpy()
RMSE_theta = RMSE_theta.cpu().numpy()
RMSE_theta_iter2 = RMSE_theta_iter2.cpu().numpy()

scipy.io.savemat(data_folder+matlab_file_name, 
        {'y_train': y_train_np,
        'y_noiseless': y_noiseless_np,

        'gt_positions_rtheta': gt_positions_rtheta_np, 
        'pred_positions_rtheta': pred_positions_rtheta_iter1_np, 
        'pred_positions_rtheta_iter2': pred_positions_rtheta_iter2_np, 
        
        'r_positions': r_positions_iter1,
        'theta_positions': theta_positions_iter1,
        'r_positions_iter2': r_positions_iter2,
        'theta_positions_iter2': theta_positions_iter2,

        'spectrum_2D': spectrum_2D_np,
        'spectrum_2D_iter2': spectrum_2D_np_iter2,
        
        'RMSE_distance': RMSE_distance,
        'RMSE_distance_iter2': RMSE_distance_iter2,
        'RMSE_r': RMSE_r,
        'RMSE_r_iter2': RMSE_r_iter2,
        'RMSE_theta': RMSE_theta,
        'RMSE_theta_iter2': RMSE_theta_iter2})

