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
data_file_name = 'data_polar_n16_test.pt'
matlab_file_name = 'result_polar_gridsearch1.mat'

# Tuning parameters
args.m_r = 11 # number of sample points of r
args.m_theta = 91 # number of sample points of theta
m = args.m_r * args.m_theta # total num of hypotheses
args.n = 16 # number of antennas

# dataset settings
args.sample = 100 # number of samples
samples_run = args.sample
args.on_grid = False # gt positions are on grid or not
args.position_gt_rleft_bound = 500
args.position_gt_rright_bound = 550
args.position_gt_thetaleft_bound = 45
args.position_gt_thetaright_bound = 135
rleft_bound_iter1 = args.position_gt_rleft_bound
rright_bound_iter1 = args.position_gt_rright_bound
thetaleft_bound_iter1 = args.position_gt_thetaleft_bound
thetaright_bound_iter1 = args.position_gt_thetaright_bound

#### Generate data ####
generator_iter1 = DataGenerator(args)
# gt_positions, x_true, y_train, y_noiseless = generator_iter1.generate_experiment_data_rtheta()
# torch.save([gt_positions, x_true, y_train, y_noiseless], data_folder+data_file_name)
[gt_positions, x_true, y_train, y_noiseless] = torch.load(data_folder+data_file_name, map_location=device)
# generate dictionary matrix A_dic, and corresponding hypothesis positions (r, theta)
A_dic, r_positions, theta_positions = generator_iter1.dictionary_matrix_rtheta() 
y_mean = y_train.mean(dim=1) # generate y_mean by averaging l snapshots for each sample


#### estimation ####
print('======================================')
# Dataset
print('# antennas = {}'.format(args.n))
print('r range = [{}, {}]'.format(args.position_gt_rleft_bound, args.position_gt_rright_bound))
print('theta range = [{}, {}] deg'.format(args.position_gt_thetaleft_bound, args.position_gt_thetaright_bound))
print('# sample points of r = {}'.format(args.m_r))
print('# sample points of theta = {}'.format(args.m_theta))

# initialize
pred_positions = torch.zeros(samples_run, args.k, 2, dtype=torch.float32, device=device)
distances = torch.zeros(samples_run, m, dtype=torch.float32, device=device)
min_distance_index = torch.zeros(samples_run, dtype=torch.int32, device=device)

start = time.time()

for i in range(samples_run):
#    print('sample {}/{}'.format(i+1, samples_run))
   # match hypothesis
   min_distance_index[i], distances[i] = utils.match_hypothesis(A_dic, y_mean[i])
   
end = time.time()
t_GridSearch = end - start
t_GridSearch_persample = t_GridSearch / samples_run

for i in range(samples_run):
   # get hypothesis position
   r_value, theta_value = utils.get_hypothesis_position(min_distance_index[i], r_positions, theta_positions)
   pred_positions[i, 0, :] = torch.tensor([r_value, theta_value], dtype=torch.float32, device=device)
#    print('matched position = [{}, {}], gt position = [{}, {}]'.format(r_value, theta_value, gt_positions[i, 0, 0], gt_positions[i, 0, 1]))

# convert to xy coordinates
pred_positions_xy = utils.batch_polar_to_cartesian(pred_positions)
gt_positions_xy = utils.batch_polar_to_cartesian(gt_positions)

# compute RMSEs
squared_diffs_xy = utils.batched_permuted_SquareDiff_2D(pred_positions_xy, gt_positions_xy) 
RMSE_distance, Empirical_variance_distance = utils.RMSE_distance_error(squared_diffs_xy)
squared_diffs_polar = utils.batched_permuted_SquareDiff_2D(pred_positions, gt_positions)
RMSE_r, RMSE_theta, Empirical_variance_r, Empirical_variance_theta = utils.RMSE_AxisWise_error(squared_diffs_polar)
RMSE_theta = RMSE_theta * 180 / math.pi
Empirical_variance_theta = Empirical_variance_theta * 180 / math.pi

# compute "spectrum"
distances_2D = utils.batch_de_flatten(distances, args.m_r, args.m_theta)
spectrum_2D_np = distances_2D.cpu().numpy()

print('Results (grid search):')
print('RMSE distance = {} [m]'.format(RMSE_distance))
print('empirical variance of distance = {} [m]'.format(Empirical_variance_distance))
print('RMSE r = {} [m]'.format(RMSE_r))
print('empirical variance of r = {} [m]'.format(Empirical_variance_r))
print('RMSE theta = {} [deg]'.format(RMSE_theta))
print('empirical variance of theta = {} [deg]'.format(Empirical_variance_theta))

# Print Run Time
print('Run Time/sample= {} [sec]'.format(t_GridSearch_persample))
if args.coherent_source:
    SNR = 10*math.log10((args.mean_c) / args.r2)
else:
    SNR = 10*math.log10((args.x_var + args.mean_c) / args.r2)
print('SNR = {} [dB]'.format(SNR))


##########################################################################################
### iteration 2 ###
# Tuning parameters for iteration 2
args.m_r = 11
args.m_theta = 91
m = args.m_r * args.m_theta # total num of hypotheses
next_iter_std_mult_r = 3
next_iter_std_mult_theta = 3

print('======================================')
# Dataset
print('new search area: pred R +/- {} * RMSE_R'.format(next_iter_std_mult_r))
print('new search area: pred theta +/- {} * RMSE_theta'.format(next_iter_std_mult_theta))
print('# sample points of r = {}'.format(args.m_r))
print('# sample points of theta = {}'.format(args.m_theta))

# initialize
pred_positions_iter2 = torch.zeros(samples_run, args.k, 2, dtype=torch.float32, device=device)
distances_iter2 = torch.zeros(samples_run, m, dtype=torch.float32, device=device)
min_distance_index_iter2 = torch.zeros(samples_run, dtype=torch.int32, device=device)
r_positions_iter2 = torch.zeros(samples_run, args.m_r, dtype=torch.float, device=device)
theta_positions_iter2 = torch.zeros(samples_run, args.m_theta, dtype=torch.float, device=device)


start = time.time()

for i in range(samples_run):
   ### New dictionaries ###
   # New search area
   args.position_gt_rleft_bound = max(pred_positions[i, 0, 0] - next_iter_std_mult_r * RMSE_r, rleft_bound_iter1)
   args.position_gt_rright_bound = min(pred_positions[i,0,0] + next_iter_std_mult_r * RMSE_r, rright_bound_iter1)
   args.position_gt_thetaleft_bound = max(pred_positions[i,0,1]*180/math.pi - next_iter_std_mult_theta * RMSE_theta, thetaleft_bound_iter1) # degree
   args.position_gt_thetaright_bound = min(pred_positions[i,0,1]*180/math.pi + next_iter_std_mult_theta * RMSE_theta, thetaright_bound_iter1) # degree  
   # Generate new dictionary matrix A_dic, and corresponding hypothesis positions (r, theta)   
   generator_iter2 = DataGenerator(args)
   A_dic, r_positions_iter2[i], theta_positions_iter2[i] = generator_iter2.dictionary_matrix_rtheta()
    
   min_distance_index_iter2[i], distances_iter2[i] = utils.match_hypothesis(A_dic, y_mean[i])
   
end = time.time()
t_GridSearch_iter2 = end - start
t_GridSearch_persample_iter2 = t_GridSearch_iter2 / samples_run

for i in range(samples_run):
   # get hypothesis position
   r_value, theta_value = utils.get_hypothesis_position(min_distance_index_iter2[i], r_positions_iter2[i], theta_positions_iter2[i])
   pred_positions_iter2[i, 0, :] = torch.tensor([r_value, theta_value], dtype=torch.float32, device=device)

# convert to xy coordinates
pred_positions_xy_iter2 = utils.batch_polar_to_cartesian(pred_positions_iter2)

# compute RMSEs
squared_diffs_xy_iter2 = utils.batched_permuted_SquareDiff_2D(pred_positions_xy_iter2, gt_positions_xy) 
RMSE_distance_iter2, Empirical_variance_distance_iter2 = utils.RMSE_distance_error(squared_diffs_xy_iter2)
squared_diffs_polar_iter2 = utils.batched_permuted_SquareDiff_2D(pred_positions_iter2, gt_positions)
RMSE_r_iter2, RMSE_theta_iter2, Empirical_variance_r_iter2, Empirical_variance_theta_iter2 = utils.RMSE_AxisWise_error(squared_diffs_polar_iter2)
RMSE_theta_iter2 = RMSE_theta_iter2 * 180 / math.pi
Empirical_variance_theta_iter2 = Empirical_variance_theta_iter2 * 180 / math.pi

print('Results (grid search iteration2):')
print('RMSE distance = {} [m]'.format(RMSE_distance_iter2))
print('empirical variance of distance = {} [m]'.format(Empirical_variance_distance_iter2))
print('RMSE r = {} [m]'.format(RMSE_r_iter2))
print('empirical variance of r = {} [m]'.format(Empirical_variance_r_iter2))
print('RMSE theta = {} [deg]'.format(RMSE_theta_iter2))
print('empirical variance of theta = {} [deg]'.format(Empirical_variance_theta_iter2))

# Print Run Time
print('Run Time/sample= {} [sec]'.format(t_GridSearch_persample_iter2))



##########################################################################################
### iteration 3 ###
# Tuning parameters for iteration 3
args.m_r = 11
args.m_theta = 91
m = args.m_r * args.m_theta # total num of hypotheses
next_iter_std_mult_r = 3
next_iter_std_mult_theta = 3

print('======================================')
# Dataset
print('new search area: pred R +/- {} * RMSE_R'.format(next_iter_std_mult_r))
print('new search area: pred theta +/- {} * RMSE_theta'.format(next_iter_std_mult_theta))
print('# sample points of r = {}'.format(args.m_r))
print('# sample points of theta = {}'.format(args.m_theta))

# initialize
pred_positions_iter3 = torch.zeros(samples_run, args.k, 2, dtype=torch.float32, device=device)
distances_iter3 = torch.zeros(samples_run, m, dtype=torch.float32, device=device)
min_distance_index_iter3 = torch.zeros(samples_run, dtype=torch.int32, device=device)
r_positions_iter3 = torch.zeros(samples_run, args.m_r, dtype=torch.float, device=device)
theta_positions_iter3 = torch.zeros(samples_run, args.m_theta, dtype=torch.float, device=device)

start = time.time()

for i in range(samples_run):
   ### New dictionaries ###
   # New search area
   args.position_gt_rleft_bound = max(pred_positions_iter2[i, 0, 0] - next_iter_std_mult_r * RMSE_r_iter2, rleft_bound_iter1)
   args.position_gt_rright_bound = min(pred_positions_iter2[i,0,0] + next_iter_std_mult_r * RMSE_r_iter2, rright_bound_iter1)
   args.position_gt_thetaleft_bound = max(pred_positions_iter2[i,0,1]*180/math.pi - next_iter_std_mult_theta * RMSE_theta_iter2, thetaleft_bound_iter1) # degree
   args.position_gt_thetaright_bound = min(pred_positions_iter2[i,0,1]*180/math.pi + next_iter_std_mult_theta * RMSE_theta_iter2, thetaright_bound_iter1) # degree  
   # Generate new dictionary matrix A_dic, and corresponding hypothesis positions (r, theta)   
   generator_iter3 = DataGenerator(args)
   A_dic, r_positions_iter3[i], theta_positions_iter3[i] = generator_iter3.dictionary_matrix_rtheta()
    
   min_distance_index_iter3[i], distances_iter3[i] = utils.match_hypothesis(A_dic, y_mean[i])
   
end = time.time()
t_GridSearch_iter3 = end - start
t_GridSearch_persample_iter3 = t_GridSearch_iter3 / samples_run

for i in range(samples_run):
   # get hypothesis position
   r_value, theta_value = utils.get_hypothesis_position(min_distance_index_iter3[i], r_positions_iter3[i], theta_positions_iter3[i])
   pred_positions_iter3[i, 0, :] = torch.tensor([r_value, theta_value], dtype=torch.float32, device=device)

# convert to xy coordinates
pred_positions_xy_iter3 = utils.batch_polar_to_cartesian(pred_positions_iter3)

# compute RMSEs
squared_diffs_xy_iter3 = utils.batched_permuted_SquareDiff_2D(pred_positions_xy_iter3, gt_positions_xy) 
RMSE_distance_iter3, Empirical_variance_distance_iter3 = utils.RMSE_distance_error(squared_diffs_xy_iter3)
squared_diffs_polar_iter3 = utils.batched_permuted_SquareDiff_2D(pred_positions_iter3, gt_positions)
RMSE_r_iter3, RMSE_theta_iter3,Empirical_variance_r_iter3,Empirical_variance_theta_iter3 = utils.RMSE_AxisWise_error(squared_diffs_polar_iter3)
RMSE_theta_iter3 = RMSE_theta_iter3 * 180 / math.pi
Empirical_variance_theta_iter3 = Empirical_variance_theta_iter3 * 180 / math.pi

print('Results (grid search iteration3):')
print('RMSE distance = {} [m]'.format(RMSE_distance_iter3))
print('empirical variance of distance = {} [m]'.format(Empirical_variance_distance_iter3))
print('RMSE r = {} [m]'.format(RMSE_r_iter3))
print('empirical variance of r = {} [m]'.format(Empirical_variance_r_iter3))
print('RMSE theta = {} [deg]'.format(RMSE_theta_iter3))
print('empirical variance of theta = {} [deg]'.format(Empirical_variance_theta_iter3))
# Print Run Time
print('Run Time/sample= {} [sec]'.format(t_GridSearch_persample_iter3))


#######################
### Save for MATLAB ###
#######################
# Save in a .mat file
# y_train_np = y_train.cpu().numpy()
# y_noiseless_np = y_noiseless.cpu().numpy()

# gt_positions_rtheta_np = gt_positions.cpu().numpy()
# pred_positions_rtheta_np = pred_positions.cpu().numpy()
# pred_positions_rtheta_np_iter2 = pred_positions_iter2.cpu().numpy()

# r_positions = r_positions.cpu().numpy()
# theta_positions = theta_positions.cpu().numpy()
# r_positions_iter2 = r_positions_iter2.cpu().numpy()
# theta_positions_iter2 = theta_positions_iter2.cpu().numpy()

# distances_iter2_2D = utils.batch_de_flatten(distances_iter2, args.m_r, args.m_theta)
# spectrum_2D_np_iter2 = distances_iter2_2D.cpu().numpy()
# RMSE_distance = RMSE_distance.cpu().numpy()
# RMSE_distance_iter2 = RMSE_distance_iter2.cpu().numpy()
# RMSE_r = RMSE_r.cpu().numpy()
# RMSE_r_iter2 = RMSE_r_iter2.cpu().numpy()
# RMSE_theta = RMSE_theta.cpu().numpy()
# RMSE_theta_iter2 = RMSE_theta_iter2.cpu().numpy()

# scipy.io.savemat(data_folder+matlab_file_name, 
#                  {'gt_positions_rtheta': gt_positions_rtheta_np, 
#                   'pred_positions_rtheta': pred_positions_rtheta_np, 
#                   'pred_positions_rtheta_iter2': pred_positions_rtheta_np_iter2,

#                   'y_train': y_train_np,
#                   'y_noiseless': y_noiseless_np,

#                   'spectrum_2D': spectrum_2D_np,
#                   'spectrum_2D_iter2': spectrum_2D_np_iter2,

#                   'r_positions': r_positions,
#                   'theta_positions': theta_positions,
#                   'r_positions_iter2': r_positions_iter2,
#                   'theta_positions_iter2': theta_positions_iter2,

#                   'RMSE_distance': RMSE_distance,
#                   'RMSE_distance_iter2': RMSE_distance_iter2,
#                   'RMSE_r': RMSE_r,
#                   'RMSE_r_iter2': RMSE_r_iter2,
#                   'RMSE_theta': RMSE_theta,
#                   'RMSE_theta_iter2': RMSE_theta_iter2})

