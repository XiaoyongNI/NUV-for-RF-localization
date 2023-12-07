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
data_file_name = 'data_polar_n32.pt'
matlab_file_name = 'result_polar_gridsearch1.mat'

# Tuning parameters
args.m_r = 10
args.m_theta = 91
m = args.m_r * args.m_theta # total num of hypotheses
args.n = 32

# dataset settings
args.sample = 10 # number of samples
samples_run = args.sample
args.on_grid = False # gt positions are on grid or not
args.plot_grid = True # plot grid or not
args.position_gt_rleft_bound = 500
args.position_gt_rright_bound = 550
args.position_gt_thetaleft_bound = 45
args.position_gt_thetaright_bound = 135

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
RMSE_distance = utils.RMSE_distance_error(squared_diffs_xy)
squared_diffs_polar = utils.batched_permuted_SquareDiff_2D(pred_positions, gt_positions)
RMSE_r, RMSE_theta = utils.RMSE_AxisWise_error(squared_diffs_polar)
RMSE_theta = RMSE_theta * 180 / math.pi

print('Results (grid search):')
print('averaged RMSE distance = {} [m]'.format(RMSE_distance))
print('averaged RMSE r = {} [m]'.format(RMSE_r))
print('averaged RMSE theta = {} [deg]'.format(RMSE_theta))
# Print Run Time
print('Run Time/sample= {} [sec]'.format(t_GridSearch_persample))
if args.coherent_source:
    SNR = 10*math.log10((args.mean_c) / args.r2)
else:
    SNR = 10*math.log10((args.x_var + args.mean_c) / args.r2)
print('SNR = {} [dB]'.format(SNR))

#######################
### Save for MATLAB ###
#######################
# Save in a .mat file
y_train_np = y_train.cpu().numpy()
y_noiseless_np = y_noiseless.cpu().numpy()

gt_positions_rtheta_np = gt_positions.cpu().numpy()
pred_positions_rtheta_np = pred_positions.cpu().numpy()

r_positions = r_positions.cpu().numpy()
theta_positions = theta_positions.cpu().numpy()

# de-flatten distances [sample, m_r*m_theta] -> [sample, m_r, m_theta]
distances_2D = utils.batch_de_flatten(distances, args.m_r, args.m_theta)
spectrum_2D_np = distances_2D.cpu().numpy()
RMSE_distance = RMSE_distance.cpu().numpy()
RMSE_r = RMSE_r.cpu().numpy()
RMSE_theta = RMSE_theta.cpu().numpy()
scipy.io.savemat(data_folder+matlab_file_name, 
                 {'gt_positions_rtheta': gt_positions_rtheta_np, 
                  'pred_positions_rtheta': pred_positions_rtheta_np, 

                  'y_train': y_train_np,
                  'y_noiseless': y_noiseless_np,

                  'spectrum_2D': spectrum_2D_np,
                  'r_positions': r_positions,
                  'theta_positions': theta_positions,

                  'RMSE_distance': RMSE_distance,
                  'RMSE_r': RMSE_r,
                  'RMSE_theta': RMSE_theta})

