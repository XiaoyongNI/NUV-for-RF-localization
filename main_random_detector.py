import torch
import math
import time

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
data_folder = 'data/'
data_file_name = 'data_polar_n16_highnoise.pt'

# Tuning parameters
args.m_r = 11 # number of sample points of r
args.m_theta = 91 # number of sample points of theta
m = args.m_r * args.m_theta # total num of hypotheses
args.n = 16 # number of antennas

# dataset settings
args.sample = 100 # number of samples
args.r2 = 1 # noise variance
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
random_detector_index = torch.zeros(samples_run, dtype=torch.int32, device=device)

start = time.time()


for i in range(samples_run):
   # Random detector
   random_detector_index[i] = utils.random_detector(A_dic)
   
end = time.time()
t_RandomDetector = end - start
t_RandomDetector_persample = t_RandomDetector / samples_run

for i in range(samples_run):
   # get hypothesis position
   r_value, theta_value = utils.get_hypothesis_position(random_detector_index[i], r_positions, theta_positions)
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

print('Results (random detector):')
print('RMSE r = {} [m]'.format(RMSE_r))
print('empirical variance of r = {} [m]'.format(Empirical_variance_r))
print('RMSE theta = {} [deg]'.format(RMSE_theta))
print('empirical variance of theta = {} [deg]'.format(Empirical_variance_theta))
print('RMSE distance = {} [m]'.format(RMSE_distance))
print('empirical variance of distance = {} [m]'.format(Empirical_variance_distance))

# Print Run Time
print('Run Time/sample= {} [sec]'.format(t_RandomDetector_persample))
if args.coherent_source:
    SNR = 10*math.log10((args.mean_c) / args.r2)
else:
    SNR = 10*math.log10((args.x_var + args.mean_c) / args.r2)
print('SNR = {} [dB]'.format(SNR))