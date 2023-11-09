##########################################################
################### simulating NUV-SSR ###################
##########################################################

import torch
import math
import time

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
# number of samples
args.sample = 2
samples_run = args.sample
# searching the best performed tuning parameter r (std of observation noise)
r_t = [1e-0]

#### Generate data ####
generator = DataGenerator(args)
gt_positions, x_true, y_train = generator.generate_experiment_data_rtheta()
torch.save([gt_positions, x_true, y_train], 'data/Vanilla_default_sample=10.pt')
[gt_positions, x_true, y_train] = torch.load('data/Vanilla_default_sample=10.pt', map_location=device)
# generate dictionary matrix A_dic, and corresponding hypothesis positions (r, theta)
A_dic, r_positions, theta_positions = generator.dictionary_matrix_rtheta() 
A_dic = A_dic.to(device)
r_positions = r_positions.to(device)
theta_positions = theta_positions.to(device)
y_mean = y_train.mean(dim=1) # generate y_mean by averaging l snapshots for each sample

# #### estimation ####
start = time.time()
for r_tuning in r_t:
    print('======================================')
    print('r_tuning = {}'.format(r_tuning))
    # initialize
    x_pred = torch.zeros(samples_run, args.m_r*args.m_theta, dtype=torch.cfloat, device=device)
    iterations = torch.zeros(samples_run, dtype=torch.int, device=device)
    
    # NUV-SSR 
    for i in range(samples_run):
        x_pred[i], iterations[i] = NUV_SSR(args, A_dic, y_mean[i], r_tuning)

    # de-flatten x_pred [sample, m_r*m_theta] -> [sample, m_r, m_theta]
    x_pred_2D = utils.batch_de_flatten(x_pred, args.m_r, args.m_theta)
    
    # find peaks [sample, k, 2]
    peak_indices = utils.batch_peak_finding_2D(x_pred_2D, args.k)
    
    # convert to positions [sample, k, 2]
    pred_positions = utils.batch_convert_to_positions(peak_indices, r_positions, theta_positions)
    
    # compute MSE
    MSE = utils.batched_permuted_mse_2D(pred_positions, gt_positions) # mean square error for all samples   
  
    mean_MSE = torch.mean(MSE)
    MSE_dB = 10 * (torch.log10(mean_MSE))
    print('averaged MSE in dB = {}'.format(MSE_dB))
    MSE_linear = torch.sqrt(mean_MSE)
    print('averaged RMSE in linear = {}'.format(MSE_linear))
    print('--------------------------------------------')

end = time.time()
t = end - start
# Print Run Time
print("Total Run Time:", t)
SNR = 10*math.log10((args.x_var + args.mean_c) / args.r2)
print('SNR = {}'.format(SNR))

#### plotting ####
import matplotlib.pyplot as plt

### suppose k = 1 ###
# data in polar coordinates (radius, angle in radians)
r_gt = torch.squeeze(gt_positions[:, :, 0]).cpu().numpy()
theta_gt = torch.squeeze(gt_positions[:, :, 1]).cpu().numpy()
r_pred = torch.squeeze(pred_positions[:, :, 0]).cpu().numpy()
theta_pred = torch.squeeze(pred_positions[:, :, 1]).cpu().numpy()

# Create a polar subplot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Plot the first set of points using circles
ax.plot(theta_gt, r_gt, 'o', label='GT')

# Plot the second set of points using asterisks
ax.plot(theta_gt, r_gt, '*', label='Pred')

# Add a legend in the upper left corner of the plot
ax.legend(loc='upper left')

# save figure
fig.savefig('simulations/Vanilla_default_sample=10.png')