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
# path names
plot_folder = 'simulations/plots/'
data_folder = 'data/'
data_file_name = 'default_sample=10.pt'

# Tuning parameters
args.q_init = 0.01
r_tuning = 10
args.m_x = 201
args.m_y = 201
m = args.m_x * args.m_y # total num of hypotheses
args.convergence_threshold = 1e-4
# dataset settings
args.sample = 10 # number of samples
samples_run = args.sample
args.on_grid = False # gt positions are on grid or not
args.plot_grid = True # plot grid or not

#### Generate data ####
generator = DataGenerator(args)
gt_positions, x_true, y_train = generator.generate_experiment_data_xy()
torch.save([gt_positions, x_true, y_train], data_folder+data_file_name)
[gt_positions, x_true, y_train] = torch.load(data_folder+data_file_name, map_location=device)
# generate dictionary matrix A_dic, and corresponding hypothesis positions 
A_dic, x_positions, y_positions = generator.dictionary_matrix_xy() 
y_mean = y_train.mean(dim=1) # generate y_mean by averaging l snapshots for each sample

#### estimation ####
start = time.time()
print('======================================')
# Tuning parameter
print('Tuning parameter:')
print('r_tuning = {}'.format(r_tuning))
print('max EM steps = {}'.format(args.max_EM_steps))
print('convergence_threshold = {}'.format(args.convergence_threshold))
print('q init = {}'.format(args.q_init))
# Dataset
print('x range = [{}, {}]'.format(args.position_gt_xleft_bound, args.position_gt_xright_bound))
print('y range = [{}, {}]'.format(args.position_gt_yleft_bound, args.position_gt_yright_bound))
print('# sample points of x = {}'.format(args.m_x))
print('# sample points of y = {}'.format(args.m_y))
# initialize
x_pred = torch.zeros(samples_run, args.m_x*args.m_y, dtype=torch.cfloat, device=device)
iterations = torch.zeros(samples_run, dtype=torch.int, device=device)

# NUV-SSR 
for i in range(samples_run):
    x_pred[i], iterations[i] = NUV_SSR(args, A_dic, y_mean[i], r_tuning, m)   
    print ('EM steps = {}'.format(iterations[i]))
print('average EM steps = {}'.format(torch.mean(iterations.float())))

x_pred_abs = torch.abs(x_pred) 

# de-flatten x_pred [sample, m_x*m_y] -> [sample, m_x, m_y]
x_pred_2D = utils.batch_de_flatten(x_pred, args.m_x, args.m_y)

# find peaks [sample, k, 2]
peak_indices = utils.batch_peak_finding_2D(x_pred_2D, args.k)

# convert to positions [sample, k, 2]
pred_positions = utils.batch_convert_to_positions(peak_indices, x_positions, y_positions)

end = time.time()
t = end - start

# compute RMSEs
squared_diffs_xy = utils.batched_permuted_SquareDiff_2D(pred_positions, gt_positions) 
RMSE_distance, Empirical_variance_distance = utils.RMSE_distance_error(squared_diffs_xy)
RMSE_x, RMSE_y,Empirical_variance_x,Empirical_variance_y = utils.RMSE_AxisWise_error(squared_diffs_xy)

print('RMSE distance = {} [m]'.format(RMSE_distance))
print('Empirical variance of distance = {} [m]'.format(Empirical_variance_distance))
print('RMSE x = {} [m]'.format(RMSE_x))
print('empirical variance of x = {} [m]'.format(Empirical_variance_x))
print('RMSE y = {} [m]'.format(RMSE_y))
print('empirical variance of y = {} [m]'.format(Empirical_variance_y))



# Print Run Time
print("Total Run Time:", t)
if args.coherent_source:
    SNR = 10*math.log10((args.mean_c) / args.r2)
else:
    SNR = 10*math.log10((args.x_var + args.mean_c) / args.r2)
print('SNR = {}'.format(SNR))

# #### plotting ####
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np
# # ### suppose k = 1 ###

# ## plot predict positions
# # data in polar coordinates (radius, angle in radians)
# torch.save([gt_positions, pred_positions], plot_folder+'positions.pt')
# [gt_positions, pred_positions] = torch.load(plot_folder+'positions.pt', map_location=device)
# x_gt = torch.squeeze(gt_positions[:, :, 0],1).cpu().numpy()
# y_gt = torch.squeeze(gt_positions[:, :, 1],1).cpu().numpy()
# x_pred = torch.squeeze(pred_positions[:, :, 0],1).cpu().numpy()
# y_pred = torch.squeeze(pred_positions[:, :, 1],1).cpu().numpy()

# fig, ax = plt.subplots()
# # Plot the first set of points using circles
# ax.plot(x_gt, y_gt, 'o', label='GT')
# # Plot the second set of points using asterisks
# ax.plot(x_pred, y_pred, '*', label='Pred')
# # draw the search range
# x_left_bound = args.position_gt_xleft_bound
# x_right_bound = args.position_gt_xright_bound
# y_left_bound = args.position_gt_yleft_bound
# y_right_bound = args.position_gt_yright_bound
# # Plot the search range as a filled area
# ax.fill_between([x_left_bound, x_right_bound], [y_left_bound, y_left_bound], [y_right_bound, y_right_bound], color='g', alpha=0.1)
# # Add a legend 
# ax.legend(loc='lower left')
# # save figure
# fig.savefig(plot_folder+'positions.png')

# ########################
# ### plot 2D spectrum ###
# ########################
# spectrum_2D = torch.abs(x_pred_2D)
# torch.save(spectrum_2D, plot_folder+'spectrum.pt')
# spectrum_2D = torch.load(plot_folder+'spectrum.pt', map_location='cpu')
# x_positions = x_positions.cpu().numpy()
# y_positions = y_positions.cpu().numpy()
# # Choose which batch item to plot, here we select the first item
# batch_index = 0
# data_to_plot = spectrum_2D[batch_index, :, :].numpy()
# # Create a meshgrid for the radius and theta arrays
# Y, X = np.meshgrid(y_positions, x_positions, indexing='xy')
# # gt positions and pred positions
# x_gt = gt_positions[batch_index, :, 0].cpu().numpy() 
# y_gt = gt_positions[batch_index, :, 1].cpu().numpy()
# x_pred = pred_positions[batch_index, :, 0].cpu().numpy()
# y_pred = pred_positions[batch_index, :, 1].cpu().numpy()
# # Create the plot
# plt.figure()
# cmap = plt.cm.viridis
# colors = cmap(np.arange(cmap.N))
# colors[:, -1] = np.linspace(0.1, 1, cmap.N)  # Start with alpha=0.1 and gradually increase to 1
# light_cmap = mcolors.LinearSegmentedColormap.from_list('light_viridis', colors)
# plt.pcolormesh(X, Y, data_to_plot, cmap=light_cmap, shading='nearest')  # Use pcolormesh

# # Plot ground truth and prediction positions
# plt.scatter(x_gt, y_gt, color='b', label='Ground Truth')
# plt.scatter(x_pred, y_pred, color='r', label='Prediction')
# # Add a color bar and legend
# plt.colorbar()
# plt.legend()
# plt.xlabel('X')
# plt.ylabel('Y')
# # add grid
# if args.plot_grid:
#     for x_line in X[:,0]:
#         plt.axvline(x=x_line, color='grey', linestyle='--', linewidth=0.5)
#     for y_line in Y[0,:]:
#         plt.axhline(y=y_line, color='grey', linestyle='--', linewidth=0.5)
# # Save the figure
# plt.savefig(plot_folder+'spectrum.png')

########################
### plot 3D spectrum ###
########################
# Plot discrete spectrum or continuous spectrum
# args.plot_discrete_spectrum = True
# spectrum_2D = torch.abs(x_pred_2D)
# torch.save(spectrum_2D, 'simulations/spectrum.pt')
# spectrum_2D = torch.load('simulations/spectrum.pt', map_location='cpu')
# x_positions = x_positions.cpu().numpy()
# y_positions = y_positions.cpu().numpy()
# # Choose which batch item to plot, here we select the first item
# batch_index = 0
# data_to_plot = spectrum_2D[batch_index, :, :].numpy()
# # Create a meshgrid 
# X, Y = np.meshgrid(x_positions, y_positions)
# # Start a new figure and add a 3D subplot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # plot gt positions and pred positions
# x_gt = gt_positions[batch_index, :, 0].cpu().numpy() 
# y_gt = gt_positions[batch_index, :, 1].cpu().numpy()
# x_pred = pred_positions[batch_index, :, 0].cpu().numpy()
# y_pred = pred_positions[batch_index, :, 1].cpu().numpy()
# x_index_gt = np.argmin(np.abs(x_positions - x_gt))
# y_index_gt = np.argmin(np.abs(y_positions - y_gt))
# x_index_pred = np.argmin(np.abs(x_positions - x_pred))
# y_index_pred = np.argmin(np.abs(y_positions - y_pred))
# z_value_gt = data_to_plot[x_index_gt, y_index_gt]
# z_value_pred = data_to_plot[x_index_pred, y_index_pred]
# ax.scatter([x_gt], [y_gt], [z_value_gt], color='r', s=50, label='Ground Truth')
# ax.scatter([x_pred], [y_pred], [z_value_pred], color='b', s=50, label='Predicted')


# if args.plot_discrete_spectrum == False:
#     ### Option1: Plot the 3D surface (continuous)
#     cmap = plt.cm.viridis
#     colors = cmap(np.arange(cmap.N))
#     colors[:, -1] = np.linspace(0.1, 1, cmap.N)  # Start with alpha=0.1 and gradually increase to 1
#     light_cmap = mcolors.LinearSegmentedColormap.from_list('light_viridis', colors)
#     surface = ax.plot_surface(X, Y, data_to_plot.T, cmap=light_cmap)
#     # add a color bar
#     cbar = fig.colorbar(surface, shrink=0.5, aspect=5)
#     cbar.set_label('Spectrum height')
# else:
#     ### Option2: Plot the 3D surface (discrete)
#     cmap = plt.cm.viridis
#     # Normalize data_to_plot for coloring
#     norm = plt.Normalize(data_to_plot.min(), data_to_plot.max())
#     colors = cmap(norm(data_to_plot.T))
#     # Flatten the X, Y, and Z data
#     X_flat = X.flatten()
#     Y_flat = Y.flatten()
#     Z_flat = data_to_plot.T.flatten()
#     # Flatten or appropriately slice the colors array
#     colors_flat = colors.reshape(-1, 4)  # Assuming colors is a 2D array with 4 columns for RGBA
#     # Plot the 3D discrete spectrum points
#     ax.scatter(X_flat, Y_flat, Z_flat, c=colors_flat, marker='o', alpha=0.6)
#     # Add a color bar for the spectrum
#     mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
#     cbar = fig.colorbar(mappable, shrink=0.5, aspect=5)
#     cbar.set_label('Spectrum height')

# # add a legend
# ax.legend(loc='upper left')
# # Set labels for axes
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Spectrum')
# # save figure
# fig.savefig('simulations/spectrum.png')
