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
matlab_file_name = 'result_polar_default.mat'
# Tuning parameters
args.q_init = 0.01
r_tuning = 1
args.m_r = 10
args.m_theta = 91
m = args.m_r * args.m_theta # total num of hypotheses
args.convergence_threshold = 4e-4
# dataset settings
args.sample = 10 # number of samples
samples_run = args.sample
args.on_grid = False # gt positions are on grid or not
args.plot_grid = True # plot grid or not

#### Generate data ####
generator = DataGenerator(args)
gt_positions, x_true, y_train, y_noiseless = generator.generate_experiment_data_rtheta()
torch.save([gt_positions, x_true, y_train, y_noiseless], data_folder+data_file_name)
[gt_positions, x_true, y_train, y_noiseless] = torch.load(data_folder+data_file_name, map_location=device)
# generate dictionary matrix A_dic, and corresponding hypothesis positions (r, theta)
A_dic, r_positions, theta_positions = generator.dictionary_matrix_rtheta() 
y_mean = y_train.mean(dim=1) # generate y_mean by averaging l snapshots for each sample

#### estimation ####
start = time.time()
print('======================================')
# Tuning parameter
print('Tuning parameter:')
print('r_tuning = {}'.format(r_tuning))
print('max EM steps = {}'.format(args.max_EM_steps))
print('convergence_threshold = {}'.format(args.convergence_threshold))
# Dataset
print('r range = [{}, {}]'.format(args.position_gt_rleft_bound, args.position_gt_rright_bound))
print('theta range = [{}, {}] deg'.format(args.position_gt_thetaleft_bound, args.position_gt_thetaright_bound))
print('# sample points of r = {}'.format(args.m_r))
print('# sample points of theta = {}'.format(args.m_theta))
# initialize
x_pred = torch.zeros(samples_run, args.m_r*args.m_theta, dtype=torch.cfloat, device=device)
EM_steps = torch.zeros(samples_run, dtype=torch.int, device=device)

# NUV-SSR 
for i in range(samples_run):
    x_pred[i], EM_steps[i] = NUV_SSR(args, A_dic, y_mean[i], r_tuning, m)   
    print ('EM steps = {}'.format(EM_steps[i]))
print('average EM steps = {}'.format(torch.mean(EM_steps.float())))

# de-flatten x_pred [sample, m_r*m_theta] -> [sample, m_r, m_theta]
x_pred_2D = utils.batch_de_flatten(x_pred, args.m_r, args.m_theta)

# find peaks [sample, k, 2]
peak_indices = utils.batch_peak_finding_2D(x_pred_2D, args.k)

# convert to positions [sample, k, 2]
pred_positions = utils.batch_convert_to_positions(peak_indices, r_positions, theta_positions)

end = time.time()
t = end - start

# convert to xy coordinates
pred_positions_xy = utils.batch_polar_to_cartesian(pred_positions)
gt_positions_xy = utils.batch_polar_to_cartesian(gt_positions)

# compute RMSEs
squared_diffs_xy = utils.batched_permuted_SquareDiff_2D(pred_positions_xy, gt_positions_xy) 
RMSE_distance = utils.RMSE_distance_error(squared_diffs_xy)
squared_diffs_polar = utils.batched_permuted_SquareDiff_2D(pred_positions, gt_positions)
RMSE_r, RMSE_theta = utils.RMSE_AxisWise_error(squared_diffs_polar)
RMSE_theta = RMSE_theta * 180 / math.pi

print('averaged RMSE distance = {} [m]'.format(RMSE_distance))
print('averaged RMSE r = {} [m]'.format(RMSE_r))
print('averaged RMSE theta = {} [deg]'.format(RMSE_theta))

# Print Run Time
print("Total Run Time:", t)
if args.coherent_source:
    SNR = 10*math.log10((args.mean_c) / args.r2)
else:
    SNR = 10*math.log10((args.x_var + args.mean_c) / args.r2)
print('SNR = {} [dB]'.format(SNR))

# #### plotting ####
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import numpy as np
# ### suppose k = 1 ###
## plot predict positions
# data in polar coordinates (radius, angle in radians)
torch.save([gt_positions, pred_positions], plot_folder+'positions.pt')
[gt_positions, pred_positions] = torch.load(plot_folder+'positions.pt', map_location=device)
r_gt = torch.squeeze(gt_positions[:, :, 0],1).cpu().numpy()
theta_gt = torch.squeeze(gt_positions[:, :, 1],1).cpu().numpy()
r_pred = torch.squeeze(pred_positions[:, :, 0],1).cpu().numpy()
theta_pred = torch.squeeze(pred_positions[:, :, 1],1).cpu().numpy()

# # Create a polar subplot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# Plot the first set of points using circles
ax.plot(theta_gt, r_gt, 'o', label='GT')
# Plot the second set of points using asterisks
ax.plot(theta_pred, r_pred, '*', label='Pred')
# draw the search range
theta_l = args.position_gt_thetaleft_bound * math.pi / 180
theta_r = args.position_gt_thetaright_bound * math.pi / 180
# Generate theta values
theta = np.linspace(theta_l, theta_r, 100)
# Create an array with repeated values of radius bounds
r1 = np.full_like(theta, args.position_gt_rleft_bound)
r2 = np.full_like(theta, args.position_gt_rright_bound)
# Plot the search range as a filled area
ax.fill_between(theta, r1, r2, color='green', alpha=0.5, label='Search Range')
# Add a legend 
ax.legend(loc='lower left')
# save figure
fig.savefig(plot_folder+'positions.png')


## plot spectrum
spectrum_2D = torch.abs(x_pred_2D)
torch.save(spectrum_2D, plot_folder+'spectrum.pt')
spectrum_2D = torch.load(plot_folder+'spectrum.pt', map_location='cpu')
r_positions = r_positions.cpu().numpy()
theta_positions = theta_positions.cpu().numpy()
# Choose which batch item to plot, here we select the first item
batch_index = 0
data_to_plot = spectrum_2D[batch_index, :, :].numpy()
# Create a meshgrid for the radius and theta arrays
Theta, R = np.meshgrid(theta_positions, r_positions, indexing='xy')
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
# gt positions and pred positions
# r_gt = gt_positions[batch_index, :, 0].cpu().numpy() 
# theta_gt = gt_positions[batch_index, :, 1].cpu().numpy()
# r_pred = pred_positions[batch_index, :, 0].cpu().numpy()
# theta_pred = pred_positions[batch_index, :, 1].cpu().numpy()
x_gt = gt_positions_xy[batch_index, :, 0].cpu().numpy() 
y_gt = gt_positions_xy[batch_index, :, 1].cpu().numpy()
x_pred = pred_positions_xy[batch_index, :, 0].cpu().numpy()
y_pred = pred_positions_xy[batch_index, :, 1].cpu().numpy()
# Create the plot
plt.figure()
cmap = plt.cm.viridis
colors = cmap(np.arange(cmap.N))
colors[:, -1] = np.linspace(0.1, 1, cmap.N)  # Start with alpha=0.1 and gradually increase to 1
light_cmap = mcolors.LinearSegmentedColormap.from_list('light_viridis', colors)
plt.pcolormesh(X, Y, data_to_plot, cmap=light_cmap, shading='nearest')  # Use pcolormesh
# Plot ground truth and prediction positions
plt.scatter(x_gt, y_gt, color='b', label='Ground Truth')
plt.scatter(x_pred, y_pred, color='r', label='Prediction')
# Plot the std
circle = Circle((x_pred, y_pred), RMSE_distance, color='r', fill=False, linewidth=0.5)
plt.gca().add_patch(circle)
gt_proxy = Line2D([0], [0], linestyle='none', marker='o', color='b', markerfacecolor='b')
pred_proxy = Line2D([0], [0], linestyle='none', marker='o', color='r', markerfacecolor='r')
circle_proxy = Line2D([0], [0], linestyle="none", marker="o", color='r', markerfacecolor="none", markersize=10, markeredgewidth=2)
# Add the proxy artists to the legend
plt.legend(handles=[gt_proxy, pred_proxy, circle_proxy], labels=['Ground Truth', 'Prediction', 'std'], loc='upper right')
# Add a color bar and label
plt.colorbar()
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
# add grid
# if args.plot_grid:
#     for r_line in R[:,0]:
#         plt.axvline(x=r_line, color='grey', linestyle='--', linewidth=0.5)
#     for theta_line in Theta[0,:]:
#         plt.axhline(y=theta_line, color='grey', linestyle='--', linewidth=0.5)
# Save the figure
plt.savefig(plot_folder+'spectrum.png')
 


########################
### plot 3D spectrum ###
########################
# Start a new figure and add a 3D subplot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot the 3D surface
# cmap = plt.cm.viridis
# colors = cmap(np.arange(cmap.N))
# colors[:, -1] = np.linspace(0.1, 1, cmap.N)  # Start with alpha=0.1 and gradually increase to 1
# light_cmap = mcolors.LinearSegmentedColormap.from_list('light_viridis', colors)
# surface = ax.plot_surface(X, Y, data_to_plot.T, cmap=light_cmap)
# # plot gt positions and pred positions
# r_index_gt = np.argmin(np.abs(r_positions - r_gt[batch_index]))
# theta_index_gt = np.argmin(np.abs(theta_positions - theta_gt[batch_index]))
# r_index_pred = np.argmin(np.abs(r_positions - r_pred[batch_index]))
# theta_index_pred = np.argmin(np.abs(theta_positions - theta_pred[batch_index]))
# z_value_gt = data_to_plot.T[theta_index_gt, r_index_gt]
# z_value_pred = data_to_plot.T[theta_index_pred, r_index_pred]
# ax.scatter([x_gt], [y_gt], [z_value_gt], color='r', s=50, label='Ground Truth')
# ax.scatter([x_pred], [y_pred], [z_value_pred], color='b', s=50, label='Predicted')
# # add a legend
# ax.legend(loc='upper left')
# # add a color bar
# cbar = fig.colorbar(surface, shrink=0.5, aspect=5)
# cbar.set_label('Spectrum height')
# # Set labels for axes
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Spectrum')
# # save figure
# fig.savefig(plot_folder+'spectrum.png')


#######################
### Save for MATLAB ###
#######################
# Save in a .mat file
gt_positions_rtheta_np = gt_positions.cpu().numpy()
pred_positions_rtheta_np = pred_positions.cpu().numpy()

y_train_np = y_train.cpu().numpy() 
y_noiseless_np = y_noiseless.cpu().numpy()  

spectrum_2D_np = spectrum_2D.cpu().numpy()
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