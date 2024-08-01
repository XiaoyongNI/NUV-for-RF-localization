import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

# def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y):
#     """Compute 2D Gaussian in Cartesian coordinates."""
#     return np.exp(-(((x - x0)**2 / (2 * sigma_x**2)) + ((y - y0)**2 / (2 * sigma_y**2))))

# # Define meshgrid in polar coordinates
# # r = np.linspace(0, 5, 100)  # Radius
# # theta = np.linspace(0, 2 * np.pi, 100)  # Angle

# # R, Theta = np.meshgrid(r, theta)

# # # Convert polar coordinates to Cartesian for evaluation
# # X = R * np.cos(Theta)
# # Y = R * np.sin(Theta)

# # # Parameters of the Gaussian in Cartesian coordinates
# # x0, y0 = 0, 0  # Center
# # sigma_x, sigma_y = 1, 1  # Standard deviations

# # # Compute the Gaussian
# # Z = gaussian_2d(X, Y, x0, y0, sigma_x, sigma_y)

# # # Plot the heat map
# # plt.figure()
# cmap = plt.cm.viridis
# colors = cmap(np.arange(cmap.N))
# colors[:, -1] = np.linspace(0.1, 1, cmap.N)  # Start with alpha=0.1 and gradually increase to 1
# light_cmap = mcolors.LinearSegmentedColormap.from_list('light_viridis', colors)

# # plt.pcolormesh(R, Theta, Z, cmap=light_cmap)
# # plt.colorbar(label='Intensity')
# # plt.ylabel('Theta (rad)')
# # plt.xlabel('Radius')
# # plt.title('2D Gaussian Distribution in Polar Coordinates')
# # plt.show()


# # Define meshgrid in Cartesian coordinates
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)

# X, Y = np.meshgrid(x, y)

# # Parameters of the Gaussian in Cartesian coordinates
# x0, y0 = 0, 0  # Center
# sigma_x, sigma_y = 1, 1  # Standard deviations

# # Compute the Gaussian
# Z = gaussian_2d(X, Y, x0, y0, sigma_x, sigma_y)

# # Plot the heat map in Cartesian coordinates
# plt.figure()
# plt.pcolormesh(X, Y, Z, cmap=light_cmap)
# plt.colorbar(label='Intensity')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('2D Gaussian Distribution in Cartesian Coordinates')
# plt.show()

# Define a complex number and a float
# complex_num = np.complex(2+3j)
# real_num = 5.0
# # Multiply using standard multiplication
# result = complex_num * real_num

# print(result)


# Convert pt file to mat file
import scipy.io as sio
data_folder = 'data/N16/'
data_file_name_test = 'data_polar_n16_r21e-2_test.pt'
matlab_file_name = 'data_polar_n16_r21e-2_test.mat'
gt_positions, x_true, y_train, y_noiseless = torch.load(data_folder + data_file_name_test)

# Convert PyTorch tensors to NumPy arrays
gt_positions_np = gt_positions.numpy()
x_true_np = x_true.numpy()
y_train_np = y_train.numpy()
y_noiseless_np = y_noiseless.numpy()

# Create a dictionary with the NumPy arrays
matlab_dict = {
    'gt_positions': gt_positions_np,
    'x_true': x_true_np,
    'y_train': y_train_np,
    'y_noiseless': y_noiseless_np
}

# Save the data to a MATLAB file
sio.savemat(data_folder + matlab_file_name, matlab_dict)