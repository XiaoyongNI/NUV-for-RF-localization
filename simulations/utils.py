import torch
from scipy.signal import find_peaks
import itertools 
from torch.nn.functional import pad

# find peaks 1D (known k)
def peak_finding(u, k):
  """
  input: u, tensor of size [m]
         k, number of output peaks
  output: indices of the first k highest peaks
  """
  spectrum = torch.abs(u)
  # Find all peaks and their properties
  peaks, _ = find_peaks(spectrum)

  # If fewer peaks are found than k, raise an error
  if len(peaks) < k:
    raise ValueError('Fewer peaks found than k')
  
  # Get the heights from the properties dictionary
  peak_heights = spectrum[peaks]
  # Get the indices that would sort the peak heights in descending order
  _, peak_height_indices = torch.topk(peak_heights, k)
  # Get the indices of the first k highest peaks
  peak_indices = peaks[peak_height_indices]

  return peak_indices

# batch version of peak finding 1D (known k)
def batch_peak_finding(u, k):
    """
    input: u, tensor of size [batch size, m]
           k, number of output peaks
    output: indices of the first k highest peaks [batch size, k]
    """
    spectrum = torch.abs(u)

    # Take the difference to find where the derivative changes sign
    diff = spectrum[:, 1:] - spectrum[:, :-1]
    
    # Find peaks (where the difference changes from positive to negative)
    peaks = (diff[:, :-1] > 0) & (diff[:, 1:] < 0)

    # Check if any batch has fewer than k peaks
    peaks_count = peaks.sum(dim=1)
    if torch.any(peaks_count < k):
        batch_ids = torch.nonzero(peaks_count < k, as_tuple=True)[0]
        raise ValueError(f'Fewer peaks found than k for batches: {batch_ids.tolist()}')

    # Adjust spectrum to match the shape of peaks for the subsequent masking
    spectrum_adjusted = spectrum[:, 1:-1]

    # Mask out the non-peak values with very negative numbers so they don't interfere with topk
    masked_spectrum = torch.where(peaks, spectrum_adjusted, torch.tensor(float('-inf'), device=u.device))

    # Find the top k peaks for each batch; values give the heights, indices give the locations
    values, batched_peak_indices = masked_spectrum.topk(k, dim=1)

    # Adjust indices to account for the shifted spectrum
    batched_peak_indices += 1

    return batched_peak_indices

# find peaks 2D (known k)
def peak_finding_2D(u, k):
  """
  input: u, 2D complex tensor of size [m_r, m_theta]
          k, number of output peaks
  output: indices of the first k highest peaks [k, 2]
  """
  spectrum = torch.abs(u)
  # Pad the map with one element on each side with '-inf' values
  padded_map = pad(spectrum, (1, 1, 1, 1), mode='constant', value=float('-inf'))
  # centers and 8 neighbors
  center = padded_map[1:-1, 1:-1]
  top = padded_map[:-2, 1:-1]
  bottom = padded_map[2:, 1:-1]
  left = padded_map[1:-1, :-2]
  right = padded_map[1:-1, 2:]
  top_left = padded_map[:-2, :-2]
  top_right = padded_map[:-2, 2:]
  bottom_left = padded_map[2:, :-2]
  bottom_right = padded_map[2:, 2:]

  # Check for local peaks - each element must be greater than its neighborhood
  peaks_mask = (
      (center > top) &
      (center > bottom) &
      (center > left) &
      (center > right) &
      (center > top_left) &
      (center > top_right) &
      (center > bottom_left) &
      (center > bottom_right) 
  )
  # Use the mask to select the indices of the peaks
  peak_indices = peaks_mask.nonzero(as_tuple=False)
  # Select the peak values using these indices
  peak_values = spectrum[peak_indices[:, 0], peak_indices[:, 1]]

  # Sort the peak values and select top k
  top_values, top_indices = torch.topk(peak_values, k)

  # Find the corresponding indices in the spectrum for the top k values
  top_peak_indices = peak_indices[top_indices] # (k, 2)

  return top_peak_indices

# batch version of peak finding 2D (known k)
def batch_peak_finding_2D(u, k):
  """
  input: u, complex tensor of size [batch_size, m_r, m_theta]
          k, number of output peaks
  output: indices of the first k highest peaks [batch_size, k, 2]
  """
  device = u.device
  spectrum = torch.abs(u)
  # Pad the map with one element on each side with '-inf' values
  padded_map = pad(spectrum, (1, 1, 1, 1), mode='constant', value=float('-inf')).to(device)
  # centers and 8 neighbors
  center = padded_map[:, 1:-1, 1:-1]
  top = padded_map[:, :-2, 1:-1]
  bottom = padded_map[:, 2:, 1:-1]
  left = padded_map[:, 1:-1, :-2]
  right = padded_map[:, 1:-1, 2:]
  top_left = padded_map[:, :-2, :-2]
  top_right = padded_map[:, :-2, 2:]
  bottom_left = padded_map[:, 2:, :-2]
  bottom_right = padded_map[:, 2:, 2:]

  peaks_mask = (
      (center > top) &
      (center > bottom) &
      (center > left) &
      (center > right) &
      (center > top_left) &
      (center > top_right) &
      (center > bottom_left) &
      (center > bottom_right) 
  )

  # Create a tensor to hold the peak values
  peak_values = torch.where(peaks_mask, center, torch.full_like(center, float('-inf'), device=device))

  # Now you want to get the top k values across the entire batch
  # Since you want to preserve the batch dimension, you use flatten(start_dim=1)
  flat_peak_values = peak_values.flatten(start_dim=1)

  # Perform topk to get the k largest elements and their indices in the flattened dimension
  topk_values, topk_flat_indices = torch.topk(flat_peak_values, k, dim=1)

  # Convert the flat indices back to 2D indices
  peak_indices_2D = (torch.div(topk_flat_indices, peak_values.size(2), rounding_mode='floor'), topk_flat_indices % peak_values.size(2))

  # Prepare the 2D indices for each batch
  peak_batch_indices = torch.arange(0, spectrum.size(0), device=device).reshape(-1, 1).expand(-1, k)

  # Combine the batch indices with the 2D indices
  peak_indices = torch.stack((peak_batch_indices, peak_indices_2D[0], peak_indices_2D[1]), dim=2)

  return peak_indices

# de-flatten, convert 1D spectrum back to 2D
def de_flatten(spectrum, m_x1, m_x2):  
  """
  Use polar coordinates as an example, cartesian coordinates are similar
  make sure the traversal order of (r, theta) is the same as dictionary matrix
  input: spectrum, tensor of size [m_r * m_theta]
          m_r, number of grids in r
          m_theta, number of grids in theta
  output: spectrum, tensor of size [m_r, m_theta]
  """
  spectrum = spectrum.reshape(m_x1, m_x2)
  return spectrum

# batch version of de-flatten
def batch_de_flatten(spectrum, m_x1, m_x2):
  """
  Use polar coordinates as an example, cartesian coordinates are similar
  input: spectrum, tensor of size [batch_size, m_r * m_theta]
          m_r, number of grids in r
          m_theta, number of grids in theta
  output: spectrum, tensor of size [batch_size, m_r, m_theta]
  """
  spectrum = spectrum.reshape(-1, m_x1, m_x2)
  return spectrum


 
# Convert from indices to DoA
def convert_to_doa(peak_indices, centers):
    """
    input: peak_indices, array of size [k]
           centers,
    output: doa, tensor of size [k]
    """
    doa = centers[peak_indices]
    doa.sort()
    return doa
   
# batch version of Convert from indices to DoA
def batch_convert_to_doa(peak_indices, m):
  """
  input: peak_indices, tensor of size [batch_size, k]
          m, number of grids
  output: doa, tensor of size [batch_size, k]
  """
  # Convert peak indices to doa
  doa = peak_indices * (180 / m) - 90

  # Sort each batch
  doa, _ = torch.sort(doa, dim=1)

  return doa

# Convert from indices to 2D positions
def convert_to_positions(peak_indices, x1_positions, x2_positions):
  """
  input: peak_indices, array of size [k, 2], (x1, x2)
          x1_positions, tensor of size [m_r or m_x]
          theta_positions, tensor of size [m_theta or m_y] 
  output: positions, tensor of size [k, 2]
  """
  pred_x1 = x1_positions[peak_indices[:, 0]]
  pred_x2 = x2_positions[peak_indices[:, 1]]
  pred_x1 = pred_x1.unsqueeze(1)
  pred_x2 = pred_x2.unsqueeze(1)
  positions = torch.cat((pred_x1, pred_x2), dim=1)

  return positions
 
# batch version of Convert from indices to 2D positions
def batch_convert_to_positions(peak_indices, x1_positions, x2_positions):
  """
  input: peak_indices, tensor of size [batch_size, k, 3], (sample_id, x1, x2)
          x1_positions, tensor of size [m_r or m_x]
          theta_positions, tensor of size [m_theta or m_y] 
  output: positions, tensor of size [batch_size, k, 2]
  """
  pred_x1 = x1_positions[peak_indices[:, :, 1]]
  pred_x2 = x2_positions[peak_indices[:, :, 2]]
  pred_x1 = pred_x1.unsqueeze(2)
  pred_x2 = pred_x2.unsqueeze(2)
  positions = torch.cat((pred_x1, pred_x2), dim=2)

  return positions

# batch version of Convert from polar to cartesian coordinates
def batch_polar_to_cartesian(positions):
  """
  input: positions, tensor of size [batch_size, k, 2], (r, theta)
  output: positions, tensor of size [batch_size, k, 2], (x, y)
  """
  pred_r = positions[:, :, 0]
  pred_theta = positions[:, :, 1]
  pred_x = pred_r * torch.cos(pred_theta)
  pred_y = pred_r * torch.sin(pred_theta)
  pred_x = pred_x.unsqueeze(2)
  pred_y = pred_y.unsqueeze(2)
  positions = torch.cat((pred_x, pred_y), dim=2)

  return positions

# permuted MSE computation
def permuted_mse_1D(pred, DOA):
    """
    input: pred, tensor of size [k]
           DOA, tensor of size [k]
    output: permuted MSE
    """
    # Step 1: Generate all possible permutations of indices [0, 1, ..., k-1]
    perms = all_permutations(len(pred))  # [k!, k]
    
    # Step 2: Use these permutations to reorder pred
    permuted_preds = pred[perms]  # [k!, k]

    # Step 3: Compute the MSE for each permutation
    mse = torch.mean((permuted_preds - DOA)**2, dim=-1)  # [k!]

    # Step 4: Find the minimum MSE
    min_mse, _ = torch.min(mse, dim=-1)  # []

    return min_mse

def all_permutations(length):
    """Return all permutations of a sequence of given length."""
    return torch.tensor(list(itertools.permutations(range(length))), dtype=torch.long)
# batch version of permuted MSE computation
def batched_permuted_mse_1D(pred, DOA):
    batch_size, k = pred.shape
    device = pred.device
    
    # Step 1: Generate all possible permutations of indices [0, 1, ..., k-1]
    perms = all_permutations(k).to(device)  # [k!, k]
    # Expand perms to match batch_size
    perms = perms.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, k!, k]
    
    # Step 2: Use these permutations to reorder each sample in the batch
    # Expand dims for broadcasting with permutations
    expanded_pred = pred.unsqueeze(1)  # [batch_size, 1, k]
    expanded_pred = expanded_pred.repeat(1, perms.shape[1], 1)  # [batch_size, k!, k]
    
    # Gather results according to permutations
    permuted_preds = torch.gather(expanded_pred, 2, perms)  # [batch_size, k!, k]

    # Step 3: Compute the MSE for each permutation of each sample
    mse = torch.mean((permuted_preds - DOA.unsqueeze(1))**2, dim=-1)  # [batch_size, k!]

    # Step 4: Find the minimum MSE for each sample
    min_mse, _ = torch.min(mse, dim=-1)  # [batch_size]

    return min_mse

# permuted MSE 2D

# batch version of permuted MSE 2D
def batched_permuted_SquareDiff_2D(pred, gt):
  """
  input: pred, tensor of size [batch_size, k, 2]
          gt, tensor of size [batch_size, k, 2]
  output: permuted MSE
  """
  batch_size, k, _ = pred.shape
  device = pred.device
  
  # Step 1: Generate all possible permutations of indices [0, 1, ..., k-1]
  perms = all_permutations(k).to(device)  # [k!, k]
  # Expand perms to match batch_size
  perms = perms.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, k!, k]
  
  # Step 2: Use these permutations to reorder each sample in the batch
  # Expand dims for broadcasting with permutations
  expanded_pred = pred.unsqueeze(1)  # [batch_size, 1, k, 2]
  expanded_pred = expanded_pred.repeat(1, perms.shape[1], 1, 1)  # [batch_size, k!, k, 2]
  
  # Gather results according to permutations
  permuted_preds = torch.gather(expanded_pred, 2, perms.unsqueeze(-1).repeat(1, 1, 1, 2))  # [batch_size, k!, k, 2]

  # Step 3: Compute the MSE for each permutation of each sample
  # Expand gt for broadcasting
  expanded_gt = gt.unsqueeze(1).expand(-1, perms.shape[1], -1, -1)  # [batch_size, k!, k, 2]
  squared_diffs = (permuted_preds - expanded_gt)**2

  return squared_diffs

def RMSE_distance_error(squared_diffs):
  """
  input: squared_diffs, tensor of size [batch_size, k!, k, 2] in cartesian coordinates
  output: RMSE, one value
  """

  # mean over k targets and 2 axis(x, y or r, theta)
  mse = torch.mean(squared_diffs, dim=(-2, -1))  # [batch_size, k!]

  # Find the minimum MSE for each sample
  min_mse, _ = torch.min(mse, dim=1)  # [batch_size]

  min_mse = min_mse * 2 # since we want distance error, no need to average over x and y or r and theta
  RMSE_error = torch.sqrt(min_mse)
  mean_RMSE_error = torch.mean(RMSE_error) # mean RMSE over all samples

  return mean_RMSE_error

def RMSE_AxisWise_error(squared_diffs):
  """
  Compute RMSE for each axis (x, y or r, theta)
  input: squared_diffs, tensor of size [batch_size, k!, k, 2]
  output: RMSE_axis1, RMSE_axis2, two values
  """
  # mean over k targets
  mse = torch.mean(squared_diffs, dim=2)  # [batch_size, k!, 2]
  # separate x and y or r and theta
  mse_axis1 = mse[:, :, 0] # [batch_size, k!]
  mse_axis2 = mse[:, :, 1] # [batch_size, k!]
  # Find the minimum MSE for each sample
  min_mse_axis1, _ = torch.min(mse_axis1, dim=1)  # [batch_size]
  min_mse_axis2, _ = torch.min(mse_axis2, dim=1)  # [batch_size]
  # RMSE for each axis
  RMSE_axis1 = torch.sqrt(min_mse_axis1)
  RMSE_axis2 = torch.sqrt(min_mse_axis2)
  mean_RMSE_axis1 = torch.mean(RMSE_axis1)
  mean_RMSE_axis2 = torch.mean(RMSE_axis2)

  return mean_RMSE_axis1, mean_RMSE_axis2
