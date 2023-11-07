import torch
from torch.nn.functional import pad

k = 3
spectrum_1 = torch.tensor([[100, 2, 300], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
spectrum_2 = torch.tensor([[100, 2, 300], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
spectrum = torch.stack([spectrum_1, spectrum_2], dim=0)
# Pad the map with one element on each side with '-inf' values
padded_map = pad(spectrum, (1, 1, 1, 1), mode='constant', value=float('-inf'))
print(padded_map)
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
peak_values = torch.where(peaks_mask, center, torch.full_like(center, float('-inf')))

# Now you want to get the top k values across the entire batch
# Since you want to preserve the batch dimension, you use flatten(start_dim=1)
flat_peak_values = peak_values.flatten(start_dim=1)

# Perform topk to get the k largest elements and their indices in the flattened dimension
topk_values, topk_flat_indices = torch.topk(flat_peak_values, k, dim=1)

# Convert the flat indices back to 2D indices
peak_indices_2D = (topk_flat_indices // peak_values.size(2), topk_flat_indices % peak_values.size(2))

# Prepare the 2D indices for each batch
peak_batch_indices = torch.arange(0, spectrum.size(0)).view(-1, 1).expand(-1, k)

# Combine the batch indices with the 2D indices
peak_indices = torch.stack((peak_batch_indices, peak_indices_2D[0], peak_indices_2D[1]), dim=2)





