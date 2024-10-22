import torch

def check_layers_in_checkpoint(checkpoint_path):
    """
    Prints out the layers that have weights included in the .pth checkpoint.

    Args:
        checkpoint_path (str): Path to the .pth checkpoint file.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract the state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Print the layers (keys) in the state dict
    print("Layers with weights in the checkpoint:")
    for key in state_dict.keys():
        print(key)

# Example usage
checkpoint_path = '/media/tom/Volume/master_thesis/Fast-BEV-Fusion/workdirs/fast_bev_fusion_box2d_centerhead_pretrained/epoch_20_lidar_pretrained.pth'
check_layers_in_checkpoint(checkpoint_path)