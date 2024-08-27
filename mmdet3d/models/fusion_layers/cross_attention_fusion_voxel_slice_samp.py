import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from ..builder import FUSION_LAYERS
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, norm_cfg=None):
        super(ConvBNReLU, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.bn = build_norm_layer(norm_cfg, out_planes)[1]
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class SliceSampDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_cfg=dict(type='BN', requires_grad=True)):
        super(SliceSampDownsample, self).__init__()
        # Depthwise convolution (groups=in_channels makes it depthwise)
        self.depthwise_conv = nn.Conv2d(in_channels*4, in_channels*4, kernel_size=kernel_size, groups=in_channels*4, padding=kernel_size//2, bias=False)
        self.depthwise_bn = build_norm_layer(norm_cfg, in_channels*4)[1]
        self.pointwise_conv = nn.Conv2d(in_channels*4, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.gelu = nn.GELU()

    def forward(self, X):
        # Step 1: Slice the input feature map
        X_slice = torch.cat([
            X[..., ::2, ::2],  # Upper-left corner
            X[..., 1::2, ::2],  # Upper-right corner
            X[..., ::2, 1::2],  # Lower-left corner
            X[..., 1::2, 1::2]  # Lower-right corner
        ], dim=1)
        
        # Step 2: Depthwise Separable Convolution
        X_depthwise = self.depthwise_conv(X_slice)
        X_depthwise = self.depthwise_bn(X_depthwise)
        X_depthwise = self.gelu(X_depthwise)
        
        # Step 3: Pointwise Convolution
        X_pointwise = self.pointwise_conv(X_depthwise)
        X_pointwise = self.pointwise_bn(X_pointwise)
        output = self.gelu(X_pointwise)
        
        return output
    
class SliceUpsamp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_cfg=dict(type='BN', requires_grad=True)):
        super(SliceUpsamp, self).__init__()
        # Depthwise convolution (groups=in_channels makes it depthwise)
        self.depthwise_conv = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=kernel_size, groups=in_channels // 4, padding=kernel_size // 2, bias=False)
        self.depthwise_bn = build_norm_layer(norm_cfg, in_channels // 4)[1]
        self.pointwise_conv = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.gelu = nn.GELU()

    def forward(self, X):
        B, C, H, W = X.shape
        assert C % 4 == 0, "Number of input channels must be divisible by 4"
        new_C = C // 4
        
        # Step 1: Inverted slicing
        X_reshaped = X.view(B, new_C, 4, H, W)
        X_reorganized = X_reshaped.permute(0, 1, 3, 4, 2).contiguous()
        X_upsampled = X_reorganized.view(B, new_C, 2*H, 2*W)
        
        # Step 2: Depthwise Separable Convolution
        X_depthwise = self.depthwise_conv(X_upsampled)
        X_depthwise = self.depthwise_bn(X_depthwise)
        X_depthwise = self.gelu(X_depthwise)
        
        X_pointwise = self.pointwise_conv(X_depthwise)
        X_pointwise = self.pointwise_bn(X_pointwise)
        output = self.gelu(X_pointwise)
        
        return output
        


class Decoder(nn.Module):

    def __init__(self, d_model = 256, hidden_dim = 512, num_heads = 8, dropout = 0.1, show_weights=False) -> None:
        super(Decoder,self).__init__()
        
        self.show_weights = show_weights
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_key = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForwardBlock(d_model=d_model, hidden_dim= hidden_dim, dropout=dropout)

        self.multiheadAttention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

    def forward(self, query, key):

        query = self.norm_query(query)
        key = self.norm_key(key)

        value = key

        attention_output, attn_weights = self.multiheadAttention(query=query, key=key, value=value, need_weights=self.show_weights)
        if self.show_weights:
            self.vis_attention_scores(attn_weights)
            #self.vis_mean_attention_scores(attn_weights)

        add_norm_0 = torch.add(attention_output, query)
        

        ff_output = self.ff(add_norm_0)
        output = torch.add(ff_output,add_norm_0)
        
        return output
    
    def vis_mean_attention_scores(self, weights):
        

        attention_heatmaps = weights.squeeze(0).cpu().detach().numpy()

        # Compute the mean attention heatmap over the specified range
        mean_attention_heatmap = np.mean(attention_heatmaps, axis=0)

        # Reshape the mean attention heatmap to 2D
        mean_attention_heatmap_2d = mean_attention_heatmap.reshape((64, 64)).T

        # Plot the mean attention heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(mean_attention_heatmap_2d, cmap='viridis', interpolation='nearest')
        plt.xlabel('Wide Image Patch X-Axis')
        plt.ylabel('Wide Image Patch Y-Axis')
        plt.title(f'Mean Attention Heatmap for Tokens')
        plt.colorbar(label='Attention Score')
        plt.show()


    def vis_attention_scores(self, weights):
         
        attention_heatmaps = weights.squeeze(0).cpu().detach().numpy()  # Remove batch dimension and convert to NumPy

        # #Create plot objects outside the loop
        fig_heatmap, axs_heatmap = plt.subplots(2)

        # Precompute highlighted grid outside the loop
        highlighted_grid = np.zeros((64, 64))

        for i in range(1000, 1200):
            # Clear previous plot
            axs_heatmap[0].clear()
            axs_heatmap[1].clear()
            
            # Reset previously highlighted point
            if i > 0:
                prev_i = i - 1
                prev_row_index = prev_i // 64
                prev_col_index = prev_i % 64
                highlighted_grid[prev_row_index, prev_col_index] = 0
            
            # Update highlighted grid
            row_index = i // 64
            col_index = i % 64
            highlighted_grid[row_index, col_index] = 1
            
            # Plot attention heatmap
            attention_heatmap = attention_heatmaps[i]
            attention_heatmap_2d = attention_heatmap.reshape((64, 64)).T
            
            axs_heatmap[0].imshow(attention_heatmap_2d, cmap='viridis', interpolation='nearest')
            axs_heatmap[0].set_xlabel('Wide Image Patch X-Axis')
            axs_heatmap[0].set_ylabel('Wide Image Patch Y-Axis')
            axs_heatmap[0].set_title(f'Attention Heatmap for Target Token {i}')
            
            # Plot highlighted grid
            axs_heatmap[1].imshow(highlighted_grid, cmap='viridis', interpolation='nearest')
            axs_heatmap[1].set_xlabel('Column Index')
            axs_heatmap[1].set_ylabel('Row Index')
            axs_heatmap[1].set_title(f'Position of Token {i} in 64x64 Grid')
            
            # Update the plots
            fig_heatmap.canvas.draw()
            plt.pause(0.2)  # Adjust the delay time as needed

        # Keep the plot windows open
        plt.show()

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model = 256, hidden_dim = 128, dropout = 0.1) -> None:
        super(FeedForwardBlock,self).__init__()

        self.linear_1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.GELU()
        self.relu_2 = nn.GELU()

        self.norm = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):

        return self.norm_2(self.dropout_2(self.relu_2(self.linear_2(self.dropout(self.relu(self.linear_1(self.norm(x))))))))

class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, norm_cfg=None):
        super(ConvBNReLU, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.bn = build_norm_layer(norm_cfg, out_planes)[1]
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class ConvTransposeBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, norm_cfg=None):
        super(ConvTransposeBNReLU, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.bn = build_norm_layer(norm_cfg, out_planes)[1]
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

"""
Same as V3 but adjusted to fit sparse resnet output. Upsample 4x compared to 2x
"""
@FUSION_LAYERS.register_module()
class MultiHeadCrossAttentionVoxelSliceSamp(nn.Module):
    def __init__(self, embed_dim = 512, num_heads=8, dropout = 0.1, out_channels=512, norm_cfg=None):
        super(MultiHeadCrossAttentionVoxelSliceSamp, self).__init__()

        self.embed_dim = embed_dim

        self.norm_cfg = norm_cfg
        self.out_channels = out_channels

        self.reduce_camera_spatialy = SliceSampDownsample(1536, self.embed_dim, kernel_size=3, norm_cfg=self.norm_cfg)
        self.reduce_camera_spatialy_conv1 = ConvBNReLU(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1, norm_cfg = self.norm_cfg)
        self.reduce_camera_spatialy_2 = SliceSampDownsample(self.embed_dim, self.embed_dim, kernel_size=3, norm_cfg=self.norm_cfg)


        self.reduce_lidar_spatially = SliceSampDownsample(512, self.embed_dim,norm_cfg = self.norm_cfg)
        self.lidar_conv_0 = ConvBNReLU(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1,norm_cfg = self.norm_cfg)


        self.lidar_camera_cross_attention = Decoder(self.embed_dim, hidden_dim=self.embed_dim, num_heads=num_heads, dropout=dropout, show_weights=False)
        
        self.pos_embed_camera = nn.Parameter(torch.randn(1, self.embed_dim, 4096) * .02) #done as in ViT: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py, (14 (image hight) * 25 image width * 6 images) / 16 (image patches)
        self.pos_embed_lidar = nn.Parameter(torch.randn(1, self.embed_dim, 8100) * .02) #done as in ViT: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py, no reduction for now

        self.upsample_layer = SliceUpsamp(embed_dim, self.embed_dim, norm_cfg = self.norm_cfg)



    def create_lidar_patches(self, lidar_tensor):
        
        #flatten all patches:   
        lidar_tensor = lidar_tensor.view(lidar_tensor.shape[0], lidar_tensor.shape[1], -1)

        lidar_tensor = torch.add(lidar_tensor, self.pos_embed_lidar)

        #reshape to match required nn.MultiheadAttention input format: batch, seq, feature
        lidar_tensor = lidar_tensor.permute(0, 2, 1)  # shape: (batch_size, sequence_length, embedding_dimension)


        return lidar_tensor
    
    def create_camera_patches(self, camera_tensor):
        
        #flatten all patches:
        camera_patches = camera_tensor.view(camera_tensor.shape[0], camera_tensor.shape[1], -1)

        #add position embedding to flattened patches
        camera_patches = torch.add(camera_patches, self.pos_embed_camera)

        # if self.pos_embed_camera.grad is not None:
        #    print(f"camera.grad: {self.pos_embed_camera.grad.abs().max()}")
           

           
        #reshape to match required nn.MultiheadAttention input format: batch, seq, feature
        camera_patches = camera_patches.permute(0, 2, 1)  # shape: (batch_size, sequence_length, embedding_dimension)

        return camera_patches
    
    def forward(self, lidar_bev_features, camera_bev_features):

        lidar_bev_features = lidar_bev_features[0]
        camera_bev_features = camera_bev_features[0]

        camera_bev_features = self.reduce_camera_spatialy(camera_bev_features)
        camera_bev_features = self.reduce_camera_spatialy_conv1(camera_bev_features)
        camera_bev_features = self.reduce_camera_spatialy_2(camera_bev_features)

        reduced_lidar_bev_features = self.reduce_lidar_spatially(lidar_bev_features)
        reduced_lidar_bev_features = self.lidar_conv_0(reduced_lidar_bev_features) #180x180


        # # get patch embeddings
        image_patch_embedding = self.create_camera_patches(camera_bev_features)
        lidar_patch_embedding = self.create_lidar_patches(reduced_lidar_bev_features) #90x90 -> 8100


        cross_attention = self.lidar_camera_cross_attention(lidar_patch_embedding, image_patch_embedding)

        # Reshape the 1d tensor back to a 2d representation used in the CenterHead
        cross_attention = cross_attention.permute(0,2,1)
        cross_attention = cross_attention.view(cross_attention.shape[0], cross_attention.shape[1], 90, 90)  # Shape: [batch * 6, 256, 64, 64]

        cross_attention = torch.add(cross_attention, reduced_lidar_bev_features)

        upsampled_once = self.upsample_layer(cross_attention)

        #residual around fusion
        output = torch.add(upsampled_once, lidar_bev_features)

        return [output]
