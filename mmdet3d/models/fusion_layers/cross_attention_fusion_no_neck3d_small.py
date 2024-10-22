import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from mmcv.cnn import build_norm_layer
from ..builder import FUSION_LAYERS


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, norm_cfg=None):
        super(ConvBNReLU, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.bn = build_norm_layer(norm_cfg, out_planes)[1]
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):

    def __init__(self, d_model = 256, hidden_dim = 512, num_heads = 8, dropout = 0.1, show_weights=False) -> None:
        super(Decoder,self).__init__()
        
        self.show_weights = show_weights
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_key = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)

        #always use at least 0.1 dropout in ffn
        if dropout < 0.1:
            self.ff = FeedForwardBlock(d_model=d_model, hidden_dim= hidden_dim, dropout=0.1)
        else:
            self.ff = FeedForwardBlock(d_model=d_model, hidden_dim= hidden_dim, dropout=dropout)

        self.multiheadAttention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

    def forward(self, query, key):

        query = self.norm_query(query)
        key = self.norm_key(key)

        value = key

        attention_output, attn_weights = self.multiheadAttention(query=query, key=key, value=value, need_weights=self.show_weights)
        if self.show_weights:
            #self.vis_attention_scores(attn_weights)
            self.vis_mean_attention_scores(attn_weights)

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
        plt.figure(figsize=(100, 80))
        plt.imshow(mean_attention_heatmap_2d, cmap='viridis', interpolation='nearest')
        plt.xlabel('Image Features X-Axis')
        plt.ylabel('Image Features Y-Axis')
        plt.title(f'Mean Attention Heatmap for Tokens')
        plt.colorbar(label='Attention Score')
        plt.show()


    def vis_attention_scores(self, weights):
         
        attention_heatmaps = weights.squeeze(0).cpu().detach().numpy()  # Remove batch dimension and convert to NumPy

        # #Create plot objects outside the loop
        fig_heatmap, axs_heatmap = plt.subplots(2)

        # Precompute highlighted grid outside the loop
        highlighted_grid = np.zeros((64, 64))


        #for i in range(200, 500):
        i = 304
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
        self.relu = nn.LeakyReLU()
        self.relu_2 = nn.LeakyReLU()

        self.norm = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):

        return self.norm_2(self.dropout_2(self.relu_2(self.linear_2(self.dropout(self.relu(self.linear_1(self.norm(x))))))))

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=512, norm_cfg = None):
        super().__init__()
        if norm_cfg is not None:
            norm = build_norm_layer(norm_cfg, num_pos_feats)[1]
        else:
            norm =  nn.BatchNorm1d(num_pos_feats)
           
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            norm,
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, x):

        position_embedding = self.position_embedding_head(x)
        return position_embedding


@FUSION_LAYERS.register_module()
class MultiHeadCrossAttentionNoNeckSmall(nn.Module):
    def __init__(self, embed_dim = 512, num_heads=8, dropout = 0.1, in_cam_channels=384, in_lidar_channels=384, output_dim = 384, norm_cfg = None, one_d_norm = None):
        super(MultiHeadCrossAttentionNoNeckSmall, self).__init__()

        self.embed_dim = embed_dim
        self.norm_cfg = norm_cfg

        self.lidar_pos_embed = PositionEmbeddingLearned(embed_dim, embed_dim, one_d_norm)
        self.camera_pos_embed = PositionEmbeddingLearned(embed_dim, embed_dim, one_d_norm)

        self.camera_embedding = ConvBNReLU(in_cam_channels, embed_dim, kernel_size=3, stride=2, padding=1, norm_cfg = self.norm_cfg)

        self.lidar_embedding = ConvBNReLU(in_lidar_channels, embed_dim, kernel_size=1, stride=1, padding=0, norm_cfg = self.norm_cfg)

        self.lidar_camera_cross_attention = Decoder(self.embed_dim, hidden_dim=self.embed_dim, num_heads= num_heads, dropout=dropout, show_weights=False)
        

        self.cross_attention_layer_norm = nn.LayerNorm(self.embed_dim)

        self.out_conv = ConvBNReLU(embed_dim, output_dim, kernel_size=1, stride=1, padding=0, norm_cfg = self.norm_cfg)


    def create_lidar_patches(self, lidar_tensor):
        
        #flatten all patches:   
        lidar_tensor = lidar_tensor.view(lidar_tensor.shape[0], lidar_tensor.shape[1], -1).contiguous()

        lidar_tensor = self.lidar_pos_embed(lidar_tensor)

        #reshape to match required nn.MultiheadAttention input format: batch, seq, feature
        lidar_tensor = lidar_tensor.permute(0, 2, 1).contiguous()  # shape: (batch_size, sequence_length, embedding_dimension)


        return lidar_tensor
    
    def create_camera_patches(self, camera_tensor):
        
        #flatten all patches:
        camera_patches = camera_tensor.view(camera_tensor.shape[0], camera_tensor.shape[1], -1).contiguous()

        camera_patches = self.camera_pos_embed(camera_patches)


        # if self.pos_embed_camera.grad is not None:
        #    print(f"camera.grad: {self.pos_embed_camera.grad.abs().max()}")
           

           
        #reshape to match required nn.MultiheadAttention input format: batch, seq, feature
        camera_patches = camera_patches.permute(0, 2, 1).contiguous()  # shape: (batch_size, sequence_length, embedding_dimension)

        return camera_patches
    
    def forward(self, lidar_bev_features, camera_bev_features):
        
        lidar_bev_features = lidar_bev_features[0]
        camera_bev_features = camera_bev_features[0]
        
        lidar_bev_features = self.lidar_embedding(lidar_bev_features)
        camera_bev_features = self.camera_embedding(camera_bev_features)

        # get patch embeddings
        image_patch_embedding = self.create_camera_patches(camera_bev_features)
        lidar_patch_embedding = self.create_lidar_patches(lidar_bev_features)
        
        #cross-attention
        
        cross_attention = self.lidar_camera_cross_attention(lidar_patch_embedding, image_patch_embedding)
        cross_attention = self.cross_attention_layer_norm(torch.add(cross_attention, lidar_patch_embedding))


        # Reshape the 1d tensor back to a 2d representation used in the CenterHead
        output = cross_attention.permute(0,2,1).contiguous()
        output = output.view(output.shape[0], output.shape[1], lidar_bev_features.shape[-2], lidar_bev_features.shape[-1]).contiguous()  # Shape: [batch * 6, 256, 64, 64]

        output = self.out_conv(output)
        
        return [output]
