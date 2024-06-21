import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from mmcv.runner import auto_fp16
from ..builder import FUSION_LAYERS


class Decoder(nn.Module):

    def __init__(self, d_model = 256, hidden_dim = 512, num_heads = 8, dropout = 0.1, show_weights=False) -> None:
        super(Decoder,self).__init__()
        
        self.show_weights = show_weights
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_key = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForwardBlock(d_model=d_model, hidden_dim= hidden_dim, dropout=dropout)

        self.multiheadAttention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
    @auto_fp16()
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
        self.relu = nn.LeakyReLU()
        self.relu_2 = nn.LeakyReLU()

        self.norm = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
    @auto_fp16()
    def forward(self, x):

        return self.norm_2(self.dropout_2(self.relu_2(self.linear_2(self.dropout(self.relu(self.linear_1(self.norm(x))))))))

"""
Same as V1 but with an addition skip connection around the cross attention
"""
@FUSION_LAYERS.register_module()
class MultiHeadCrossAttentionFlippedMoreDecoders(nn.Module):
    def __init__(self, embed_dim = 512, num_heads=8, dropout = 0.1):
        super(MultiHeadCrossAttentionFlippedMoreDecoders, self).__init__()

        self.embed_dim = embed_dim
        

        self.camera_self_attention = Decoder(self.embed_dim, hidden_dim=self.embed_dim * 2, num_heads= num_heads, dropout=dropout, show_weights=False)
        self.lidar_camera_fusion = Decoder(self.embed_dim, hidden_dim=self.embed_dim * 2, num_heads= num_heads, dropout=dropout, show_weights=False)

        self.decoder_1 = Decoder(self.embed_dim, hidden_dim=self.embed_dim * 2, num_heads= num_heads, dropout=dropout, show_weights=False)
        self.decoder_2 = Decoder(self.embed_dim, hidden_dim=self.embed_dim * 2, num_heads= num_heads, dropout=dropout, show_weights=False)
        self.decoder_3 = Decoder(self.embed_dim, hidden_dim=self.embed_dim * 2, num_heads= num_heads, dropout=dropout, show_weights=False)

        self.norm_camera_self_attention = nn.LayerNorm(self.embed_dim)
        self.norm_lidar_camera_fusion = nn.LayerNorm(self.embed_dim)
        self.norm_1 = nn.LayerNorm(self.embed_dim)
        self.norm_2 = nn.LayerNorm(self.embed_dim)
        self.norm_3 = nn.LayerNorm(self.embed_dim)

        self.pos_embed_camera = nn.Parameter(torch.randn(1, self.embed_dim, 1024) * .02) #done as in ViT: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py, (14 (image hight) * 25 image width * 6 images) / 16 (image patches)
        self.pos_embed_lidar = nn.Parameter(torch.randn(1, self.embed_dim, 1024) * .02) #done as in ViT: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py, no reduction for now


        # Patch creation
        self.lidar_patch_creation = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=4, stride=4)
        self.lidar_patch_creation_act = nn.LeakyReLU(inplace=True)
        self.lidar_patch_creation_norm = nn.BatchNorm2d(self.embed_dim)

        self.camera_patch_creation = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=4, stride=4)
        self.camera_patch_creation_act = nn.LeakyReLU(inplace=True)
        self.camera_patch_creation_norm = nn.BatchNorm2d(self.embed_dim)

        #Reverse patch creation
        self.reverse_lidar_patch_creation = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=4, stride=4)
        self.reverse_lidar_patch_creation_act = nn.LeakyReLU(inplace=True)
        self.reverse_lidar_patch_creation_norm = nn.BatchNorm2d(self.embed_dim)


        self.increase_lidar_features = nn.Conv2d(384, self.embed_dim, kernel_size=3, stride=1, padding=1)
        self.increase_lidar_features_act = nn.LeakyReLU(inplace=True)
        self.increase_lidar_features_norm = nn.BatchNorm2d(self.embed_dim)


        self.increase_camera_features = nn.Conv2d(256, self.embed_dim, kernel_size=3, stride=1, padding=1)
        self.increase_camera_features_norm = nn.BatchNorm2d(self.embed_dim)
        self.increase_camera_features_act = nn.LeakyReLU(inplace=True)

    
    def create_lidar_patches(self, lidar_tensor):

        lidar_patches = self.lidar_patch_creation_act(self.lidar_patch_creation_norm(self.lidar_patch_creation(lidar_tensor)))
        
        #flatten all patches:   
        lidar_patches = lidar_patches.view(lidar_patches.shape[0], lidar_patches.shape[1], -1)

        lidar_patches = torch.add(lidar_patches, self.pos_embed_lidar)

        #reshape to match required nn.MultiheadAttention input format: batch, seq, feature
        lidar_patches = lidar_patches.permute(0, 2, 1)  # shape: (batch_size, sequence_length, embedding_dimension)


        return lidar_patches
    
    def create_camera_patches(self, camera_tensor):

        camera_patches = self.camera_patch_creation_act(self.camera_patch_creation_norm(self.camera_patch_creation(camera_tensor)))
        
        #flatten all patches:
        camera_patches = camera_patches.view(camera_patches.shape[0], camera_patches.shape[1], -1)

        #add position embedding to flattened patches
        camera_patches = torch.add(camera_patches, self.pos_embed_camera)

        # if self.pos_embed_camera.grad is not None:
        #    print(f"camera.grad: {self.pos_embed_camera.grad.abs().max()}")
           

           
        #reshape to match required nn.MultiheadAttention input format: batch, seq, feature
        camera_patches = camera_patches.permute(0, 2, 1)  # shape: (batch_size, sequence_length, embedding_dimension)

        return camera_patches
    
    def forward(self, lidar_bev_features, camera_bev_features):
        
        # Make a copy of the tensor and convert it to CPU
        camera_bev_features_cpu = camera_bev_features.clone().cpu()

        # Check for NaN values
        if torch.isnan(camera_bev_features_cpu).any():
            print("Camera Tensor contains NaN values.")

        # Check for Inf values
        if torch.isinf(camera_bev_features_cpu).any():
            print("Camera Tensor contains Inf values.")
        
        lidar_bev_features = self.increase_lidar_features_act(self.increase_lidar_features_norm(self.increase_lidar_features(lidar_bev_features)))

        camera_bev_features = self.increase_camera_features_act(self.increase_camera_features_norm(self.increase_camera_features(camera_bev_features)))
        

        # # get patch embeddings
        image_patch_embedding = self.create_camera_patches(camera_bev_features)
        lidar_patch_embedding = self.create_lidar_patches(lidar_bev_features)

        camera_self_attention = self.norm_camera_self_attention(self.camera_self_attention(image_patch_embedding, image_patch_embedding))

        cross_attention_0 = self.norm_lidar_camera_fusion(self.lidar_camera_fusion(lidar_patch_embedding, camera_self_attention))
        cross_attention_1 = self.norm_1(self.decoder_1(cross_attention_0, cross_attention_0))
        cross_attention_1 = torch.add(cross_attention_0, cross_attention_1)
        cross_attention_2 = self.norm_2(self.decoder_2(cross_attention_1, cross_attention_1))
        cross_attention_2 = torch.add(cross_attention_1, cross_attention_2)
        cross_attention_3 = self.norm_3(self.decoder_3(cross_attention_2, cross_attention_2))
        final_cross_attention = torch.add(cross_attention_2, cross_attention_3)


        # Reshape the 1d tensor back to a 2d representation used in the CenterHead
        output = final_cross_attention.permute(0,2,1)
        output = output.view(output.shape[0], output.shape[1], 32, 32)  # Shape: [batch * 6, 256, 64, 64]

        #Reverse the patch creation op
        output = self.reverse_lidar_patch_creation_act(self.reverse_lidar_patch_creation_norm(self.reverse_lidar_patch_creation(output)))
        
        output = torch.add(output, lidar_bev_features)


        
        return output