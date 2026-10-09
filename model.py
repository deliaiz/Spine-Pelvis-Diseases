import torch
from torch import nn, einsum
import torch.nn.functional as F
from argparse import Namespace
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from longformer import Longformer
from linformer import Linformer
from transformer import Transformer
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from torchvision import transforms
class VTN(nn.Module):
    def __init__(self, *, frames, num_classes, img_size, patch_size, spatial_frozen, spatial_size, spatial_args, temporal_type, temporal_args, spatial_suffix=''):
        super().__init__()
        self.frames = frames
        self.num_tokens = (img_size // patch_size) ** 2 + 1

        # Convert args
        spatial_args = Namespace(**spatial_args)
        temporal_args = Namespace(**temporal_args)

        self.collapse_frames = Rearrange('b f c h w -> (b f) c h w')

        #[Spatial] Transformer attention 
        self.spatial_transformer = timm.create_model(
            f'vit_{spatial_size}_patch{patch_size}_{img_size}{spatial_suffix}',
            pretrained=False,
            **vars(spatial_args)
        )
        state = torch.load('/public/home/liufj/perl5/Transformer/vit_base_patch16_224_in21k_miil-887286df.pth')
        self.spatial_transformer.load_state_dict(state, strict=False)
        # Freeze spatial backbone
        self.spatial_frozen = spatial_frozen
        if spatial_frozen:
          self.spatial_transformer.eval()
        # Spatial preprocess
        self.preprocess = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          # transforms.RandomCrop(img_size),
          #transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=self.spatial_transformer.default_cfg['mean'], std=self.spatial_transformer.default_cfg['std'])
        ])
        # Spatial Training preprocess
        config = resolve_data_config({}, model=self.spatial_transformer)
        # original trian_preprocess
        self.train_preprocess = create_transform(**config, is_training=True)
        #Spatial to temporal rearrange
        self.spatial2temporal = Rearrange('(b f) d -> b f d', f=frames)

        #[Temporal] Transformer_attention
        assert temporal_type in ['longformer', 'linformer', 'transformer'], "Only longformer, linformer, transformer are supported"
        # Copy seq_len to frames
        temporal_args.seq_len = frames
        
        if temporal_type == 'longformer':
          self.temporal_transformer = Longformer(**vars(temporal_args))
        elif temporal_type == 'linformer':
          self.temporal_transformer = Linformer(**vars(temporal_args))
        elif temporal_type == 'transformer':
          self.temporal_transformer = Transformer(**vars(temporal_args))

        # Classifer
        self.mlp_head_coarse = nn.Sequential(
            nn.LayerNorm(temporal_args.dim),
            nn.Linear(temporal_args.dim, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 2)
        )
        self.mlp_head_mid = nn.Sequential(
            nn.LayerNorm(temporal_args.dim),
            nn.Linear(temporal_args.dim, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 4)
        )
        # self.mlp_head_fine = nn.Sequential(
        #     nn.LayerNorm(temporal_args.dim),
        #     nn.Linear(temporal_args.dim, 256),
        #     nn.Linear(256, 128),
        #     nn.Linear(128, 7)
        # )
        # Random init 0.0 mean, 0.02 std
        nn.init.normal_(self.mlp_head_coarse[1].weight, mean=0.0, std=0.02)
        nn.init.normal_(self.mlp_head_mid[1].weight, mean=0.0, std=0.02)
        # nn.init.normal_(self.mlp_head_fine[1].weight, mean=0.0, std=0.02)
        # self.relu = nn.ReLU()
        # self.classifier_1 = nn.Sequential(
        #     nn.Linear(768, 2),
        #      # nn.Sigmoid()
        # )
        # self.classifier_2 = nn.Sequential(
        #     nn.Linear(768, 4),
        #     # nn.Sigmoid()
        # )
        # self.classifier_3 = nn.Sequential(
        #     nn.Linear(768, 7),
        #     # nn.Sigmoid()
        # )
        # self.classifier_3_1 = nn.Sequential(
        #     nn.Linear(768, 7)
        # )


    def forward(self, img):
        # 'b f c h w -> (b f) c h w'
        x = self.collapse_frames(img)
        # Spatial Transformer
        if self.spatial_frozen:
          with torch.no_grad():
            x = self.spatial_transformer.forward_features(x)
            if x.dim() == 3:
              if x.shape[1] == self.num_tokens:
                x = x[:, 0]
              elif x.shape[0] == self.num_tokens:
                x = x[0]
              else:
                x = x[:, 0]
        else:
          x = self.spatial_transformer.forward_features(x) #（128,197,768)
          if x.dim() == 3:
            if x.shape[1] == self.num_tokens:
                x = x.mean(dim=1)
              # x = x[:, 0] #(128,768) 原来的
            elif x.shape[0] == self.num_tokens:
              x = x[0]
            else:
              x = x[:, 0]
        # Spatial to temporal
        x = self.spatial2temporal(x) #[32,768]
        # Temporal Transformer
        x = self.temporal_transformer(x)  #[B,768]


        # Classifier
        # y_coarse = self.classifier_1(self.relu(x))  # (B,num_class)=(B,2)
        # y_mid = self.classifier_2(self.relu(x))  # (B,num_class)=(B,4)
        # y_fine = self.classifier_3(self.relu(x))  # (B,num_class)=(B,7)
        # y_fine_sof = self.classifier_3_1(self.relu(x))  # (B,num_class)=(B,7)
        y_coarse = self.mlp_head_coarse(x)  # (B,num_class)=(B,2)
        y_mid = self.mlp_head_mid(x) # (B,num_class)=(B,4)
        # y_fine = self.mlp_head_fine(x)  # (B,num_class)=(B,7)
        return  y_coarse,y_mid
