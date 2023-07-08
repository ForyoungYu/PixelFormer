import torch
import torch.nn as nn
import torch.nn.functional as F

# from biformer.models.biformer import Block as BiDecoder
# from biformer.models.biformer import (biformer_base, biformer_small,
#                                       biformer_tiny)
from .layers import AttractorLayer, AttractorLayerUnnormed
from .layers.localbins_layers import (
    Projector,
    SeedBinRegressor,
    SeedBinRegressorUnnormed,
)

from .PQI import PSP
from .SAM import SAM
from .swin_transformer import SwinTransformer

########################################################################################################################


class BCP(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        max_depth,
        min_depth,
        in_features=512,
        hidden_features=512 * 4,
        out_features=256,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x):
        x = torch.mean(x.flatten(start_dim=2), dim=2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        bins = torch.softmax(x, dim=1)
        bins = bins / bins.sum(dim=1, keepdim=True)
        bin_widths = (self.max_depth - self.min_depth) * bins
        bin_widths = nn.functional.pad(
            bin_widths, (1, 0), mode="constant", value=self.min_depth
        )
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.contiguous().view(n, dout, 1, 1)
        return centers


class PixelFormer(nn.Module):
    def __init__(
        self,
        version=None,
        inv_depth=False,
        pretrained=None,
        frozen_stages=-1,
        # Biformer:
        # backbone="biformer_base",
        min_depth=1e-3,
        max_depth=100.0,
        # Zoe params:
        n_bins=64,
        bin_embedding_dim=128,
        bin_centers_type="softplus",
        n_attractors=[16, 8, 4, 1],
        attractor_alpha=300,
        attractor_gamma=2,
        attractor_kind="sum",
        attractor_type="exp",
        **kwargs,
    ):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False

        norm_cfg = dict(type="BN", requires_grad=True)
        # norm_cfg = dict(type="GN", requires_grad=True, num_groups=8)

        window_size = int(version[-2:])

        if version[:-2] == "base":
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == "large":
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == "tiny":
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages,
        )

        # TODO: change backbone to biformer
        # if backbone == "biformer_tiny":
        #     self.backbone = biformer_tiny()
        #     self.decoder_ = biformer_tiny(out_kv=True)
        #     self.in_channels = [64, 128, 256, 512]
        # elif backbone == "biformer_small":
        #     self.backbone = biformer_small()
        #     self.decoder_ = biformer_small(out_kv=True)
        #     self.in_channels = [64, 128, 256, 512]
        # elif backbone == "biformer_base":
        #     self.backbone = biformer_base()
        #     self.decoder_ = biformer_base(out_kv=True)
        #     self.in_channels = [96, 192, 384, 768]

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False,
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        v_dim = decoder_cfg["num_classes"] * 4
        win = 7
        sam_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]

        self.sam4 = SAM(
            input_dim=in_channels[3],
            embed_dim=sam_dims[3],
            window_size=win,
            v_dim=v_dims[3],
            num_heads=32,
        )
        self.sam3 = SAM(
            input_dim=in_channels[2],
            embed_dim=sam_dims[2],
            window_size=win,
            v_dim=v_dims[2],
            num_heads=16,
        )
        self.sam2 = SAM(
            input_dim=in_channels[1],
            embed_dim=sam_dims[1],
            window_size=win,
            v_dim=v_dims[1],
            num_heads=8,
        )
        self.sam1 = SAM(
            input_dim=in_channels[0],
            embed_dim=sam_dims[0],
            window_size=win,
            v_dim=v_dims[0],
            num_heads=4,
        )

        # TODO: replace PSP module
        self.decoder = PSP(**decoder_cfg)
        self.disp_head1 = DispHead(input_dim=sam_dims[0], n_bins=n_bins)

        self.bcp = BCP(max_depth=max_depth, min_depth=min_depth)

        # ZoeDepth Metric Bins Module
        self.bin_centers_type = bin_centers_type
        if bin_centers_type == "normed":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayer
        elif bin_centers_type == "softplus":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid1":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid2":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayer
        else:
            raise ValueError(
                "bin_centers_type should be one of 'normed', 'softplus', 'hybrid1', 'hybrid2'"
            )

        # btlnck_features = in_channels[-2]
        btlnck_features = embed_dim
        self.seed_bin_regressor = SeedBinRegressorLayer(
            btlnck_features, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth
        )
        self.seed_projector = Projector(btlnck_features, bin_embedding_dim)
        self.projectors = nn.ModuleList(
            [Projector(num_out, bin_embedding_dim) for num_out in sam_dims[::-1]] # 原：in_channels[::-1]]
        )

        self.attractors = nn.ModuleList(
            [
                Attractor(
                    bin_embedding_dim,
                    n_bins,
                    n_attractors=n_attractors[i],
                    min_depth=min_depth,
                    max_depth=max_depth,
                    alpha=attractor_alpha,
                    gamma=attractor_gamma,
                    kind=attractor_kind,
                    attractor_type=attractor_type,
                )
                for i in range(len(in_channels))
            ]
        )

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f"== Load encoder backbone from: {pretrained}")
        self.backbone.init_weights(pretrained=pretrained)
        self.decoder.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def forward(self, imgs):
        enc_feats = self.backbone(imgs)
        # print("Backbone:")
        # print(enc_feats[0].shape)
        # print(enc_feats[1].shape)
        # print(enc_feats[2].shape)
        # print(enc_feats[3].shape)
        # print("=========================")

        if self.with_neck:
            enc_feats = self.neck(enc_feats)

        x_blocks = []
        q4 = self.decoder(enc_feats)  # [1, 512, 7, 7]
        # print(q4.shape)

        q3 = self.sam4(enc_feats[3], q4)  # [1, 1024, 7, 7]
        # print(q3.shape)
        x_blocks.append(q3)
        q3 = nn.PixelShuffle(2)(q3)
        q2 = self.sam3(enc_feats[2], q3)  # [1, 512, 14, 14]
        # print(q2.shape)
        x_blocks.append(q2)
        q2 = nn.PixelShuffle(2)(q2)
        q1 = self.sam2(enc_feats[1], q2)  # [1, 256, 28, 28]
        # print(q1.shape)
        x_blocks.append(q1)
        q1 = nn.PixelShuffle(2)(q1)
        q0 = self.sam1(enc_feats[0], q1)  # [1, 128, 56, 56]
        # print(q0.shape)
        x_blocks.append(q0)

        # ZoeDepth Metric Bins Module
        _, seed_b_centers = self.seed_bin_regressor(q4) # 将q4的特征图映射到bin_dim:64

        if self.bin_centers_type == "normed" or self.bin_centers_type == "hybrid2":
            # normalize seed bins
            b_prev = (seed_b_centers - self.min_depth) / (
                self.max_depth - self.min_depth
            )
        else:
            b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector(q4)  # btlnck:512 -> bin_dim:128
        for projector, attractor, x in zip(self.projectors, self.attractors, x_blocks):
            b_embedding = projector(x)  # q:[1536, 768, 384, 192] -> bin_dim:128
            b, bin_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True
            )
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        # bin_centers = self.bcp(q4)
        # bin_centers = self.att

        f = self.disp_head1(q0, bin_centers, 4)

        return f


class DispHead(nn.Module):
    def __init__(self, input_dim=100, n_bins=256):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, n_bins, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, centers, scale):
        x = self.conv1(x)
        x = x.softmax(dim=1)
        x = torch.sum(x * centers, dim=1, keepdim=True)
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(
        x, scale_factor=scale_factor, mode=mode, align_corners=align_corners
    )
