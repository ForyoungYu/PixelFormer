in_channels = [128, 256, 512, 1024]
out = in_channels[::-1]
print(out)

        # head_dim = 64
        # qk_dims = [None, None, None, None]
        # embed_dim = [64, 128, 320, 512]
        # drop_path_rate = 0
        # depth = [3, 4, 8, 3]
        # nheads = [dim // head_dim for dim in qk_dims]
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        # cur = 0

        # for i in range(4):
        #     stage = nn.Sequential(
        #         *[
        #             BiDecoder(
        #                 dim=embed_dim[i],
        #                 decoder=True,
        #                 drop_path=dp_rates[cur + j],
        #                 layer_scale_init_value=layer_scale_init_value,
        #                 topk=topks[i],
        #                 num_heads=nheads[i],
        #                 n_win=n_win,
        #                 qk_dim=qk_dims[i],
        #                 qk_scale=qk_scale,
        #                 kv_per_win=kv_per_wins[i],
        #                 kv_downsample_ratio=kv_downsample_ratios[i],
        #                 kv_downsample_kernel=kv_downsample_kernels[i],
        #                 kv_downsample_mode=kv_downsample_mode,
        #                 param_attention=param_attention,
        #                 param_routing=param_routing,
        #                 diff_routing=diff_routing,
        #                 soft_routing=soft_routing,
        #                 mlp_ratio=mlp_ratios[i],
        #                 mlp_dwconv=mlp_dwconv,
        #                 side_dwconv=side_dwconv,
        #                 before_attn_dwconv=before_attn_dwconv,
        #                 pre_norm=pre_norm,
        #                 auto_pad=auto_pad,
        #             )
        #             for j in range(depth[i])
        #         ],
        #     )
        #     if i in use_checkpoint_stages:
        #         stage = checkpoint_wrapper(stage)
        #     self.stages.append(stage)