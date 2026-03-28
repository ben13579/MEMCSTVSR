import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..core import gradient_checkpoint_forward
from .wan_video_vae import AttentionBlock, CausalConv3d, ResidualBlock


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

def make_coord_3d(shape, flatten=True, device=None, dtype=torch.float32):
    axes = []
    for size in shape:
        if size == 1:
            seq = torch.zeros(1, device=device, dtype=dtype)
        else:
            radius = 1.0 / size
            seq = torch.linspace(-1.0 + radius, 1.0 - radius, steps=size, device=device, dtype=dtype)
        axes.append(seq)

    mesh = torch.meshgrid(*axes, indexing="ij")
    coord = torch.stack(mesh, dim=-1)
    if flatten:
        coord = coord.view(-1, coord.shape[-1])
    return coord


def make_coord_2d(shape, flatten=True, device=None, dtype=torch.float32):
    axes = []
    for size in shape:
        if size == 1:
            seq = torch.zeros(1, device=device, dtype=dtype)
        else:
            radius = 1.0 / size
            seq = torch.linspace(-1.0 + radius, 1.0 - radius, steps=size, device=device, dtype=dtype)
        axes.append(seq)

    mesh = torch.meshgrid(*axes, indexing="ij")
    coord = torch.stack(mesh, dim=-1)
    if flatten:
        coord = coord.view(-1, coord.shape[-1])
    return coord


def make_coord_cell_3d(batch_size, output_size, device=None, dtype=torch.float32):
    coord = make_coord_3d(output_size, flatten=True, device=device, dtype=dtype)
    coord = coord.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    cell = torch.ones_like(coord)
    for axis, size in enumerate(output_size):
        cell[:, :, axis] *= 2.0 / size
    return coord, cell


def make_coord_cell_2d(batch_size, output_size, device=None, dtype=torch.float32):
    coord = make_coord_2d(output_size, flatten=True, device=device, dtype=dtype)
    coord = coord.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    cell = torch.ones_like(coord)
    for axis, size in enumerate(output_size):
        cell[:, :, axis] *= 2.0 / size
    return coord, cell


def sinusoidal_embed_coord(coord, embed_dim):
    if embed_dim % 2 != 0:
        raise ValueError(f"sinusoidal coordinate embedding requires an even embed_dim, got {embed_dim}")

    embeddings = []
    for axis in range(coord.shape[-1]):
        axis_coord = coord[..., axis].reshape(-1)
        axis_embedding = sinusoidal_embedding_1d(embed_dim, axis_coord)
        axis_embedding = axis_embedding.view(*coord.shape[:-1], embed_dim)
        embeddings.append(axis_embedding)
    return torch.cat(embeddings, dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        last_dim = in_dim
        for hidden_dim in hidden_list:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.GELU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NoUpsampleDecoder3d(nn.Module):
    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_upsample=(True, True, False),
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_upsample = list(temperal_upsample)
        self.dropout = dropout
        self.out_dim = dim * self.dim_mult[-1]

        self.conv1 = CausalConv3d(z_dim, self.out_dim, 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(self.out_dim, self.out_dim, dropout),
            AttentionBlock(self.out_dim),
            ResidualBlock(self.out_dim, self.out_dim, dropout),
        )

        scale = 1.0 / 2 ** (len(self.dim_mult) - 2)
        stages = []
        for _ in range(len(self.dim_mult)):
            blocks = []
            for _ in range(num_res_blocks + 1):
                blocks.append(ResidualBlock(self.out_dim, self.out_dim, dropout))
                if scale in self.attn_scales:
                    blocks.append(AttentionBlock(self.out_dim))
            stages.append(nn.Sequential(*blocks))
            scale *= 2.0
        self.stages = nn.ModuleList(stages)

    def forward(self, z, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False):
        enable_checkpoint = torch.is_grad_enabled()
        x = self.conv1(z)
        for layer in self.middle:
            x = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing and enable_checkpoint,
                use_gradient_checkpointing_offload and enable_checkpoint,
                x,
            )
        for stage in self.stages:
            x = gradient_checkpoint_forward(
                stage,
                use_gradient_checkpointing and enable_checkpoint,
                use_gradient_checkpointing_offload and enable_checkpoint,
                x,
            )
        return x


class LIIF3D(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=3,
        hidden_list=(256, 256, 256, 256),
        cell_decode=True,
        local_ensemble=True,
        coord_embed_mode="raw",
        coord_embed_dim=16,
        neighbor_fusion_mode="weighted",
    ):
        super().__init__()
        self.cell_decode = cell_decode
        self.local_ensemble = local_ensemble
        self.coord_embed_mode = coord_embed_mode
        self.coord_embed_dim = coord_embed_dim
        self.neighbor_fusion_mode = neighbor_fusion_mode

        if coord_embed_mode == "raw":
            coord_in_dim = 3
        elif coord_embed_mode == "sinusoidal":
            coord_in_dim = 3 * coord_embed_dim
        else:
            raise ValueError(f"Unsupported coord_embed_mode: {coord_embed_mode}")
        self.coord_in_dim = coord_in_dim

        self.num_neighbors = 8 if local_ensemble else 1

        if neighbor_fusion_mode == "weighted":
            imnet_in_dim = in_dim + coord_in_dim
            if cell_decode:
                imnet_in_dim += 3
        elif neighbor_fusion_mode == "concat_once":
            imnet_in_dim = self.num_neighbors * (in_dim + coord_in_dim)
            if cell_decode:
                imnet_in_dim += 3
        else:
            raise ValueError(f"Unsupported neighbor_fusion_mode: {neighbor_fusion_mode}")
        self.imnet = MLP(imnet_in_dim, out_dim, hidden_list)

    def _get_feat_coord(self, feat):
        b, _, t, h, w = feat.shape
        feat_coord = make_coord_3d((t, h, w), flatten=False, device=feat.device, dtype=feat.dtype)
        feat_coord = feat_coord.permute(3, 0, 1, 2).unsqueeze(0).expand(b, -1, -1, -1, -1)
        return feat_coord

    def _encode_coord(self, rel_coord):
        if self.coord_embed_mode == "sinusoidal":
            return sinusoidal_embed_coord(rel_coord, self.coord_embed_dim)
        return rel_coord

    def _build_rel_cell(self, cell, feat):
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-3]
        rel_cell[:, :, 1] *= feat.shape[-2]
        rel_cell[:, :, 2] *= feat.shape[-1]
        return rel_cell

    def _neighbor_offsets(self):
        if self.local_ensemble:
            vt_lst = (-1, 1)
            vy_lst = (-1, 1)
            vx_lst = (-1, 1)
            eps_shift = 1e-6
        else:
            vt_lst = vy_lst = vx_lst = (0,)
            eps_shift = 0.0

        offsets = []
        for vt in vt_lst:
            for vy in vy_lst:
                for vx in vx_lst:
                    offsets.append((vt, vy, vx))
        return offsets, eps_shift

    def _sample_neighbors(self, feat, coord):
        offsets, eps_shift = self._neighbor_offsets()

        rt = 1.0 / feat.shape[-3]
        ry = 1.0 / feat.shape[-2]
        rx = 1.0 / feat.shape[-1]
        feat_coord = self._get_feat_coord(feat)

        sampled_feats = []
        coord_inputs = []
        rel_coords = []
        for vt, vy, vx in offsets:
            coord_ = coord.clone()
            coord_[:, :, 0] += vt * rt + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_[:, :, 2] += vx * rx + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

            grid = coord_.reshape(coord_.shape[0], 1, 1, coord_.shape[1], 3)
            grid = grid[..., [2, 1, 0]]

            q_feat = F.grid_sample(
                feat,
                grid,
                mode="nearest",
                align_corners=False,
            )[:, :, 0, 0, :].permute(0, 2, 1)
            q_coord = F.grid_sample(
                feat_coord,
                grid,
                mode="nearest",
                align_corners=False,
            )[:, :, 0, 0, :].permute(0, 2, 1)

            rel_coord = coord - q_coord
            rel_coord[:, :, 0] *= feat.shape[-3]
            rel_coord[:, :, 1] *= feat.shape[-2]
            rel_coord[:, :, 2] *= feat.shape[-1]

            sampled_feats.append(q_feat)
            coord_inputs.append(self._encode_coord(rel_coord))
            rel_coords.append(rel_coord)

        sampled_feats = torch.stack(sampled_feats, dim=2)
        coord_inputs = torch.stack(coord_inputs, dim=2)
        rel_coords = torch.stack(rel_coords, dim=2)
        return sampled_feats, coord_inputs, rel_coords, offsets

    def _query_rgb_weighted(self, feat, coord, cell):
        sampled_feats, coord_inputs, rel_coords, offsets = self._sample_neighbors(feat, coord)
        bs, q = coord.shape[:2]
        rel_cell = self._build_rel_cell(cell, feat) if self.cell_decode else None

        preds = []
        for neighbor_id in range(sampled_feats.shape[2]):
            inp = torch.cat([sampled_feats[:, :, neighbor_id, :], coord_inputs[:, :, neighbor_id, :]], dim=-1)
            if self.cell_decode:
                inp = torch.cat([inp, rel_cell], dim=-1)
            pred = self.imnet(inp.reshape(bs * q, -1)).view(bs, q, -1)
            preds.append(pred)

        areas = torch.abs(rel_coords[:, :, :, 0] * rel_coords[:, :, :, 1] * rel_coords[:, :, :, 2]) + 1e-9

        if self.local_ensemble:
            corner_to_index = {corner: index for index, corner in enumerate(offsets)}
            reordered_areas = [None] * len(offsets)
            for index, corner in enumerate(offsets):
                opposite_corner = (-corner[0], -corner[1], -corner[2])
                reordered_areas[index] = areas[:, :, corner_to_index[opposite_corner]]
            areas = torch.stack(reordered_areas, dim=2)

        tot_area = areas.sum(dim=2)
        ret = 0
        for neighbor_id, pred in enumerate(preds):
            area = areas[:, :, neighbor_id]
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def _query_rgb_concat_once(self, feat, coord, cell):
        sampled_feats, coord_inputs, _, _ = self._sample_neighbors(feat, coord)
        bs, q = coord.shape[:2]

        inp = torch.cat([sampled_feats, coord_inputs], dim=-1).reshape(bs, q, -1)
        if self.cell_decode:
            rel_cell = self._build_rel_cell(cell, feat)
            inp = torch.cat([inp, rel_cell], dim=-1)
        return self.imnet(inp.reshape(bs * q, -1)).view(bs, q, -1)

    def query_rgb(self, feat, coord, cell):
        if self.neighbor_fusion_mode == "weighted":
            return self._query_rgb_weighted(feat, coord, cell)
        return self._query_rgb_concat_once(feat, coord, cell)

    def batched_predict(
        self,
        feat,
        coord,
        cell,
        bsize,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        preds = []
        ql = 0
        n = coord.shape[1]
        enable_checkpoint = torch.is_grad_enabled()
        while ql < n:
            qr = min(ql + bsize, n)
            pred = gradient_checkpoint_forward(
                self.query_rgb,
                use_gradient_checkpointing=use_gradient_checkpointing and enable_checkpoint,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload and enable_checkpoint,
                feat=feat,
                coord=coord[:, ql:qr, :],
                cell=cell[:, ql:qr, :],
            )
            preds.append(pred)
            ql = qr
        return torch.cat(preds, dim=1)

    def forward(
        self,
        feat,
        coord=None,
        cell=None,
        output_size=None,
        return_img=True,
        bsize=65536,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        if return_img and output_size is None:
            raise ValueError("output_size is required when return_img=True")

        if coord is None:
            if output_size is None:
                raise ValueError("output_size is required when coord is None")
            coord, cell = make_coord_cell_3d(
                feat.shape[0],
                output_size,
                device=feat.device,
                dtype=feat.dtype,
            )
        elif cell is None:
            if output_size is None:
                raise ValueError("output_size is required when cell is None")
            cell = torch.ones_like(coord)
            for axis, size in enumerate(output_size):
                cell[:, :, axis] *= 2.0 / size

        if bsize > 0:
            out = self.batched_predict(
                feat,
                coord,
                cell,
                bsize,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        else:
            out = gradient_checkpoint_forward(
                self.query_rgb,
                use_gradient_checkpointing=use_gradient_checkpointing and torch.is_grad_enabled(),
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload and torch.is_grad_enabled(),
                feat=feat,
                coord=coord,
                cell=cell,
            )

        if return_img:
            out = rearrange(
                out,
                "b (t h w) c -> b c t h w",
                t=output_size[0],
                h=output_size[1],
                w=output_size[2],
            )
        return out


class LIIF2D(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=3,
        hidden_list=(256, 256, 256, 256),
        cell_decode=True,
        local_ensemble=True,
        coord_embed_mode="raw",
        coord_embed_dim=16,
        neighbor_fusion_mode="weighted",
    ):
        super().__init__()
        self.cell_decode = cell_decode
        self.local_ensemble = local_ensemble
        self.coord_embed_mode = coord_embed_mode
        self.coord_embed_dim = coord_embed_dim
        self.neighbor_fusion_mode = neighbor_fusion_mode

        if coord_embed_mode == "raw":
            coord_in_dim = 2
        elif coord_embed_mode == "sinusoidal":
            coord_in_dim = 2 * coord_embed_dim
        else:
            raise ValueError(f"Unsupported coord_embed_mode: {coord_embed_mode}")
        self.coord_in_dim = coord_in_dim

        self.num_neighbors = 4 if local_ensemble else 1

        if neighbor_fusion_mode == "weighted":
            imnet_in_dim = in_dim + coord_in_dim
            if cell_decode:
                imnet_in_dim += 2
        elif neighbor_fusion_mode == "concat_once":
            imnet_in_dim = self.num_neighbors * (in_dim + coord_in_dim)
            if cell_decode:
                imnet_in_dim += 2
        else:
            raise ValueError(f"Unsupported neighbor_fusion_mode: {neighbor_fusion_mode}")
        self.imnet = MLP(imnet_in_dim, out_dim, hidden_list)

    def _get_feat_2d(self, feat):
        if feat.ndim != 5:
            raise ValueError(f"LIIF2D expects feat to be 5D [B, C, T, H, W], got shape={tuple(feat.shape)}")
        if feat.shape[2] != 1:
            raise ValueError(f"LIIF2D expects a single-frame feature volume, got T={feat.shape[2]}")
        return feat[:, :, 0, :, :]

    def _normalize_output_size(self, output_size):
        if len(output_size) == 3:
            if output_size[0] != 1:
                raise ValueError(f"LIIF2D expects output temporal size 1, got output_size={output_size}")
            return output_size[1], output_size[2]
        if len(output_size) == 2:
            return output_size
        raise ValueError(f"LIIF2D expects output_size to have length 2 or 3, got {output_size}")

    def _normalize_coord_and_cell(self, coord, cell):
        if coord.shape[-1] == 3:
            coord = coord[:, :, 1:]
        if cell is not None and cell.shape[-1] == 3:
            cell = cell[:, :, 1:]
        if coord.shape[-1] != 2:
            raise ValueError(f"LIIF2D expects coord to have last dim 2 or 3, got shape={tuple(coord.shape)}")
        if cell is not None and cell.shape[-1] != 2:
            raise ValueError(f"LIIF2D expects cell to have last dim 2 or 3, got shape={tuple(cell.shape)}")
        return coord, cell

    def _get_feat_coord(self, feat_2d):
        b, _, h, w = feat_2d.shape
        feat_coord = make_coord_2d((h, w), flatten=False, device=feat_2d.device, dtype=feat_2d.dtype)
        feat_coord = feat_coord.permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)
        return feat_coord

    def _encode_coord(self, rel_coord):
        if self.coord_embed_mode == "sinusoidal":
            return sinusoidal_embed_coord(rel_coord, self.coord_embed_dim)
        return rel_coord

    def _build_rel_cell(self, cell, feat_2d):
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat_2d.shape[-2]
        rel_cell[:, :, 1] *= feat_2d.shape[-1]
        return rel_cell

    def _neighbor_offsets(self):
        if self.local_ensemble:
            vy_lst = (-1, 1)
            vx_lst = (-1, 1)
            eps_shift = 1e-6
        else:
            vy_lst = vx_lst = (0,)
            eps_shift = 0.0

        offsets = []
        for vy in vy_lst:
            for vx in vx_lst:
                offsets.append((vy, vx))
        return offsets, eps_shift

    def _sample_neighbors(self, feat, coord, cell):
        feat_2d = self._get_feat_2d(feat)
        coord, cell = self._normalize_coord_and_cell(coord, cell)
        offsets, eps_shift = self._neighbor_offsets()

        ry = 1.0 / feat_2d.shape[-2]
        rx = 1.0 / feat_2d.shape[-1]
        feat_coord = self._get_feat_coord(feat_2d)

        sampled_feats = []
        coord_inputs = []
        rel_coords = []
        for vy, vx in offsets:
            coord_ = coord.clone()
            coord_[:, :, 0] += vy * ry + eps_shift
            coord_[:, :, 1] += vx * rx + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

            grid = coord_.flip(-1).unsqueeze(1)

            q_feat = F.grid_sample(
                feat_2d,
                grid,
                mode="nearest",
                align_corners=False,
            )[:, :, 0, :].permute(0, 2, 1)
            q_coord = F.grid_sample(
                feat_coord,
                grid,
                mode="nearest",
                align_corners=False,
            )[:, :, 0, :].permute(0, 2, 1)

            rel_coord = coord - q_coord
            rel_coord[:, :, 0] *= feat_2d.shape[-2]
            rel_coord[:, :, 1] *= feat_2d.shape[-1]

            sampled_feats.append(q_feat)
            coord_inputs.append(self._encode_coord(rel_coord))
            rel_coords.append(rel_coord)

        sampled_feats = torch.stack(sampled_feats, dim=2)
        coord_inputs = torch.stack(coord_inputs, dim=2)
        rel_coords = torch.stack(rel_coords, dim=2)
        return feat_2d, coord, cell, sampled_feats, coord_inputs, rel_coords, offsets

    def _query_rgb_weighted(self, feat, coord, cell):
        feat_2d, coord, cell, sampled_feats, coord_inputs, rel_coords, offsets = self._sample_neighbors(feat, coord, cell)
        bs, q = coord.shape[:2]
        rel_cell = self._build_rel_cell(cell, feat_2d) if self.cell_decode else None

        preds = []
        for neighbor_id in range(sampled_feats.shape[2]):
            inp = torch.cat([sampled_feats[:, :, neighbor_id, :], coord_inputs[:, :, neighbor_id, :]], dim=-1)
            if self.cell_decode:
                inp = torch.cat([inp, rel_cell], dim=-1)
            pred = self.imnet(inp.reshape(bs * q, -1)).view(bs, q, -1)
            preds.append(pred)

        areas = torch.abs(rel_coords[:, :, :, 0] * rel_coords[:, :, :, 1]) + 1e-9

        if self.local_ensemble:
            corner_to_index = {corner: index for index, corner in enumerate(offsets)}
            reordered_areas = [None] * len(offsets)
            for index, corner in enumerate(offsets):
                opposite_corner = (-corner[0], -corner[1])
                reordered_areas[index] = areas[:, :, corner_to_index[opposite_corner]]
            areas = torch.stack(reordered_areas, dim=2)

        tot_area = areas.sum(dim=2)
        ret = 0
        for neighbor_id, pred in enumerate(preds):
            area = areas[:, :, neighbor_id]
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def _query_rgb_concat_once(self, feat, coord, cell):
        feat_2d, coord, cell, sampled_feats, coord_inputs, _, _ = self._sample_neighbors(feat, coord, cell)
        bs, q = coord.shape[:2]

        inp = torch.cat([sampled_feats, coord_inputs], dim=-1).reshape(bs, q, -1)
        if self.cell_decode:
            rel_cell = self._build_rel_cell(cell, feat_2d)
            inp = torch.cat([inp, rel_cell], dim=-1)
        return self.imnet(inp.reshape(bs * q, -1)).view(bs, q, -1)

    def query_rgb(self, feat, coord, cell):
        if self.neighbor_fusion_mode == "weighted":
            return self._query_rgb_weighted(feat, coord, cell)
        return self._query_rgb_concat_once(feat, coord, cell)

    def batched_predict(
        self,
        feat,
        coord,
        cell,
        bsize,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        preds = []
        ql = 0
        n = coord.shape[1]
        enable_checkpoint = torch.is_grad_enabled()
        while ql < n:
            qr = min(ql + bsize, n)
            pred = gradient_checkpoint_forward(
                self.query_rgb,
                use_gradient_checkpointing=use_gradient_checkpointing and enable_checkpoint,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload and enable_checkpoint,
                feat=feat,
                coord=coord[:, ql:qr, :],
                cell=cell[:, ql:qr, :],
            )
            preds.append(pred)
            ql = qr
        return torch.cat(preds, dim=1)

    def forward(
        self,
        feat,
        coord=None,
        cell=None,
        output_size=None,
        return_img=True,
        bsize=65536,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        if return_img and output_size is None:
            raise ValueError("output_size is required when return_img=True")

        spatial_output_size = None
        if output_size is not None:
            spatial_output_size = self._normalize_output_size(output_size)

        if coord is None:
            if spatial_output_size is None:
                raise ValueError("output_size is required when coord is None")
            coord, cell = make_coord_cell_2d(
                feat.shape[0],
                spatial_output_size,
                device=feat.device,
                dtype=feat.dtype,
            )
        else:
            coord, cell = self._normalize_coord_and_cell(coord, cell)
            if cell is None:
                if spatial_output_size is None:
                    raise ValueError("output_size is required when cell is None")
                cell = torch.ones_like(coord)
                for axis, size in enumerate(spatial_output_size):
                    cell[:, :, axis] *= 2.0 / size

        if bsize > 0:
            out = self.batched_predict(
                feat,
                coord,
                cell,
                bsize,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        else:
            out = gradient_checkpoint_forward(
                self.query_rgb,
                use_gradient_checkpointing=use_gradient_checkpointing and torch.is_grad_enabled(),
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload and torch.is_grad_enabled(),
                feat=feat,
                coord=coord,
                cell=cell,
            )

        if return_img:
            out = rearrange(
                out,
                "b (h w) c -> b c 1 h w",
                h=spatial_output_size[0],
                w=spatial_output_size[1],
            )
        return out


class WanVideoIND(nn.Module):
    def __init__(
        self,
        decoder_config=None,
        liif_config=None,
    ):
        super().__init__()
        # decoder_defaults = {
        #     "dim": 96,
        #     "z_dim": 16,
        #     "dim_mult": [1, 2, 4, 4],
        #     "num_res_blocks": 2,
        #     "attn_scales": [],
        #     "temperal_upsample": [True, True, False],
        #     "dropout": 0.0,
        # }
        decoder_defaults = {
            "dim": 96,
            "z_dim": 16,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temperal_upsample": [True, True, False],
            "dropout": 0.0,
        }
        liif_defaults = {
            "hidden_list": [256, 256, 256, 256],
            "cell_decode": True,
            "local_ensemble": True,
            "neighbor_fusion_mode": "concat_once",
            "coord_embed_mode": "raw",
            "coord_embed_dim": 16,
            "mode": "3d",
        }
        if decoder_config is not None:
            decoder_defaults.update(decoder_config)
        if liif_config is not None:
            liif_defaults.update(liif_config)

        z_dim = decoder_defaults["z_dim"]
        feature_dim = decoder_defaults["dim"] * decoder_defaults["dim_mult"][-1]
        inr_mode = liif_defaults.pop("mode")

        self.pre_conv = nn.Conv3d(z_dim, z_dim, kernel_size=1)
        self.decoder = NoUpsampleDecoder3d(**decoder_defaults)
        if inr_mode == "2d":
            self.inr = LIIF2D(in_dim=feature_dim, out_dim=3, **liif_defaults)
        elif inr_mode == "3d":
            self.inr = LIIF3D(in_dim=feature_dim, out_dim=3, **liif_defaults)
        else:
            raise ValueError(f"Unsupported LIIF mode: {inr_mode}")
        self.inr_mode = inr_mode

    @staticmethod
    def default_output_size(z):
        return (4 * z.shape[2] - 3, 8 * z.shape[3], 8 * z.shape[4])

    def decode_features(self, z, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False):
        pre_conv_weight = self.pre_conv.weight
        if z.device != pre_conv_weight.device or z.dtype != pre_conv_weight.dtype:
            z = z.to(device=pre_conv_weight.device, dtype=pre_conv_weight.dtype)
        return self.decoder(
            self.pre_conv(z),
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    def query_rgb(self, feat, coord, cell):
        return self.inr.query_rgb(feat, coord, cell)

    def forward(
        self,
        z,
        coord=None,
        cell=None,
        output_size=None,
        return_img=True,
        bsize=0,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        feat = self.decode_features(
            z,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        if output_size is None and return_img:
            output_size = self.default_output_size(z)
        return self.inr(
            feat,
            coord=coord,
            cell=cell,
            output_size=output_size,
            return_img=return_img,
            bsize=bsize,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
