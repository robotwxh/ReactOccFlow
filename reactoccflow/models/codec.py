"""Independent occupancy VAE inference using copied OccFM pretrained components."""
import torch
from torch import nn
from easydict import EasyDict
from .modules.embedding.voxel_height_sem_embed import VoxelHeightSemEmbed
from .modules.encoders.SwinSingleFrameDownV2 import SwinSingleFrameDownV2
from .modules.decoders.SwinSingleFrameUpV2 import SwinSingleFrameUpV2
from .modules.quantization.vae_quant import VaeQuant
from ._utils import read_state_dict, strict_load


class OccupancyCodec(nn.Module):
    """Encode labels B[,T],H,W,Z into unscaled B[,T],C,h,w latents.

    Posterior means are deterministic. Sampling requires a local generator.
    Visibility masks replace hidden labels with the configured empty class;
    this preserves the pretrained embedding width, which has no mask channel.
    """
    def __init__(self, image_size=200, encoder_depth=(2, 6), decoder_depth=(6, 2),
                 latent_dim=16, num_classes=18, empty_class=17):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.empty_class = empty_class
        self.embedding = VoxelHeightSemEmbed(EasyDict(
            FEAT_DIM=8, VOXEL_SIZE=[.4, .4, .4], HEIGHT_NUM=16,
            NUM_CATE=num_classes, ENCODER_DIM=128,
            OCC_RANGE=[-40, -40, -1, 40, 40, 5.4], skip=False))
        self.encoder = SwinSingleFrameDownV2(
            img_size=image_size, patch_size=1, input_channel=128,
            patch_embed_dim=128, depth=list(encoder_depth), latent_dim=latent_dim,
            downsample=['conv'] * len(encoder_depth))
        self.quantization = VaeQuant()
        self.decoder = SwinSingleFrameUpV2(
            input_channel=latent_dim, patch_embed_dim=128, depth=list(decoder_depth),
            out_dim=128, patched_size=image_size // (2 ** len(encoder_depth)),
            upsample=['interp'] * len(decoder_depth))

    def load_pretrained(self, path, *, weights="state_dict"):
        """Strictly map only module./model. wrappers; never ignore unmatched keys."""
        return strict_load(self, read_state_dict(path, weights=weights))

    def encode(self, occupancy, visibility=None, *, sample=False, generator=None):
        if occupancy.ndim not in (4, 5) or tuple(occupancy.shape[-3:]) != (self.image_size, self.image_size, 16):
            raise ValueError('occupancy shape must be B[,T],image_size,image_size,16')
        if sample and generator is None:
            raise ValueError('Posterior sampling requires a local torch.Generator (generator=...)')
        if visibility is not None:
            if visibility.shape != occupancy.shape:
                raise ValueError('visibility shape must match occupancy')
            occupancy = torch.where(visibility.bool(), occupancy, self.empty_class)
        if ((occupancy < 0) | (occupancy >= self.embedding.cate_dim)).any():
            raise ValueError('occupancy contains invalid class labels in visible voxels')
        data = self.embedding({'semantic_occ': occupancy})
        moments = self.encoder(data)['compressed_features']
        mean, logvar = moments.chunk(2, dim=1)
        if sample:
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
            latent = mean + torch.exp(logvar / 2).to(mean.dtype) * noise
        else:
            latent = mean
        if occupancy.ndim == 5:
            latent = latent.reshape(*occupancy.shape[:2], *latent.shape[1:])
        return latent

    def decode(self, latents, *, return_logits=False):
        if latents.ndim not in (4, 5) or latents.shape[-3] != self.latent_dim:
            raise ValueError('latents shape must be B[,T],latent_dim,h,w')
        leading = latents.shape[:-3]
        maps = self.decoder({'sampled_features': latents.reshape(-1, *latents.shape[-3:])})['decoded_map']
        b, _, h, w = maps.shape
        features = maps.reshape(b, 16, 8, h, w).permute(0, 3, 4, 1, 2)
        logits = features @ self.embedding.class_embeds.weight.T.detach()
        logits = logits.reshape(*leading, h, w, 16, self.embedding.cate_dim)
        return logits if return_logits else logits.argmax(-1)
