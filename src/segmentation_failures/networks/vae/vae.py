"""
This code is adapted from David Zimmerer's VAE implementation. Credits to him.
"""

import torch
import torch.distributions as dist

from segmentation_failures.networks.vae.encoder_decoder import (
    Encoder,
    EncoderLiu,
    Generator,
    GeneratorLiu,
)


class VAE(torch.nn.Module):
    def __init__(
        self,
        input_size,
        z_dim=512,
        h_sizes=(16, 32, 64, 256, 1024),
        kernel_size=3,
        to_1x1=True,
        conv_op=torch.nn.Conv2d,
        upsample_op=torch.nn.ConvTranspose2d,
        normalization_op=None,
        activation_op=torch.nn.LeakyReLU,
        conv_params=None,
        activation_params=None,
        block_op=None,
        block_params=None,
        output_channels=None,
        additional_input_slices=None,
        symmetric_decoder=True,
        liu_architecture=False,
        *args,
        **kwargs,
    ):

        super(VAE, self).__init__()

        input_size_enc = list(input_size)
        input_size_dec = list(input_size)
        if output_channels is not None:
            input_size_dec[0] = output_channels
        if additional_input_slices is not None:
            input_size_enc[0] += additional_input_slices * 2

        min_spatial_size = 4
        strides = []
        curr_sizes = input_size[1:]
        for layer in h_sizes:
            curr_stride = [1 if 0.5 * sz < min_spatial_size else 2 for sz in curr_sizes]
            if any([sz % st != 0 for sz, st in zip(curr_sizes, curr_stride)]):
                raise ValueError(
                    f"Spatial size {curr_sizes} for layer {layer} not divisible by stride {curr_stride}"
                )
            curr_sizes = [sz // st for sz, st in zip(curr_sizes, curr_stride)]
            strides.append(curr_stride)
        strides = strides[::-1]  # the strided convolutions should rather come at the end
        if liu_architecture:
            enc_cls = EncoderLiu
            dec_cls = GeneratorLiu
        else:
            enc_cls = Encoder
            dec_cls = Generator
        self.enc = enc_cls(
            image_size=input_size_enc,
            h_size=h_sizes,
            z_dim=z_dim * 2,
            kernel_size=kernel_size,
            strides=strides,
            normalization_op=normalization_op,
            to_1x1=to_1x1,
            conv_op=conv_op,
            conv_params=conv_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
        )
        self.dec = dec_cls(
            image_size=input_size_dec,
            h_size=h_sizes[::-1] if symmetric_decoder else h_sizes,
            # David uses an asymmetric decoder
            # the model has more parameters with symmetric decoder and also I observed checkerboard patterns
            z_dim=z_dim,
            kernel_size=kernel_size,
            strides=strides[::-1] if symmetric_decoder else strides,
            normalization_op=normalization_op,
            to_1x1=to_1x1,
            upsample_op=upsample_op,
            conv_params=conv_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
        )
        self.hidden_size = self.enc.output_size

    def forward(self, inpt, sample=True, ret_y=False, **kwargs):
        y1 = self.enc(inpt, **kwargs)

        mu, log_std = torch.chunk(y1, 2, dim=1)

        std = torch.exp(log_std)
        z_dist = dist.Normal(mu, std)
        if sample:
            z_sample = z_dist.rsample()
        else:
            z_sample = mu

        x_rec = self.dec(z_sample)

        if ret_y:
            return x_rec, y1
        else:
            return x_rec, z_dist

    def generate_samples(self, num_samples, device):
        latent_size = (int(0.5 * self.hidden_size[0]), *self.hidden_size[1:])
        z = torch.randn(num_samples, *latent_size, device=device)
        return self.dec(z)

    def encode(self, inpt, **kwargs):
        enc = self.enc(inpt, **kwargs)
        mu, log_std = torch.chunk(enc, 2, dim=1)
        return mu, log_std

    def decode(self, inpt, **kwargs):
        x_rec = self.dec(inpt, **kwargs)
        return x_rec


class VAE3d(VAE):
    def __init__(
        self, conv_op=torch.nn.Conv3d, upsample_op=torch.nn.ConvTranspose3d, *args, **kwargs
    ):
        super().__init__(conv_op=conv_op, upsample_op=upsample_op, *args, **kwargs)
