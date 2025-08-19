"""Generic conditional VAE class without specified encoder and decoder archtitecture. Adapted from https://github.com/gulnazaki/counterfactual-benchmark"""

from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
import torch
import numpy as np

import sys
sys.path.append("../../")
from models.utils import linear_warmup

@torch.jit.script
def sample_gaussian(loc, logscale):
    return loc + logscale.exp() * torch.randn_like(loc)

@torch.jit.script
def gaussian_kl(q_loc, q_logscale, p_loc, p_logscale):
    return (
        -0.5
        + p_logscale
        - q_logscale
        + 0.5
        * (q_logscale.exp().pow(2) + (q_loc - p_loc).pow(2))
        / p_logscale.exp().pow(2)
    )


class CondVAE(pl.LightningModule):
    def __init__(self, encoder, decoder, likelihood, latent_dim, beta=4, lr=1e-3, weight_decay=0.01, name="image_vae"):
        super().__init__()

        self.latent_dim = latent_dim
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood
        self.beta = beta
        self.lr = lr
        self.weight_decay = weight_decay

    def _vae_loss(self, h, x, mu, logvar, prior_mu, prior_var, beta):
        nll_pp = self.likelihood.nll(h, x)  # (batch,)
        kl_pp = gaussian_kl(mu, logvar, prior_mu, prior_var) # (bz, latent)
        kl_pp = kl_pp.sum(dim=-1) / np.prod(x.shape[1:])
        # print(f'The kl_pp is normalized by the number of voxels, which is {np.prod(x.shape[1:])}')
        loss = nll_pp.mean() + beta * kl_pp.mean()
        nll_slice = self.likelihood.nll_slice(h, x)
        return loss, nll_pp.mean(), kl_pp.mean(), nll_slice.mean()

    def forward(self, x, cond):
        mu_u, logvar_u = self.encoder(x, cond)
        u = sample_gaussian(mu_u, logvar_u)
        h = self.decoder(u, cond)
        # print(h.shape): (batch, latent, 32, 32, 32)
        return h, mu_u, logvar_u


    def abduct(self, x, parents):
        q_loc, q_logscale = self.encoder(x, cond=parents)  # q(z|x,pa)
        z = sample_gaussian(q_loc, q_logscale)
        return [z.detach()]


    def forward_latents(self, latents, parents, return_loc = False):
        h = self.decoder(cond=parents, u=latents[0])
        return self.likelihood.sample(h, return_loc)

    def encode(self, x, cond):
        z = self.abduct(x, cond)

        rec_loc, rec_scale = self.forward_latents(z, parents=cond, return_loc=True)

        eps = (x - rec_loc) / rec_scale.clamp(min=1e-12)

        return  z , eps

    def decode(self, u, cond):
        z , e  = u
        t_u = self.temperature if hasattr(self, 'temperature') else 0.1
        cf_loc, cf_scale = self.forward_latents(z, parents=cond, return_loc=True)

        cf_scale = cf_scale * t_u
        x = torch.clamp(cf_loc + cf_scale * e, min=-1, max=1)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=[0.9, 0.999])
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=linear_warmup(100)
        )
        print(f'Printing optimizer info below ----- {optimizer}')
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, train_batch, batch_idx):
        # may happen if the last batch is of size 1 (dropout error)
        if train_batch[0].shape[0] == 1 and self.trainer.is_last_batch:
            return

        x, cond = train_batch
        # print(x.shape)
        h, mu_u, logvar_u = self.forward(x, cond)
        # print(h.shape)
        prior_mu, prior_var = self.decoder.prior(cond)
        loss, nll_pp, kl_pp, nll_slice = self._vae_loss(h, x, mu_u, logvar_u, prior_mu, prior_var, beta=self.beta)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("nll_train", nll_pp, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("kl_train", kl_pp, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log("nll_slice_train", nll_slice, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, train_batch, batch_idx):
        x, cond = train_batch
        h, mu_u, logvar_u = self.forward(x, cond)
        prior_mu, prior_var = self.decoder.prior(cond)
        loss, nll_pp, kl_pp, nll_slice = self._vae_loss(h, x, mu_u, logvar_u, prior_mu, prior_var, beta=self.beta)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("nll_val", nll_pp, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("kl_val", kl_pp, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log("nll_slice_val", nll_slice, on_step=False, on_epoch=True, prog_bar=True)
        return loss


class DGaussNet(nn.Module):
    def __init__(self, latent_dim, fixed_logvar, input_channels):
        super().__init__()
        self.x_loc = nn.Conv3d(
            latent_dim, input_channels, kernel_size=1, stride=1
        )
        self.x_logscale = nn.Conv3d(
            latent_dim, input_channels, kernel_size=1, stride=1
        )

        assert fixed_logvar == "False" or type(fixed_logvar) == float, \
            f'fixed_logvar can either be "False" or a float value, not: {fixed_logvar}'
        if fixed_logvar != "False":
            nn.init.zeros_(self.x_logscale.weight)
            nn.init.constant_(self.x_logscale.bias, fixed_logvar)
            self.x_logscale.weight.requires_grad = False
            self.x_logscale.bias.requires_grad = False

    def forward(self, h):
        loc, logscale = self.x_loc(h), self.x_logscale(h).clamp(min=-9)
        # print(f'DGaussNet Please check the shape of mu {loc.shape} and sigma {logscale.shape}.')
        return loc, logscale

    def approx_cdf(self, x):
        return 0.5 * (
            1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def nll(self, h, x):
        # print(f'DGaussNet, nll, Please check the shape {x.shape}.')
        loc, logscale = self.forward(h)      # (bz, 1, 128, 128, 128)
        centered_x = x - loc
        inv_stdv = torch.exp(-logscale)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(
                x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
            ),
        )
        # print(f'DGaussNet, nll, Please check the log_probs shape {log_probs.shape}. The mean is computed along dim=(1,2,3,4)')
        return -1.0 * log_probs.mean(dim=(1, 2, 3, 4))
    
    def nll_slice(self, h, x):
        loc, logscale = self.forward(h)
        loc = loc[:,:,:,:,16]
        logscale = logscale[:,:,:,:,16]
        x = x[:,:,:,:,16]
        # print(loc.shape, logscale.shape, x.shape)  = (batch, 1, 32, 32)
        centered_x = x - loc
        inv_stdv = torch.exp(-logscale)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(
                x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
            ),
        )
        return -1.0 * log_probs.mean(dim=(1, 2, 3))

    def sample(
        self, h, return_loc: bool = True, t=None):
        if return_loc:
            x, logscale = self.forward(h)
        else:
            loc, logscale = self.forward(h, t)
            x = loc + torch.exp(logscale) * torch.randn_like(loc)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x, logscale.exp()
