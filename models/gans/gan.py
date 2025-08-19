"""Generic conditional GAN class without specified encoder and decoder archtitecture. Adapted from https://github.com/gulnazaki/counterfactual-benchmark"""
from torch import nn
import pytorch_lightning as pl
import torch
import numpy as np
import sys
import os
from diffusers import AutoencoderKLCogVideoX
from models.utils import init_weights

sys.path.append("../../")


class CondGAN(pl.LightningModule):
    def __init__(self, encoder, decoder, discriminator, latent_dim, d_updates_per_g_update, gradient_clip_val, finetune, lr=1e-4, name="image_gan"):
        super().__init__()
        self.latent_dim = latent_dim
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.finetune = finetune
        self.lr = lr
        self.d_updates_per_g_update = d_updates_per_g_update
        self.gradient_clip_val = gradient_clip_val
        self.automatic_optimization = False
        if self.finetune == 1:
            self.set_to_finetune()
        else:
            self.apply(init_weights)


    def encode(self, x, cond):
        return self.encoder(x, cond)

    def decode(self, u, cond, enc1, enc2, enc3, enc4, enc5):
        return self.decoder(u, cond, enc1, enc2, enc3, enc4, enc5)

    def discriminate(self, x, u, cond):
        return self.discriminator(x, u, cond)

    def gan_loss(self, y_hat, y):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat.squeeze(), y.squeeze())
        return loss

    def l1_loss(self, z, ex, reduction="mean"):
        criterion = nn.L1Loss(reduction=reduction)
        loss = criterion(z, ex)
        return loss

    def l2_loss(self, x, xr):
        criterion = nn.MSELoss()
        loss = criterion(x.squeeze(), xr.squeeze())
        return loss

    def forward_enc(self, x, cond):
        ex, enc1, enc2, enc3, enc4, enc5 = self.encode(x, cond)
        return ex, enc1, enc2, enc3, enc4, enc5

    def forward_dec(self, u, cond, enc1, enc2, enc3, enc4, enc5):
        gu = self.decode(u, cond, enc1, enc2, enc3, enc4, enc5)
        return gu

    def forward_discr(self, x, z, cond):
        return self.discriminate(x, z, cond)

    def configure_optimizers(self):
        if self.finetune == 0:
            optimizer_E = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                        lr=self.lr, betas=(0.5, 0.999))
            optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                        lr=self.lr, betas=(0.5, 0.999))
            return optimizer_E, optimizer_D
        else:
            optimizer_E = torch.optim.Adam(list(self.encoder.parameters()),
                                        lr=self.lr, betas=(0.5, 0.999))
            return optimizer_E


    def set_to_finetune(self):
        # Freeze the parameters of the decoder & discriminator
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def training_step(self, train_batch, batch_idx):
        # may happen if the last batch is of size 1 (dropout error)
        if train_batch[0].shape[0] == 1 and self.trainer.is_last_batch:
            return

        if self.finetune == 1:
            optimizer_E  = self.optimizers()
            x, cond = train_batch

            # latent loss
            z_mean = torch.zeros((len(x), self.latent_dim, 1, 1, 1)).float()
            z = torch.normal(z_mean, z_mean + 1).to(x.device)

            ex, enc1, enc2, enc3, enc4, enc5 = self.forward_enc(x, cond)
            gz = self.forward_dec(z, cond, enc1, enc2, enc3, enc4, enc5)
            egz,_,_,_,_,_ = self.forward_enc(gz,cond)
            latent_loss = self.l2_loss(z, egz)
            gex = self.forward_dec(ex, cond, enc1, enc2, enc3, enc4, enc5)
            image_loss = self.l1_loss(x, gex)

            loss = latent_loss + image_loss

            optimizer_E.zero_grad()
            self.manual_backward(loss)
            optimizer_E.step()

            self.log_dict({"latent_loss": latent_loss, "image_loss": image_loss}, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        else:

            x, cond = train_batch

            batch_size = x.shape[0]

            optimizer_E, optimizer_D = self.optimizers()
            valid = torch.ones((batch_size, 1), device=x.device)
            fake = torch.zeros((batch_size, 1), device=x.device)

            z_mean = torch.zeros((len(x), self.latent_dim, 1, 1, 1)).float()
            z = torch.normal(z_mean, z_mean + 1).to(x.device)

            ex, enc1, enc2, enc3, enc4, enc5 = self.forward_enc(x, cond)
            gz = self.forward_dec(z, cond, enc1, enc2, enc3, enc4, enc5)
            gex = self.forward_dec(ex, cond, enc1, enc2, enc3, enc4, enc5)
            # ##########################
            # # Optimize Discriminator #
            # ##########################

            self.toggle_optimizer(optimizer_D)

            D_valid = self.forward_discr(x, ex.detach(), cond)
            loss_D_valid = self.gan_loss(D_valid, valid)

            D_fake = self.forward_discr(gz.detach(), z, cond)
            loss_D_fake = self.gan_loss(D_fake, fake)

            loss_D = loss_D_valid + loss_D_fake

            optimizer_D.zero_grad()
            self.manual_backward(loss_D)
            optimizer_D.step()
            self.untoggle_optimizer(optimizer_D)

            ##############################
            # Optimize Encoder & Decoder #
            ##############################

            if batch_idx % self.d_updates_per_g_update == 0:
                self.toggle_optimizer(optimizer_E)

                EG_valid = self.forward_discr(gz, z, cond)
                loss_EG_valid = self.gan_loss(EG_valid, valid)

                EG_fake = self.forward_discr(x, ex, cond)
                loss_EG_fake = self.gan_loss(EG_fake, fake)

                loss_EG = loss_EG_valid + loss_EG_fake

                optimizer_E.zero_grad()
                self.manual_backward(loss_EG)
                #self.clip_gradients(optimizer_E, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
                optimizer_E.step()
                self.untoggle_optimizer(optimizer_E)
            else:
                loss_EG = None

            if loss_EG is not None:
                self.log_dict({"eg_loss": loss_EG, "d_loss": loss_D}, on_step=False, on_epoch=True, prog_bar=True)

            return loss_EG


    def validation_step(self, train_batch, batch_idx):

        x, cond = train_batch
        bz = x.shape[0]

        if self.finetune == 1:

            ex, enc1, enc2, enc3, enc4, enc5 = self.forward_enc(x, cond)
            # gz = self.forward_dec(z, cond, enc1, enc2, enc3, enc4, enc5)
            gex = self.forward_dec(ex, cond, enc1, enc2, enc3, enc4, enc5)
            vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b", subfolder="vae", torch_dtype=torch.bfloat16).to('cuda')
            assert gex.shape == (bz, 1, 144,176,144)
            xx = torch.permute(x, (0,1,4,3,2)).expand(-1,3,-1,-1,-1)
            gex = torch.permute(gex, (0,1,4,3,2)).expand(-1,3,-1,-1,-1)
            assert gex.shape == (bz, 3,144,176,144)
            posterior_x = vae.encode(xx.to(torch.bfloat16)).latent_dist
            z_x = posterior_x.mode().flatten()
            posterior_gex = vae.encode(gex.to(torch.bfloat16)).latent_dist
            z_gex = posterior_gex.mode().flatten()
            lpips_score = self.l1_loss(z_x, z_gex, reduction='sum')

            self.log("val_loss", lpips_score, on_step=False, on_epoch=True, prog_bar=True)
        else:
            # sample noise
            z_mean = torch.zeros((len(x), self.latent_dim, 1, 1, 1)).float()
            z = torch.normal(z_mean, z_mean + 1).to(x.device)
            ex, enc1, enc2, enc3, enc4, enc5 = self.forward_enc(x, cond)
            gz = self.forward_dec(z, cond, enc1, enc2, enc3, enc4, enc5)
   
            # gex = self.forward_dec(ex, cond, enc1, enc2, enc3, enc4, enc5)

            valid = torch.ones((bz, 1), device=x.device)
            fake = torch.zeros((bz, 1), device=x.device)

            EG_valid = self.forward_discr(gz, z, cond)
            loss_EG_valid = self.gan_loss(EG_valid, valid)
            EG_fake = self.forward_discr(x, ex, cond)
            loss_EG_fake = self.gan_loss(EG_fake, fake)

            loss_EG = loss_EG_valid + loss_EG_fake
            self.log("val_loss", loss_EG, on_step=False, on_epoch=True, prog_bar=True)

        epoch = self.current_epoch
        n_show = 2
        save_images_every = 1
        path = os.getcwd()
        image_output_path = os.path.join(path, 'training_images_gan' + ('_finetuned' if self.finetune == 1 else ''))
        os.makedirs(image_output_path, exist_ok=True)

        if batch_idx == 0 and epoch % save_images_every == 0:
            reals = []
            geners = []
            recons = []
            # generate images from same class as real ones
            for i in range(n_show):
                images = x[i]
                images = torch.unsqueeze(images, 0).float()

                attrs = cond[i]
                attrs = attrs.reshape((1, attrs.shape[0]))

                z_mean = torch.zeros((len(images), self.latent_dim, 1, 1, 1)).float()
                z = torch.normal(z_mean, z_mean + 1).to(x.device)

                ex, enc1, enc2, enc3, enc4, enc5 = self.encode(images, attrs)
                gener = self.decode(z, attrs, enc1, enc2, enc3, enc4, enc5)
                recon = self.decode(ex, attrs, enc1, enc2, enc3, enc4, enc5)
                real = images.cpu().numpy()

                # Get 2d slices here
                gener = gener[...,72].reshape(gener.shape[1], gener.shape[2], gener.shape[3]).cpu().numpy()
                recon = recon[...,72].reshape(recon.shape[1], recon.shape[2], recon.shape[3]).cpu().numpy()
                real = real[...,72][0]

                recons.append(recon)
                geners.append(gener)
                reals.append(real)

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
            fig.subplots_adjust(wspace=0.05, hspace=0)
            plt.rcParams.update({'font.size': 20})
            fig.suptitle('Epoch {}'.format(epoch))
            fig.text(0.04, 0.75, 'G(z, c)', ha='left')
            fig.text(0.04, 0.5, 'x', ha='left')
            fig.text(0.04, 0.25, 'G(E(x, c), c)', ha='left')

            if  geners[0].shape[0]==3:
                for i in range(n_show):
                    geners[i] = np.transpose(geners[i], (1, 2, 0))
                    ax[0, i].imshow(geners[i])
                    ax[0, i].axis('off')
                    real = np.transpose(reals[i], (1, 2, 0))
                    ax[1, i].imshow(real)
                    ax[1, i].axis('off')
                    recons[i] = np.transpose(recons[i], (1, 2, 0))
                    ax[2, i].imshow(recons[i])
                    ax[2, i].axis('off')

            else:
                for i in range(n_show):
                    ax[0, i].imshow(geners[i][0], cmap='gray', vmin=0, vmax=1)
                    ax[0, i].axis('off')
                    ax[1, i].imshow(reals[i][0], cmap='gray', vmin=0, vmax=1)
                    ax[1, i].axis('off')
                    ax[2, i].imshow(recons[i][0], cmap='gray', vmin=0, vmax=1)
                    ax[2, i].axis('off')

            plt.savefig(f'{image_output_path}/epoch-{epoch}.png', format='png')
            plt.close()

        return
