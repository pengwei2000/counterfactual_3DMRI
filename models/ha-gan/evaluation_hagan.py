from tqdm import tqdm
import torch
import numpy as np
import os

from Model_HA_GAN_144_192_144 import Generator, Encoder, Sub_Encoder
from utils import trim_state_dict_name
import sys
sys.path.append("../../")
from evaluation.metrics.ssim import SSIM
from datasets.adni_3d.dataset import ADNI3d
from datasets.adni_3d.dataset import normalize as normalize_adni
from datasets.ncanda.dataset import NCANDA
from datasets.ncanda.dataset import normalize as normalize_ncanda
from datasets.transforms import ReturnDictTransform
attribute_size={
        "age": 1,
        "sex": 2,
        "diagnosis": 5,
        "fro": 1,
        "par": 1,
        "tem": 1,
        "occ": 1,
        "cin": 1,
        "ins": 1,
        "ven": 1
    }
transform = ReturnDictTransform(attribute_size)
data = ADNI3d(attribute_size,split="test",transform=transform)
test_data_loader = torch.utils.data.DataLoader(data, batch_size=460, shuffle=False, num_workers=8)

def l1_distance(images, steps, embedding=None, embedding_fn=None):

    distances = {}
    for idx in range(len(steps)):
        step = steps[idx]
        if embedding == 'ssim':
            ssim_caller = SSIM(data_dim=3, keep_batch_dim=True).cuda()
            distances[step] = ssim_caller((images[idx+1]+1)/2,(images[0]+1)/2).cpu().numpy()
        elif embedding is None:
            distances[step] = np.mean(np.abs(images[idx+1].cpu().numpy() - images[0].cpu().numpy()), axis=(1,2,3,4))         # shape: (batch,)
    return distances

checkpoint_root= "models/ha-gan/checkpoint/"
exp_name = 'HA_GAN_condition_on_anatomy'
ckpt_fname = "G_iterSTEP.pth"
step = 38500
# save_dir = "./BrainSample/HA-GAN_retrained"

G = Generator(mode='eval', latent_dim=1024, num_class=7).cuda()
E = Encoder().cuda()
Sub_E = Sub_Encoder().cuda()

ckpt_path = checkpoint_root + exp_name + "/G_iter"+str(step)+".pth"
ckpt = torch.load(ckpt_path)['model']
ckpt = trim_state_dict_name(ckpt)
G.load_state_dict(ckpt)

ckpt_path = checkpoint_root+exp_name+"/E_iter"+str(step)+".pth"
ckpt = torch.load(ckpt_path)['model']
ckpt = trim_state_dict_name(ckpt)
E.load_state_dict(ckpt)

ckpt_path = checkpoint_root+exp_name+"/Sub_E_iter"+str(step)+".pth"
ckpt = torch.load(ckpt_path)['model']
ckpt = trim_state_dict_name(ckpt)
Sub_E.load_state_dict(ckpt)

print(exp_name, step, "step weights loaded.")
del ckpt

G = G.cuda()
E = E.cuda()
Sub_E = Sub_E.cuda()

G.eval()
E.eval()
Sub_E.eval()
###########################################################
# composition

data = ADNI3d(attribute_size,split="test")
test_data_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=False, num_workers=8)

cycles = [1,10]
composition_scores = []
images_t = []
images_c = []
images_s = []
with torch.no_grad():
    for idx, factual_batch in enumerate(tqdm(test_data_loader)):
        
        batch_size = factual_batch[0].shape[0]
        # print(f'attr has shape {factual_batch[1].shape[1]}, trimming for vol attr only')
        real_images = factual_batch[0]
        images = [real_images]
        attr = factual_batch[1][:,-7:].cuda()
        assert attr.shape[1] == 7

        for i in range(max(cycles)):

            z_hat_i_list = []
            # Process all sub-volume and concatenate
            for crop_idx_i in range(0,144,144//6): # TODO: hardcoded
                real_images_crop_i = real_images[:,:,crop_idx_i:crop_idx_i+144//6,:,:].cuda()
                # print("~~~~~~~~Cropping~~~~~~~~`!")
                # print(real_images_crop_i.shape)
                z_hat_i = E(real_images_crop_i)
                # print(z_hat_i.shape)
                assert z_hat_i.shape[2:] == (6, 44, 36)
                z_hat_i_list.append(z_hat_i)
            z_hat = torch.cat(z_hat_i_list, dim=2)   # 36,44,36
            # print(z_hat.shape)
            sub_z_hat = Sub_E(z_hat)
            # print(sub_z_hat.shape)  # bz, 1024
            # Reconstruction
            sub_x_hat_rec = G(sub_z_hat, class_label=attr)  # bz, 1, 144,176,144
            # print(sub_x_hat_rec.shape)
            assert sub_x_hat_rec.shape == real_images.shape
            images.append(sub_x_hat_rec)
            real_images = sub_x_hat_rec

        t_idx = images[0].shape[-1] // 2
        c_idx = images[0].shape[-2] // 2
        s_idx = images[0].shape[-3] // 2
        # stack images for all cycles
        image_batch_t = np.concatenate([image[:,:,:,:,t_idx].cpu().numpy() for image in images], axis=3)  # (batch, 1, 144, 176 * (max(cycles)+1) )  
        image_batch_c = np.concatenate([image[:,:,:,c_idx,:].cpu().numpy() for image in images], axis=3) # (batch, 1, 144, 144 * (max(cycles)+1) )  
        image_batch_s = np.concatenate([image[:,:,s_idx,:,:].cpu().numpy() for image in images], axis=3) # (batch, 1, 144, 144 * (max(cycles)+1) )  
    
        images_t.append(image_batch_t)
        images_c.append(image_batch_c)
        images_s.append(image_batch_s)
        assert len(images) == 11
        # print('Calculating the composition score in this batch for you...')
        score_batch = l1_distance(images, steps=cycles, embedding='ssim')

        composition_scores.append(score_batch)

composition_scores = {cycle: np.concatenate([composition_batch[cycle] for composition_batch in composition_scores]) for cycle in cycles}
images_t = np.concatenate(images_t) # (allbatch, 1, 144, 176 * (max(cycle)+1))
images_c = np.concatenate(images_c)
images_s = np.concatenate(images_s)

for cycle in cycles:
    print(f"Average composition score for {cycle} cycles: mean {round(np.mean(composition_scores[cycle]), 3):.3f} std {round(np.std(composition_scores[cycle]), 3):.3f}")


#######################################################################
# evaluate reversibility
attr = ['fro', 'par', 'tem', 'occ', 'cin', 'ins', 'ven']
directory = f"/SynthSeg/evaluation/reversibility/hagan"
if not os.path.exists(directory):
    os.makedirs(directory)

with torch.no_grad():
    
    for i in range(len(attr)):
        metadata_interv = np.load('counterfactual-benchmark/counterfactual_benchmark/methods/deepscm3d/evaluate_effectiveness_target_metadata.npz')
        do_parent = attr[i]
        print(f'calculating reversibility for pa: {do_parent}')
        metadata = metadata_interv[do_parent]  #460,7
        metadata_interv = np.zeros((metadata.shape[0],metadata.shape[1]))  # 460,7
        for j in range(metadata_interv.shape[-1]):
            metadata_interv[:,j] = normalize(value=metadata[:,j],name=attr[j])
        

        for idx, factual_batch in enumerate(tqdm(test_data_loader)):
            batch_size = factual_batch[0].shape[0]
            if idx == 0:
                print(f'using batch size {batch_size}')

            real_images = factual_batch[0]
            metadata_raw = factual_batch[1][:,-7:].cuda()
            metadata = torch.from_numpy(metadata_interv[idx*batch_size:idx*batch_size+batch_size,:]).to(torch.float32).cuda()
            assert metadata_raw.shape == (4,7)
            assert metadata.shape == (4,7)
            for k in range(1,4):

                z_hat_i_list = []
                # Process all sub-volume and concatenate
                for crop_idx_i in range(0,144,144//6): # TODO: hardcoded
                    real_images_crop_i = real_images[:,:,crop_idx_i:crop_idx_i+144//6,:,:].cuda()
                    z_hat_i = E(real_images_crop_i)
                    assert z_hat_i.shape[2:] == (6, 44, 36)
                    z_hat_i_list.append(z_hat_i)
                z_hat = torch.cat(z_hat_i_list, dim=2)   # 36,44,36
                sub_z_hat = Sub_E(z_hat)
                sub_x_hat_rec = G(sub_z_hat, class_label=metadata)  # bz, 1, 144,176,144
                assert sub_x_hat_rec.shape == real_images.shape

                real_images = sub_x_hat_rec

                z_hat_i_list = []
                # Process all sub-volume and concatenate
                for crop_idx_i in range(0,144,144//6): # TODO: hardcoded
                    real_images_crop_i = real_images[:,:,crop_idx_i:crop_idx_i+144//6,:,:].cuda()
                    z_hat_i = E(real_images_crop_i)
                    assert z_hat_i.shape[2:] == (6, 44, 36)
                    z_hat_i_list.append(z_hat_i)
                z_hat = torch.cat(z_hat_i_list, dim=2)   # 36,44,36
                sub_z_hat = Sub_E(z_hat)
                sub_x_hat_rec = G(sub_z_hat, class_label=metadata_raw)  # bz, 1, 144,176,144
                assert sub_x_hat_rec.shape == real_images.shape
                
                # vol_data = sub_x_hat_rec.squeeze().cpu().numpy()  # 144,176,144
                for bz in range(batch_size):
                    vol_data = sub_x_hat_rec.squeeze()[bz].cpu().numpy()
                    assert vol_data.shape == (144,176,144)
                    file_path = os.path.join(directory, f"counterfactual_doparent{do_parent}_cycle-{k}_sample_{str(idx*batch_size+bz).zfill(3)}.npz")
                    np.savez_compressed(file_path, vol_data=vol_data)

                real_images = sub_x_hat_rec

    print('done')


##############################################################
# read and calculate reversibility
test_data_loader = torch.utils.data.DataLoader(data, batch_size=460, shuffle=False, num_workers=8)
cycles = [1,3]
attr = ['fro', 'par', 'tem', 'occ', 'cin', 'ins', 'ven']
folder_path = f"/SynthSeg/evaluation/reversibility/hagan"

for i in range(len(attr)):
    composition_scores = []
    images_t = []
    images_c = []
    images_s = []
    do_parent = attr[i]
    print(f'calculation do parent {do_parent}')
    for idx, factual_batch in enumerate(test_data_loader):
        factual_batch = factual_batch['image']   # 460, 1, 144,176,144 
        images = [factual_batch]
        batch_size = factual_batch.shape[0]
        for j in cycles:
            counterfactual_batch = torch.empty(batch_size, factual_batch.shape[1], factual_batch.shape[2], factual_batch.shape[3], factual_batch.shape[4])
            for k in range(batch_size):
                counterfactual_i = np.load(folder_path+f'/counterfactual_doparent{do_parent}_cycle-{j}_sample_{str(idx*batch_size + k).zfill(3)}.npz')
                counterfactual_batch[k,0,:,:,:] = torch.from_numpy(counterfactual_i['vol_data'])
            images.append(counterfactual_batch)

        assert len(images) == 3
        score_batch = l1_distance(images, steps=cycles)
        composition_scores.append(score_batch)

        t_idx = images[0].shape[-1] // 2
        c_idx = images[0].shape[-2] // 2
        s_idx = images[0].shape[-3] // 2
        # stack images for all cycles
        image_batch_t = np.concatenate([image[:,:,:,:,t_idx].cpu().numpy() for image in images], axis=3)  # (batch, 1, 144, 176 * (max(cycles)+1) )  
        image_batch_c = np.concatenate([image[:,:,:,c_idx,:].cpu().numpy() for image in images], axis=3) # (batch, 1, 144, 144 * (max(cycles)+1) )  
        image_batch_s = np.concatenate([image[:,:,s_idx,:,:].cpu().numpy() for image in images], axis=3) # (batch, 1, 144, 144 * (max(cycles)+1) )  
        images_t.append(image_batch_t)
        images_c.append(image_batch_c)
        images_s.append(image_batch_s)

    composition_scores = {cycle: np.concatenate([composition_batch[cycle] for composition_batch in composition_scores]) for cycle in cycles}
    for cycle in cycles:
        print(f"Average reversibility score for {cycle} cycles: mean {round(np.mean(composition_scores[cycle]), 3):.3f} std {round(np.std(composition_scores[cycle]), 3):.3f}")
    images_t = np.concatenate(images_t) # (allbatch, 1, 144, 176 * (max(cycle)+1))
    images_c = np.concatenate(images_c)
    images_s = np.concatenate(images_s)

#######################################
# evaluate effectiveness

attribute_size={
        "age": 1,
        "sex": 2,
        "diagnosis": 5,
        "fro": 1,
        "par": 1,
        "tem": 1,
        "occ": 1,
        "cin": 1,
        "ins": 1,
        "ven": 1
    }

data = ADNI3d(attribute_size,split="test")
test_data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
directory = f"/SynthSeg/evaluation/age/hagan"
if not os.path.exists(directory):
    os.makedirs(directory)
attr = ['fro', 'par', 'tem', 'occ', 'cin', 'ins', 'ven']
with torch.no_grad():
    for idx, factual_batch in enumerate(tqdm(test_data_loader)):
        if idx != 0:
            break
        real_images = factual_batch[0].reshape(1,1,144,176,144)
        real_images = real_images.repeat(2,1,1,1,1)
        print(real_images.shape)
        for i in range(7):
            metadata = np.load(f'/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm3d/do_age/gan-doparent-age-metadata-{i}.npy')
            print(metadata)
            metadata = np.tile(metadata, (2,1))
            # print(metadata.shape)  # 1,7
            metadata_interv = np.zeros((metadata.shape[0],metadata.shape[1]))
            for j in range(metadata_interv.shape[-1]):
                metadata_interv[:,j] = normalize_adni(value=metadata[:,j],name=attr[j])

            metadata = torch.from_numpy(metadata_interv).to(torch.float32).cuda()
            print(metadata.shape)

            z_hat_i_list = []
            # Process all sub-volume and concatenate
            for crop_idx_i in range(0,144,144//6): # TODO: hardcoded
                real_images_crop_i = real_images[:,:,crop_idx_i:crop_idx_i+144//6,:,:].cuda()
                z_hat_i = E(real_images_crop_i)
                assert z_hat_i.shape[2:] == (6, 44, 36)
                z_hat_i_list.append(z_hat_i)
            z_hat = torch.cat(z_hat_i_list, dim=2)   # 36,44,36
            sub_z_hat = Sub_E(z_hat)
            sub_x_hat_rec = G(sub_z_hat, class_label=metadata)  # bz, 1, 144,176,144
            assert sub_x_hat_rec.shape == real_images.shape

            vol_data = sub_x_hat_rec[0].squeeze().cpu().numpy()
            assert vol_data.shape == (144,176,144)
            file_path = os.path.join(directory, f"counterfactual_age_{i}.npz")
            np.savez_compressed(file_path, vol_data=vol_data)