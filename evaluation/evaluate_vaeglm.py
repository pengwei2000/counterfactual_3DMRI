import torch
import numpy as np
from diffusers import AutoencoderKLCogVideoX
import os
import sys
sys.path.append("../")
from datasets.adni_3d.dataset import ADNI3d
from datasets.adni_3d.dataset import normalize as normalize_adni
from datasets.ncanda.dataset import normalize as normalize_ncanda
from datasets.ncanda.dataset import NCANDA
from datasets.transforms import ReturnDictTransform
from evaluation.metrics.ssim import SSIM


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

def l1_distance(images, steps, embedding=None):

    distances = {}
    for idx in range(len(steps)):
        step = steps[idx]
        if embedding == 'ssim':
            ssim_caller = SSIM(data_dim=3, keep_batch_dim=True).cuda()
            distances[step] = ssim_caller((images[idx+1]+1)/2,(images[0]+1)/2).cpu().numpy()
        elif embedding is None:
            distances[step] = np.mean(np.abs(images[idx+1].cpu().numpy() - images[0].cpu().numpy()), axis=(1,2,3,4))         # shape: (batch,)
    return distances

MIN_MAX = {
    'image': [-1.0, 1.0],
    'age': [55.25, 96.],
    'fro': [136061.0, 212592.0],
    'par':[88797.0, 139170.0],
    'tem':[94958.0, 175299.0],
    'occ':[36153.0, 70856.0],
    'cin':[20419.0, 34470.0],
    'ins':[12401.0, 22950.0],
    'ven':[9527.0, 157058.0]
}

def normalize(value, name):
    if name not in MIN_MAX or name == 'image':
        raise ValueError
    value = (value - MIN_MAX[name][0]) / (MIN_MAX[name][1] - MIN_MAX[name][0])
    value = value * 2 - 1  # [0,1] -> [-1,1]
    return value

def norm_list(data):
    # Calculate mean and standard deviation
    mean = np.sum(data) / len(data)
    std_dev = np.std(data)
    # Normalize the list to z-scores
    normalized_data = ((data - mean) / std_dev).tolist()

    return normalized_data, mean, std_dev

def convert_metadata(Metadata, metadata_raw):
    Metadata_New = []
    for i in range(Metadata.shape[-1]):
        _, mean, std = norm_list(metadata_raw[:,i])
        vol = ((Metadata[:,i] - mean) / std).tolist()
        Metadata_New.append(vol)
    return Metadata_New

def get_kernel(Metadata):
    N = len(Metadata[0])
    metadata = np.ones((N,8))  # add residue here
    metadata[:,1] = Metadata[0] ## sex## we may donot need sex in our model, so put at there(May just asign the original number)
    metadata[:,2] = Metadata[1]## Age 
    metadata[:,3] = Metadata[2]# diagnosis 
    metadata[:,4] = Metadata[3]# svol
    metadata[:,5] = Metadata[4]# Frontal_raw, 
    metadata[:,6] = Metadata[5]# Insula_raw
    metadata[:,7] = Metadata[6]# Parietal_raw 
    
    cf_kernel = 0
    return cf_kernel, torch.from_numpy(metadata).float()

def GLM(X_feature, Metadata):
    N = X_feature.shape[0]
    X_vec = X_feature
    cf_kernel, Metadata = get_kernel(Metadata)
    Meta_T = torch.transpose(Metadata, 0, 1) #(8xN)
    pinv = torch.mm(cf_kernel, Meta_T)
    Beta = torch.mm(pinv, X_vec)   #(8xN vs Nx228096)
    X_r = torch.mm(Metadata, Beta) #torch.mm(X_batch[:, 1:], B[1:]) 
    residual = X_vec -  X_r
    residual = residual.reshape(X_feature.shape)

    return residual, Beta, Metadata

##########################################
# evaluate effectiveness

attr = ['fro', 'par', 'tem', 'occ', 'cin', 'ins', 'ven']
directory = f"SynthSeg/evaluation/age/vaeglm"
if not os.path.exists(directory):
    os.makedirs(directory)
vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b", subfolder="vae", torch_dtype=torch.bfloat16).to('cuda')
with torch.no_grad():
    for i in range(len(attr)):
        do_parent = attr[i]
        print(do_parent)
        ### Load the activations and passed to the decoder. The activations are derived from the GLM, given the interventions.
        z = torch.load(f'models\vaeglm\tmp\ncanda\embedding_activation_testset_doparent_{do_parent}')
        print(z.shape)
        z = z.to(dtype=torch.bfloat16, device='cuda')  # 460,1,16,...
        for j in range(z.shape[0]): #460
            dec = vae.decode(z[j]).sample.to(torch.float32) # (1,3,144,176,144)
            dec = torch.permute(torch.mean(dec, 1, False).squeeze(),(2,1,0)) # 144,176,144
            file_path = os.path.join(directory, f"counterfactual_{do_parent}_{str(j).zfill(3)}.npz")
            vol_data = dec.cpu().numpy()
            assert vol_data.shape == (144,176,144)
            np.savez_compressed(file_path, vol_data=vol_data)
    print('done')


#########################################################################
# evaluate reversibility
attr = ['fro', 'par', 'tem', 'occ', 'cin', 'ins', 'ven']
directory = f"SynthSeg/evaluation/reversibility/vaeglm"
if not os.path.exists(directory):
    os.makedirs(directory)
vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b", subfolder="vae", torch_dtype=torch.bfloat16).to('cuda')
with torch.no_grad():
    Beta = torch.load('models\vaeglm\tmp\Beta')  # Beta does not change
    metadata_raw = torch.load('models\vaeglm\tmp\test_metadata').numpy()
    print(f'metadata_raw has shape {metadata_raw.shape}')
    # z = torch.from_numpy(np.load('models\vaeglm\tmp\test_feature_extracted_by_CogVideoX-5b.npy')).to(torch.bfloat16).cuda()

    for i in range(len(attr)):
        residual = torch.load('models\vaeglm\tmp\Residual_test').to(torch.bfloat16)
        print(f'residual has shape {residual.shape}')
        metadata_interv = np.load('counterfactual-benchmark/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm3d/evaluate_effectiveness_target_metadata.npz')
        do_parent = attr[i]
        print(f'calculating reversibility for pa: {do_parent}')
        metadata = metadata_interv[do_parent]  #460,7
        metadata_interv = np.zeros((metadata.shape[0],metadata.shape[1]))
        for j in range(metadata_interv.shape[-1]):
            metadata_interv[:,j] = normalize(value=metadata[:,j],name=attr[j])
        for k in range(1,4):
            print(f'reversibility cycle: {k}')
            Metadata = convert_metadata(metadata_interv, metadata_raw)
            _, Metadata = get_kernel(Metadata)
            X_feature = residual + torch.mm(Metadata, Beta)
            z = X_feature.reshape(Metadata.shape[0], 1, 16, 36, 22, 18)
            prior_new = torch.empty(Metadata.shape[0], 1, 16, 36, 22, 18)

            print('Caltulating forward reversibility: do_parent from A to B')
            for j in range(z.shape[0]): #460
                prior = z[j].to(torch.bfloat16).cuda()       # 1,16,36,22,18
                assert prior.shape == (1,16,36,22,18)
                dec = vae.decode(prior).sample # (1,3,144,176,144)
                assert dec.shape == (1,3, 144,176,144)

                posterior = vae.encode(dec).latent_dist
                prior = posterior.mode()
                assert prior.shape == (1,16,36,22,18)
                prior_new[j] = prior  # update the j-th item of prior_new
            ### After have full piror new, 460,1,16,36,...., need to calculate the residual
            prior_new = prior_new.reshape(prior_new.shape[0], -1) 
            residual = prior_new - torch.mm(Metadata, Beta)
            ## update metadata to "metadata_raw", now doing the second half of the reversibility: back to raw
            Metadata = convert_metadata(metadata_raw, metadata_raw)
            _,Metadata = get_kernel(Metadata)
            X_feature = residual + torch.mm(Metadata, Beta)
            z = X_feature.reshape(Metadata.shape[0], 1, 16, 36, 22, 18)

            prior_new = torch.empty(Metadata.shape[0], 1, 16, 36, 22, 18)

            print('Caltulating backward reversibility: do_parent from B to A')
            for j in range(z.shape[0]): #460
                prior = z[j].to(torch.bfloat16).cuda()       # 1,16,36,22,18
                assert prior.shape == (1,16,36,22,18)
                dec = vae.decode(prior).sample # (1,3,144,176,144)
                assert dec.shape == (1,3, 144,176,144)
                vol_data = torch.permute(torch.mean(dec.to(torch.float32), 1, False).squeeze(),(2,1,0)).cpu().numpy()  # 144,176,144
                file_path = os.path.join(directory, f"counterfactual_doparent{do_parent}_cycle-{k}_sample_{str(j).zfill(3)}.npz")
                np.savez_compressed(file_path, vol_data=vol_data)
                if k != 3:
                    posterior = vae.encode(dec).latent_dist
                    prior = posterior.mode()
                    assert prior.shape == (1,16,36,22,18)
                    prior_new[j] = prior  # update the j-th item of prior_new
            print(f'saved images at cycle {k} for all 460 sample')
            if k != 3:
                prior_new = prior_new.reshape(prior_new.shape[0], -1)
                residual = prior_new - torch.mm(Metadata, Beta)
    print('done')



cycles = [1,3]

attr = ['fro', 'par', 'tem', 'occ', 'cin', 'ins', 'ven']
folder_path = f"SynthSeg/evaluation/reversibility/vaeglm"
for i in range(len(attr)):
    reversibility_scores = []
    reversibility_scores = []
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
            print(f'Successfully loaded counterfactual of cycle {j}...')
        assert len(images) == 3
        print('Calculating the reversibility score in this batch for you...')
        score_batch = l1_distance(images, steps=cycles)
        reversibility_scores.append(score_batch)
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
    reversibility_scores = {cycle: np.concatenate([reversibility_batch[cycle] for reversibility_batch in reversibility_scores]) for cycle in cycles}
    for cycle in cycles:
        print(f"Average reversibility score for {cycle} cycles: mean {round(np.mean(reversibility_scores[cycle]), 3):.3f} std {round(np.std(reversibility_scores[cycle]), 3):.3f}")
    images_t = np.concatenate(images_t) # (allbatch, 1, 144, 176 * (max(cycle)+1))
    images_c = np.concatenate(images_c)
    images_s = np.concatenate(images_s)

