import torch
import numpy as np
from metrics.ssim import SSIM
from tqdm import tqdm
import sys
sys.path.append("../")
from datasets.adni_3d.dataset import ADNI3d
from datasets.transforms import ReturnDictTransform
import os
from diffusers import AutoencoderKLCogVideoX

## generate and save composition counterfactuals

attr = ['fro', 'par', 'tem', 'occ', 'cin', 'ins', 'ven']
directory = f"/SynthSeg/evaluation/composition/vaeglm"
if not os.path.exists(directory):
    os.makedirs(directory)
vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b", subfolder="vae", torch_dtype=torch.bfloat16).to('cuda')
with torch.no_grad():
    z = torch.from_numpy(np.load('models\vaeglm\tmp\test_feature_extracted_by_CogVideoX-5b.npy')).to(torch.bfloat16).cuda()
    assert z.shape == (460,1,16,36,22,18)

    for j in range(z.shape[0]): #460
        prior = z[j]       # 1,16,36,22,18
        assert prior.shape == (1,16,36,22,18)
        for i in range(10):
            dec = vae.decode(prior).sample # (1,3,144,176,144)
            assert dec.shape == (1,3, 144,176,144)
            if i == 0 or i == 9:
                vol_data = torch.permute(torch.mean(dec.to(torch.float32), 1, False).squeeze(),(2,1,0)).cpu().numpy()  # 144,176,144
                file_path = os.path.join(directory, f"counterfactual_cycle-{i+1}_sample_{str(j).zfill(3)}.npz")
                np.savez_compressed(file_path, vol_data=vol_data)
                print(f'saved images at cycle {i+1} for sample {j}')
            posterior = vae.encode(dec).latent_dist
            prior = posterior.mode()
            assert prior.shape == (1,16,36,22,18)
    print('done')

## Read and evaluate composition counterfactuals

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
test_data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False, num_workers=8)

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


save_dir = "composition_samples"

composition_scores = []
cycles = [1,10]
counterfactual_dir = 'SynthSeg/evaluation/composition/vaeglm/'

for idx, factual_batch in enumerate(tqdm(test_data_loader)):
    factual_batch = factual_batch['image']   # 460, 1, 144,176,144 
    # print(f'factual batch has shape {factual_batch.shape}')
    images = [factual_batch]
    batch_size = factual_batch.shape[0]
    for i in cycles:
        counterfactual_batch = torch.empty(factual_batch.shape[0], 1, factual_batch.shape[2],factual_batch.shape[3],factual_batch.shape[4])
        for j in range(factual_batch.shape[0]):
            data = np.load(counterfactual_dir + f'counterfactual_cycle-{i}_sample_{str(idx*batch_size + j).zfill(3)}.npz')
            print(f'loaded file _cycle-{i}_sample_{str(idx*batch_size + j).zfill(3)}')
            data = torch.from_numpy(data['vol_data'])
            assert data.shape == (144,176,144)
            counterfactual_batch[j,0,:,:,:] = data
        counterfactual_batch[counterfactual_batch.isnan()] = -1.   # Now one small volume has bug, image value has nan
        images.append(counterfactual_batch)
        print(f'Successfully loaded counterfactual of cycle {i}...')
    assert len(images) == 3
    print('Calculating the composition score in this batch for you...')
    score_batch = l1_distance(images, steps=cycles, embedding='ssim')

    composition_scores.append(score_batch)

composition_scores = {cycle: np.concatenate([composition_batch[cycle] for composition_batch in composition_scores]) for cycle in cycles}

#os.makedirs(save_dir, exist_ok=True)
#save_selected_images_3d(images_t, images_c, images_s, composition_scores[cycles[-1]], save_dir=save_dir)

for cycle in cycles:
    print(f"Average composition score for {cycle} cycles: mean {round(np.mean(composition_scores[cycle]), 3):.3f} std {round(np.std(composition_scores[cycle]), 3):.3f}")

