# Evaluation of 3D Counterfactual Brain MRI Generation

Code to reproduce the paper, which is published on the [5th Deep Generative Models Workshop @ MICCAI 2025](https://dgm4miccai.github.io).

## Repository organization
```
counterfactual_benchmark/
│
├── datasets/                # Data loading, preprocessing, and transforms
│   ├── adni_3d/             # ADNI dataset loader and scripts
│   └── ncanda/              # NCANDA dataset loader and scripts
│
├── models/                  
│   ├── vaes/                # VAE and HVAE
│   ├── gans/                # GAN and GAN-Finetuned
│   ├── ha-gan/              # HA-GAN
│   ├── vaeglm/              # VAE+GLM
│   └── utils.py             # Model utilities
│
├── evaluation/              # Evaluation scripts and metrics
│   ├── metrics/             # Metric implementations (FID, SSIM, etc.)
│   ├── embeddings/          # Embedding models (VGG, ResNet, etc.)
│   └── ...                  # Evaluation code for VAE+GLM
│
└── methods/       
    └── deepscm3d/           # 3D SCM methods
        ├── configs/         # Configuration files
        └── scripts          # Training and experiment scripts
```

## How to run
### 1. Training
**1.1 Train a VAE/HVAE/GAN model**
```
python train.py -c configs/adni3d/vae.json
python train.py -c configs/adni3d/hvae.json
python train.py -c configs/adni3d/gan.json
```

**1.2 Train GLM model with existing 3D VAE:**

Save the metadata as GLM input:
```
python create_metadata.py
```

Extracts VAE features from 3D MRI scans using CogVideoX-5b and saves them to /tmp:
```
python vae.py --train_csv /path/to/train.csv --train_data_dir /path/to/train/ --test_csv /path/to/test.csv --test_data_dir /path/to/test/ --train_num 4118 --test_num 582
```

Train the GLM through:
```
python GLM.py --train_metadata /tmp/train_metadata --train_feature /tmp/train_feature_extracted_by_CogVideoX-5b.npy --test_metadata /tmp/test_metadata --test_feature /tmp/test_feature_extracted_by_CogVideoX-5b.npy --beta /tmp/Beta --train_residual /tmp/train_Residual --test_residual /tmp/test_Residual
```

**1.3 Train HA-GAN model:**
```
python models\ha-gan\train_MRI_High.py
```
### 2. Inference and Evaluation

**2.1 Composition and Reversibiltiy**

For VAE-GLM, simply run:
```
python evaluation\composition_vaeglm.py
```

For HA-GAN, simply run:
```
python models\ha-gan\evaluation_hagan.py
```

For other models:
```
python methods\deepscm3d\evaluate.py
```

**2.2 Effectiveness and Minimality**

Run the inference and save the counterfactual MRIs:
```
python methods\deepscm3d\evaluate_effectiveness.py -c ./configs/adni3d/gan.json --model ganfinetune
```

Run the SynthSeg pipeline to do the cortex parcellation on the counterfactual MRIs:
```
python /SynthSeg/scripts/commands/SynthSeg_predict.py --i /SynthSeg/evaluation --o /SynthSeg/evaluation_output --robust --parc --vol /SynthSeg/evaluation/gan_volumes.csv --qc /SynthSeg/evaluation/gan_qc.csv
```

Read and organize the effectiveness scores:
```
python methods\deepscm3d\effectiveness_after_synthseg_adni.py
```

For VAE-GLM, simply run:
```
python evaluation\evaluate_vaeglm.py
```

For HA-GAN, simply run:
```
python models\ha-gan\evaluation_hagan.py
```

**2.3 Realism**
```
python methods\deepscm3d\fid_score.py
```
**2.4 Generalizability**
Find possible intervention values for each attribute:
```
python methods\deepscm3d\find_different_value_vol_attr.py
```
Then follow steps in 3.1.
### 3. Datasets

**3.1 ADNI**
Following the Synthseg <https://github.com/BBillot/SynthSeg> pipeline to do the cortex parcellation on the 3D MRI scans. The result is stored in datasets\adni_3d\adni_metadata.csv. The dataset split index is stored in datasets\adni_3d\test_idx.npy and datasets\adni_3d\train_idx.npy, respectively.

**3.2 NCANDA**
Following the Synthseg <https://github.com/BBillot/SynthSeg> pipeline to do the cortex parcellation on the 3D MRI scans. The result is stored in datasets\ncanda\ncanda_test.csv.

## Credits
We borrowed code from <https://github.com/gulnazaki/counterfactual-benchmark> for VAE, HVAE, GAN model architectures and the training scripts.

We use SynthSeg <https://github.com/BBillot/SynthSeg> as an oracle to perform cortex parcellation on the real image and counterfactuals.

## Citation
If you find this work helpful in your research, cite:
```
    @misc{sun2025evaluation3dcounterfactualbrain,
        title={Evaluation of 3D Counterfactual Brain MRI Generation}, 
        author={Pengwei Sun and Wei Peng and Lun Yu Li and Yixin Wang and Kilian M. Pohl},
        year={2025},
        eprint={2508.02880},
        archivePrefix={arXiv},
        primaryClass={eess.IV},
        url={https://arxiv.org/abs/2508.02880}, 
    }
```