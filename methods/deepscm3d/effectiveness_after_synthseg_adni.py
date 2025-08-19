import pandas as pd
import numpy as np

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
        return value
    value = value / MIN_MAX[name][1]
    return value

def unnormalize(value, name):
    if name not in MIN_MAX or name == 'image':
        raise ValueError
    value = (value + 1) / 2  # [-1,1] -> [0,1]
    # [0,1] -> [min,max]
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) +  MIN_MAX[name][0]
    return value

frontal_lh = ['ctx-lh-caudalmiddlefrontal','ctx-lh-lateralorbitofrontal','ctx-lh-medialorbitofrontal','ctx-lh-frontalpole',
           'ctx-lh-precentral', 'ctx-lh-paracentral','ctx-lh-rostralmiddlefrontal','ctx-lh-superiorfrontal',
           'ctx-lh-parstriangularis', 'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis']
parietal_lh = ['ctx-lh-inferiorparietal', 'ctx-lh-superiorparietal','ctx-lh-supramarginal','ctx-lh-postcentral','ctx-lh-precuneus']
temporal_lh = ['ctx-lh-fusiform','ctx-lh-entorhinal','ctx-lh-inferiortemporal','ctx-lh-middletemporal', 'ctx-lh-parahippocampal',
            'ctx-lh-superiortemporal','ctx-lh-temporalpole','ctx-lh-transversetemporal', 'ctx-lh-bankssts']
occipital_lh = ['ctx-lh-cuneus','ctx-lh-lateraloccipital','ctx-lh-lingual','ctx-lh-pericalcarine',]
cingulate_lh = ['ctx-lh-caudalanteriorcingulate', 'ctx-lh-isthmuscingulate', 'ctx-lh-posteriorcingulate','ctx-lh-rostralanteriorcingulate']
insula_lh = ['ctx-lh-insula']
ventricle = ['left lateral ventricle', '3rd ventricle','4th ventricle','right lateral ventricle']
ctx_lh = [frontal_lh, parietal_lh, temporal_lh, occipital_lh, cingulate_lh, insula_lh]
ctx = [[],[],[],[],[],[]]
for idx in range(len(ctx_lh)):
    for part in ctx_lh[idx]:
        ctx[idx].append(part)
        ctx[idx].append(part.replace('-lh-','-rh-'))

vol_path = '/SynthSeg/evaluation/hagan_volumes.csv'
df = pd.read_csv(vol_path)
brain_area = df.columns.tolist()

target_arrays = np.load('/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm3d/evaluate_effectiveness_target_metadata.npz')
attribute_list = ['fro','par','tem','occ','cin','ins','ven']
for i in range(7):
    do_parent = attribute_list[i]
    filtered_df = df[df["subject"].str.contains(do_parent)]
    assert len(filtered_df) == 460
    # filtered_df = df[df["subject"].str.contains('npz')]
    filtered_df = filtered_df.sort_values(by='subject')
    target = target_arrays[do_parent]      # (460,7)
    predictions = np.zeros((target.shape[0],len(attribute_list)))   # (460,7)

    for j in range(len(filtered_df)):
        for k in range(7):
            if i < 6:
                ctx_part = ctx[i]
            else:
                ctx_part = ventricle
            vol = 0
            for sub_part in ctx_part:
                vol += float(filtered_df.iloc[j,brain_area.index(sub_part)])
            predictions[j,k] = vol

    mae = np.abs(target - predictions)
    effectiveness_scores = {}
    for j in range(len(attribute_list)):
        pa = attribute_list[j]
        effectiveness_scores[attribute_list[j]] = (round(np.mean(normalize(mae[:,j],pa)),3), round(np.std(normalize(mae[:,j],pa)),3))

    print(f"Effectiveness score do({do_parent}): {effectiveness_scores}")

