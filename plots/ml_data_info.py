import h5py
import re
import fact.io
from sklearn.metrics import roc_auc_score, r2_score
import numpy as np
from ruamel.yaml import YAML
yaml = YAML(typ='safe')


def h5py_get_n_rows(file_path, key="telescope_events"):
    with h5py.File(file_path, "r") as f:
        group = f.get(key)

        if group is None:
            raise IOError('File does not contain group "{}"'.format(key))

        return group[next(iter(group.keys()))].shape[0]


infile = "./build/cta_analysis/gamma_train.h5"
outfile = "./build/len_train_gamma.txt"
with open(outfile, "w") as f:
    f.write(f'\\num{{{h5py_get_n_rows(infile, key="telescope_events")}}}')

infile = "./build/cta_analysis/gamma_test.h5"
outfile = "./build/len_test_gamma.txt"
with open(outfile, "w") as f:
    f.write(f'\\num{{{h5py_get_n_rows(infile, key="telescope_events")}}}')


infile = "./build/cta_analysis/proton_train.h5"
outfile = "./build/len_train_proton.txt"
with open(outfile, "w") as f:
    f.write(f'\\num{{{h5py_get_n_rows(infile, key="telescope_events")}}}')

infile = "./build/cta_analysis/proton_test.h5"
outfile = "./build/len_test_proton.txt"
with open(outfile, "w") as f:
    f.write(f'\\num{{{h5py_get_n_rows(infile, key="telescope_events")}}}')


with open('./configs/aict/iact_config.yaml', 'r') as f:
    d = yaml.load(f)
    k_cv = d['separator']['n_cross_validations']   
    s = d['separator']['classifier'] 
    n_estimators = re.findall('n_estimators=(\d+),', s)[0]  
    min_split = re.findall('min_samples_split=(\d+),', s)[0]  

    n_features = len(d['separator']['features']) + len(d['separator']['feature_generation']['features'])  


outfile = "./build/classifier_k_cv.txt"
with open(outfile, "w") as f:
    f.write(f'{k_cv}\\xspace')

outfile = "./build/classifier_n_estimators.txt"
with open(outfile, "w") as f:
    f.write(n_estimators)

outfile = "./build/classifier_min_split.txt"
with open(outfile, "w") as f:
    f.write(min_split)

outfile = "build/classifier_num_features.txt"
with open(outfile, "w") as f:
    f.write(f'{n_features}')


df = fact.io.read_data('./build/cta_analysis/aict_predictions_separation.h5', key='data')   

aucs = []
for n, g in df.groupby('cv_fold'):
    y_true = g['label']
    y_score = g['probabilities']
    aucs.append(roc_auc_score(y_true, y_score))

aucs = np.array(aucs)
print(aucs)

with open('./build/cv_auc.txt', 'w') as f:
    f.write(f'$\\num{{{aucs.mean():.2f}}} \pm {aucs.std():.3f}$')




################### regressor


with open('./configs/aict/iact_config.yaml', 'r') as f:
    d = yaml.load(f)
    k_cv = d['energy']['n_cross_validations']   
    s = d['energy']['regressor'] 
    n_estimators = re.findall('n_estimators=(\d+),', s)[0]  
    min_split = re.findall('min_samples_split=(\d+),', s)[0]  

    n_features = len(d['energy']['features']) + len(d['energy']['feature_generation']['features'])  


outfile = "./build/regressor_k_cv.txt"
with open(outfile, "w") as f:
    f.write(f'{k_cv}\\xspace')

outfile = "./build/regressor_n_estimators.txt"
with open(outfile, "w") as f:
    f.write(n_estimators)

outfile = "./build/regressor_min_split.txt"
with open(outfile, "w") as f:
    f.write(min_split)

outfile = "build/regressor_num_features.txt"
with open(outfile, "w") as f:
    f.write(f'{n_features}')


df = fact.io.read_data('./build/cta_analysis/aict_predictions_regression.h5', key='data')   

r2s = []
for n, g in df.groupby('cv_fold'):
    y_true = g['label']
    y_score = g['label_prediction']
    r2s.append(r2_score(y_true, y_score))

r2s = np.array(r2s)

with open('./build/cv_r2.txt', 'w') as f:
    f.write(f'$\\num{{{r2s.mean():.3f}}} \pm {r2s.std():.3f}$%')



