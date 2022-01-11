import pandas as pd 
t_15 = pd.read_csv('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/preprocessing_scripts/slicing_scripts/1.5T_imgs/ids_ADNI1_15T.txt', header= None)
t3 =   pd.read_csv('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/preprocessing_scripts/slicing_scripts/3T_imgs/ids_ADNI2_3T.txt', header= None)
tmp = pd.read_csv('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/preprocessing_scripts/slicing_scripts/3T_imgs/id_validation.txt', header = None)

li, li2, vali = t_15[0].unique(), t3[0].unique(), tmp[0].unique()

a = set(li)
b = set(vali)
c = a - b
print(f'\n Number of validation images in original FreeSurfer ADNI1 dataset is: {len(a)-len(c)}')

e = set(li2)
f = set(vali)
g = e -f
print(f'\n Number of validation images in original FreeSurfer ADNI2 dataset is: {len(e)-len(g)}')

subjects_to_run_freesurfer_on = list(a - c)
print(subjects_to_run_freesurfer_on)