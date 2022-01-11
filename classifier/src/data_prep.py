import pandas as pd 
import glob

generated = False

df_adni2 = pd.read_csv('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/csv_data/adni2.csv')
subs = glob.glob('/dtu-compute/ADNIbias/freesurfer_ADNI2/*')
subjects = []
for sub in subs:
    subjects.append(sub[-10:])

subjects.sort()
subjects = subjects[1:] # Remove fsavrage from subjects

out = pd.DataFrame()
for sub in subjects:
    try:
        tmp = df_adni2[df_adni2.Subject == sub].head(1)
        if (tmp['Group'].item() == 'AD') | (tmp['Group'].item() == 'CN'):
            tmp = tmp[[ 'Subject', 'Group', 'Sex', 'Age']]
            if generated:
                tmp['Path'] = glob.glob(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/1.5T_generated_from_3T_ADNI2/*{sub}*')[0]
                tmp['Type'] = 'Generated'
            else:
                tmp['Path'] = glob.glob(f'/dtu-compute/ADNIbias/freesurfer_ADNI2/*{sub}*/norm_mni305.mgz')[0]
                tmp['Type'] = '3'
                
            tmp['Label'] = 1 if tmp['Group'].item() == 'AD' else 0
            
            
            out = out.append(tmp)
    except:
        continue

df_adni1 =  pd.read_csv('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/csv_data/adni1.csv')
subs = glob.glob('/dtu-compute/ADNIbias/freesurfer_ADNI1/*')
subjects = []
for sub in subs:
    subjects.append(sub[-10:])
subjects.sort()
subjects = subjects[1:] # Remove fsavrage from subjects
subjects = subjects[:-10] # Remove random file names at end
for sub in subjects:
    try:
        tmp = df_adni1[df_adni1.Subject == sub].head(1)
        if (tmp['Group'].item() == 'AD') | (tmp['Group'].item() == 'CN'):
            tmp = tmp[[ 'Subject', 'Group', 'Sex', 'Age']]
            tmp['Path'] = glob.glob(f'/dtu-compute/ADNIbias/freesurfer_ADNI1/*{sub}*/norm_mni305.mgz')[0]
            tmp['Label'] = 1 if tmp['Group'].item() == 'AD' else 0
            if generated:
                tmp['Type'] = 'Original'
            else:
                tmp['Type'] = '1.5'
            out = out.append(tmp)
    except:
        continue

print(out.reset_index().drop(columns=['index']))
num_gen  = len(out[out.Type == 'Generated'])
num_orig = len(out[out.Type == 'Original'])
num_male = len(out[out.Sex == 'M'])
num_fem  = len(out[out.Sex == 'F'])
print(f'Number of Alz vs. Non-Alz in data:\n{out.Label.sum()} vs. {len(out)- out.Label.sum()}\n')
print(f'Number of Generated vs. Originial:\n{num_gen} vs. {num_orig}\n')
print(f'Number of Male vs. Female:\n{num_male} vs. {num_fem}\n')

#out.to_csv('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/csv_data/alz_data.csv')
out.to_csv('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/csv_data/alz_data_without_generated.csv')