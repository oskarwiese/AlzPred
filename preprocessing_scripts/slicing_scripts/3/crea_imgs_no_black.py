from nipy import load_image
from skimage import io
from tqdm import tqdm
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

def subject_images(sub):
	print(sub)
	path = f'/dtu-compute/ADNIbias/freesurfer_ADNI2/{sub}/norm_mni305.mgz'
	img = load_image(path)
	for z in range(0,256):
            img_xy = np.array(img[:,:,z]).astype(np.uint8)
            if img_xy.sum() != 0:
                io.imsave(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/xy/train/B/B/{sub}_xy_z_{z}_3T.png', img_xy, check_contrast=False)
            img_xz = np.array(img[:,z,:]).astype(np.uint8)
            if img_xz.sum() != 0:
                io.imsave(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/xz/train/B/B/{sub}_xz_y_{z}_3T.png', img_xz, check_contrast=False)
            img_yz = np.array(img[z,:,:]).astype(np.uint8)
            if img_yz.sum() != 0:
                io.imsave(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/yz/train/B/B/{sub}_yz_x_{z}_3T.png', img_yz, check_contrast=False)

if __name__ == "__main__":

        li = []
        f = open("/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/preprocessing_scripts/slicing_scripts/3T_imgs/ids_ADNI2_3T.txt", "r")
        for x in f:
            li.append(x.strip())
        f.close()

        vali = []
        f = open("/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/preprocessing_scripts/slicing_scripts/3T_imgs/id_validation.txt")
        for x in f:
            vali.append(x.strip())
        f.close()


        a = set(li)
        b = set(vali)
        c = a - b
        print(f'\n Number of validation imagez in original FreeSurfer ADNI2 dataset is: {len(a)-len(c)}')


        num_cores = multiprocessing.cpu_count()
        subjects = tqdm(list(c), ascii = True)
        processed_list = Parallel(n_jobs=num_cores)(delayed(subject_images)(sub) for sub in subjects)
