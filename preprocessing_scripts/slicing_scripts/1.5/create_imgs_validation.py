from nipy import load_image
from skimage import io
from tqdm import tqdm
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

def subject_images(sub):
    print(sub)
    path = f'/dtu-compute/ADNIbias/freesurfer_ADNI1/{sub}/norm_mni305.mgz'
    try:
        img = load_image(path)
        for z in range(0,256):
                img_xy = np.array(img[:,:,z]).astype(np.uint8)
                if img_xy.sum() != 0:
                    io.imsave(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/xy/test/A/A/{sub}_xy_z_{z}_15T.png', img_xy, check_contrast=False)
                img_xz = np.array(img[:,z,:]).astype(np.uint8)
                if img_xz.sum() != 0:
                    io.imsave(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/xz/test/A/A/{sub}_xz_y_{z}_15T.png', img_xz, check_contrast=False)
                img_yz = np.array(img[z,:,:]).astype(np.uint8)
                if img_yz.sum() != 0:
                    io.imsave(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/yz/test/A/A/{sub}_yz_x_{z}_15T.png', img_yz, check_contrast=False)
    except:
        pass 


if __name__ == "__main__":

        vali = []
        f = open("/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/preprocessing_scripts/slicing_scripts/1.5T_imgs/id_validation.txt")
        for x in f:
            vali.append(x.strip())
        f.close()


        #a = set(li)
        b = set(vali)
        print(f'\n Number of validation imagez in original FreeSurfer ADNI1 dataset is: {len(b)}')


        num_cores = multiprocessing.cpu_count()
        subjects = tqdm(list(b), ascii = True)
        processed_list = Parallel(n_jobs=num_cores)(delayed(subject_images)(sub) for sub in subjects)