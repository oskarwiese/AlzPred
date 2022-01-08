from nipy import load_image
from skimage import io
from tqdm import tqdm
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

def subject_images(sub):
	path = f'/dtu-compute/ADNIbias/freesurfer_ADNI2/{sub}/norm_mni305.mgz'
	img = load_image(path)
	for z in range(0,256):
		img_xy = np.array(img[:,:,z]).astype(np.uint8)
		io.imsave(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/3T_imgs_real/xy/{sub}_xy_z_{z}_3T.png', img_xy)
		img_xz = np.array(img[:,z,:]).astype(np.uint8)
		io.imsave(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/3T_imgs_real/xz/{sub}_xz_y_{z}_3T.png', img_xz)
		img_yz = np.array(img[z,:,:]).astype(np.uint8)
		io.imsave(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/3T_imgs_real/yz/{sub}_yz_x_{z}_3T.png', img_yz)

if __name__ == "__main__":

	li = []
	f = open("/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/3T_imgs_real/ids_ADNI2_3T.txt", "r")
	for x in f:
  		li.append(x.strip())
	f.close()

	vali = []
	f = open("/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/3T_imgs_real/id_validation.txt")
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