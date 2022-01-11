from nipy import load_image
import numpy as np 
import imageio as io
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np
import cv2
from reconstruct import subject_images
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def _save_slice(img: np.ndarray, path: str, filename: str):
    img = Image.fromarray(img.astype("uint8"))
    img = img.convert("RGB")
    img.save(f'{path}{filename}')

def _middle_slices(nii: str):

    imgs_xy, imgs_xz, imgs_yz = [], [], []
    for z in tqdm(range(0,256)):
            img_xy = np.array(nii[:,:,z])
            if img_xy.sum() != 0:
                imgs_xy.append((img_xy,z))
            img_xz = np.array(nii[:,z,:])
            if img_xz.sum() != 0:
                imgs_xz.append((img_xz,z))
                
            img_yz = np.array(nii[z,:,:])
            if img_yz.sum() != 0:
                imgs_yz.append((img_yz, z))

    idx_black_xy = [min([val[1] for val in imgs_xy]), max([val[1] for val in  imgs_xy])]
    idx_black_xz = [min([val[1] for val in imgs_xz]), max([val[1] for val in  imgs_xz])]
    idx_black_yz = [min([val[1] for val in imgs_yz]), max([val[1] for val in  imgs_yz])]

    xy_slice = [val[0] for val in imgs_xy]
    xz_slice = [val[0] for val in imgs_xz]
    yz_slice = [val[0] for val in imgs_yz]
    xy_middle = xy_slice[len(xy_slice)//2]
    xz_middle = xz_slice[len(xz_slice)//2]
    yz_middle = yz_slice[len(yz_slice)//2]

    return xy_middle, xz_middle, yz_middle

def _image_stretching(ims: np.array(np.ndarray), v_min_d: int, v_max_d: int):
    '''
        Function to stretch image to have pixel vals between 0 and 255 
    '''
    v_min = ims.min()
    v_max = ims.max()
    for im in ims:
        if im.ndim == 3:
            shape = im.shape
            for i in tqdm(range(shape[0])):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        im[i,j,k] = ( (v_max_d - v_min_d) / (v_max - v_min) ) * (im[i,j,k] - v_min) + v_min_d
        
        else:        
            shape = im.shape
            for i in tqdm(range(shape[0])):
                for j in range(shape[1]):
                    im[i,j] = ( (v_max_d - v_min_d) / (v_max - v_min) ) * (im[i,j] - v_min) + v_min_d
    return ims

def lims(im: np.ndarray):
    im = im.copy()
    im[im<0] = 0
    im[im>255] = 255
    return im

def normalize(im: np.ndarray):
    return im / 255

def denormalize(im: np.ndarray):
    im = np.round(im * 255).astype(int)
    im = lims(im)
    return im.astype(int)

def _gamma_correction(ims: np.array(np.ndarray), gamma: float):
    for im in ims:
        im = normalize(im)
        im = denormalize(im ** gamma)
    return ims

    '''
        Function to gamma correct an image for lighter or darker pixels
    '''


def _geometric_accuracy(im, GT):
    '''
        Function to calculate the geometric accuracy. 
        How many pixels are exact in the generated compared to GT
        256 ** 3 amounts to perfect accuracy.
    '''

    out = 0
    shape = im.shape
    for i in tqdm(range(shape[0])):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if im[i,j,k] == GT[i,j,k]:
                    out +=1
    out_normalized = out / (256**3)
    return out, out_normalized

def _similarities(subjects: str, verbose: bool):
    '''
        Function to see difference between GT and cycleGAN generated image. gt1 is the same tesla as the generated image
    '''

    orig_rmses = []
    fake_rmses = []
    diff_rmses = []
    orig_geometric_vals = []
    orig_geometric_norms = []
    fake_geometric_vals = []
    fake_geometric_norms = []
    diff_geometric_vals = []
    diff_geometric_norms = []
    for sub in tqdm(subjects):
        # generated = f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/imgs_cycleGAN/xz_3_{sub}.nii'
        # generated = f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/yz_1.5_{sub}.nii'
        generated = f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/all_final_1.5_{sub}.nii'
        # gt_1_5_path = f'/dtu-compute/ADNIbias/freesurfer_ADNI1/{sub}/norm_mni305.mgz'
        # gt_3_path = f'/dtu-compute/ADNIbias/freesurfer/{sub}/norm_mni305.mgz'
        gt_3_path = f'/dtu-compute/ADNIbias/freesurfer_ADNI1/{sub}/norm_mni305.mgz'
        gt_1_5_path = f'/dtu-compute/ADNIbias/freesurfer/{sub}/norm_mni305.mgz'


        # print(f'Running on subject: {subject}')
        gen = load_image(generated)
        gen = np.array(gen[:,:,:])*255 # Convert from float to the pixel ---> loos of information?
        gen = gen.astype(np.uint8)
        
        gt1  = load_image(gt_1_5_path)
        gt1  = np.array(gt1[:,:,:]).astype(np.uint8)

        gt2  = load_image(gt_3_path)
        gt2  = np.array(gt2[:,:,:]).astype(np.uint8)
        
        # raw1 = abs(gen - gt1).sum()
        # raw1 = abs(gen - gt2).sum()

        # Histogram strech before subtracting GT from cycleGAN generated image
        # stretched_subtraction1 = abs(_image_stretching(gen) - _image_stretching(gt1)).sum()
        # stretched_subtraction2 = abs(_image_stretching(gen) - _image_stretching(gt2)).sum()

        orig_geometric_val, orig_geometric_norm = _geometric_accuracy(gt1, gt2)
        fake_geometric_val, fake_geometric_norm = _geometric_accuracy(gen, gt1)
        diff_geometric_val = abs(orig_geometric_val - fake_geometric_val)
        diff_geometric_norm = abs(orig_geometric_norm - fake_geometric_norm)

        gen_2d = gen.reshape(256, 256*256)
        gt1_2d = gt1.reshape(256, 256*256)
        gt2_2d = gt2.reshape(256, 256*256)
        orig_rmse = mean_squared_error(gt1_2d, gt2_2d, squared=False)
        fake_rmse = mean_squared_error(gen_2d, gt2_2d, squared=False)
        diff_rmse = abs(orig_rmse - fake_rmse)
        orig_rmses.append(orig_rmse)
        fake_rmses.append(fake_rmse)
        diff_rmses.append(diff_rmse)
        orig_geometric_vals.append(orig_geometric_val)
        orig_geometric_norms.append(orig_geometric_norm)
        fake_geometric_vals.append(fake_geometric_val)
        fake_geometric_norms.append(fake_geometric_norm)
        diff_geometric_vals.append(diff_geometric_val)
        diff_geometric_norms.append(diff_geometric_norm)

        if verbose:
            print(f'Origianl RMSE:\n{orig_rmse}')
            print(f'Generated RMSE:\n{fake_rmse}')
            print(f'Difference:\n{diff_rmse}\n')
            print(f'Geo_accuracy to 1.5T:\n{geometrics}')
            # print(f'Geo_accuracy to 3T:\n{geom_accuracy2}') 
            # print(f'Raw subtraction:\n{raw_subtraction}\n')
            # print(f'Stretched subtraction:\n{stretched_subtraction}\n')

    print(np.mean(orig_rmses))  # 1.5344034993958493
    print(np.mean(fake_rmses))  # 1.0253746188316661
    print(np.mean(diff_rmses))  # 0.5090288805641832
    print(np.mean(orig_geometric_vals))
    print(np.mean(orig_geometric_norms))
    print(np.mean(fake_geometric_vals))
    print(np.mean(fake_geometric_norms))
    print(np.mean(diff_geometric_vals))
    print(np.mean(diff_geometric_norms))

    return

def _diff_img(subject: str, savepath: str):
    generated_1_5 = f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/xy_1.5_{subject}.nii'
    # generated_3 = f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/xy_1.5_{subject}.nii'
    gt_1_5_path = f'/dtu-compute/ADNIbias/freesurfer_ADNI1/{subject}/norm_mni305.mgz'
    gt_3_path = f'/dtu-compute/ADNIbias/freesurfer/{subject}/norm_mni305.mgz'


    gen_1_5 = load_image(generated_1_5)
    gen_1_5 = np.array(gen_1_5[:,:,:])*255 # Convert from float to the pixel ---> loos of information?
    gen_1_5 = gen_1_5.astype(np.uint8)
    
    gt1  = load_image(gt_1_5_path)
    gt1  = np.array(gt1[:,:,:]).astype(np.uint8)

    gt2  = load_image(gt_3_path)
    gt2  = np.array(gt2[:,:,:]).astype(np.uint8)

    xy_gen, xz_gen, yz_gen = _middle_slices(gen_1_5)
    xy_gt1, xz_gt1, yz_gt1 = _middle_slices(gt1)
    xy_gt2, xz_gt2, yz_gt2 = _middle_slices(gt2)
    _save_slice(xy_gen, savepath, f"ALL_model_{subject}_xy_3.jpg")
    _save_slice(xz_gen, savepath, f"ALL_model_{subject}_xz_3.jpg")
    _save_slice(yz_gen, savepath, f"ALL_model_{subject}_yz_3.jpg")

    imgs = np.array([cv2.subtract(xy_gt2,xy_gt1), cv2.subtract(xy_gen,xy_gt1), cv2.subtract(xz_gt2,xz_gt1), cv2.subtract(xz_gen,xz_gt1), cv2.subtract(yz_gt2,yz_gt1), cv2.subtract(yz_gen,yz_gt1)])
    imgs = _image_stretching(imgs, v_min_d = 0, v_max_d = 255)
    # imgs = _gamma_correction(imgs, gamma = 2)
    _save_slice(imgs[0], savepath, f"{subject}_xy_gts_comparison.jpg")
    _save_slice(imgs[1], savepath, f"{subject}_xy_1.5_1.5gen_comparison.jpg")
    _save_slice(imgs[2], savepath, f"{subject}_xz_gts_comparison.jpg")
    _save_slice(imgs[3], savepath, f"{subject}_xz_1.5_1.5gen_comparison.jpg")
    _save_slice(imgs[4], savepath, f"{subject}_yz_gts_comparison.jpg")
    _save_slice(imgs[5], savepath, f"{subject}_yz_1.5_1.5gen_comparison.jpg")

    # _save_slice(cv2.subtract(xy_gen,xy_gt1), savepath, f"{subject}_xy_1.5_1.5gen_comparison.jpg")
    # _save_slice(cv2.subtract(xy_gen,xy_gt2), savepath, f"{subject}_xy_3_1.5gen_comparison.jpg")
    # _save_slice(cv2.subtract(xz_gen,xz_gt1), savepath, f"{subject}_xz_1.5_1.5gen_comparison.jpg")
    # _save_slice(cv2.subtract(xz_gen,xz_gt2), savepath, f"{subject}_xz_3_1.5gen_comparison.jpg")
    # _save_slice(cv2.subtract(yz_gen,yz_gt1), savepath, f"{subject}_yz_1.5_1.5gen_comparison.jpg")
    # _save_slice(cv2.subtract(yz_gen,yz_gt2), savepath, f"{subject}_yz_3_1.5gen_comparison.jpg")

if __name__ == '__main__':
    # subjects = ["136_S_0196", "082_S_0469", "002_S_0559"]
    # for subject in subjects:
    #      _diff_img(subject = subject, savepath = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/136_S_0196/")
    
    val_subs = ["002_S_0413", "023_S_0030", "023_S_0331", "023_S_0855", "023_S_1190", "067_S_0607", "082_S_0640", 
                "136_S_0299", "002_S_0559", "023_S_0031", "023_S_0376", "023_S_0916", "023_S_1247", "068_S_0442", 
                "136_S_0086", "136_S_0300", "002_S_0729", "023_S_0058", "023_S_0388", "023_S_0963", "023_S_1262", 
                "068_S_0476", "136_S_0184", "136_S_0426", "002_S_0816", "023_S_0061", "023_S_0604", "023_S_1046", 
                "023_S_1289", "068_S_0478", "136_S_0194", "136_S_0429", "002_S_0954", "023_S_0078", "023_S_0613", 
                "023_S_1104", "053_S_0507", "082_S_0304", "136_S_0195", "136_S_0579", "018_S_0369", "023_S_0139", 
                "023_S_0625", "023_S_1126", "067_S_0290", "082_S_0469", "136_S_0196"]
    invalid_subs = ['068_S_0442', '068_S_0476', '068_S_0478']
    subjects = [sub for sub in val_subs if sub not in invalid_subs]

    _similarities(subjects, verbose = False)