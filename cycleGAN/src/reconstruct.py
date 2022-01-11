
from re import S
import torch
import os
import argparse
import glob
from torchvision.datasets import folder
import config
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import nibabel as nib
from skimage import io
from nipy import load_image
from torchvision import transforms
import torchvision.datasets as datasets 
import torch.optim as optim
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_MRI import Generator
from torch.utils.data import DataLoader
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def subject_images(sub, tesla = 3):
    print(f'Working on subject: {sub}\n')
    
    if tesla == 3:
        #path = f'/dtu-compute/ADNIbias/freesurfer_ADNI2/{sub}/norm_mni305.mgz'
        path = f'/dtu-compute/ADNIbias/freesurfer/{sub}/norm_mni305.mgz' # Running on our validation data
    else:
        path = f'/dtu-compute/ADNIbias/freesurfer_ADNI1/{sub}/norm_mni305.mgz'
       
    img = load_image(path)
    imgs_xy, imgs_xz, imgs_yz = [], [], []
    for z in tqdm(range(0,256)):
            img_xy = np.array(img[:,:,z]).astype(np.uint8)
            if img_xy.sum() != 0:
                imgs_xy.append((img_xy,z))
                io.imsave(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/xy/{tesla}/{tesla}/{str(z).zfill(3)}_{sub}_xy_z_{z}_{tesla}T.png', img_xy, check_contrast=False)
            img_xz = np.array(img[:,z,:]).astype(np.uint8)
            if img_xz.sum() != 0:
                imgs_xz.append((img_xz,z))
                io.imsave(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/xz/{tesla}/{tesla}/{str(z).zfill(3)}_{sub}_xz_y_{z}_{tesla}T.png', img_xz, check_contrast=False)
                
            img_yz = np.array(img[z,:,:]).astype(np.uint8)
            if img_yz.sum() != 0:
                imgs_yz.append((img_yz, z))
                io.imsave(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/yz/{tesla}/{tesla}/{str(z).zfill(3)}_{sub}_yz_x_{z}_{tesla}T.png', img_yz, check_contrast=False)

    return imgs_xy, imgs_xz, imgs_yz

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=dev)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def test_model(img):
    img_channel = 1
    img_size = 256
    x = torch.randn((1, img_channel, img_size, img_size))
    gen = Generator(img_channel, 9)
    print(gen(img))
    print(gen(img).shape)

def imshow(image, name, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return plt.savefig(name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() ##
    parser.add_argument("-p", '--plane', type=str,
                    help="Plane to reconstruct: xy, xz or yz")
    parser.add_argument("-s", '--subject', type=str,
                    help="Example of subject: 002_S_4171")
    parser.add_argument("-t", '--tesla', type=int,
                    help="Either 1.5 or 3 tesla, note that if you choose 3T a 1.5T version will be generated")
    parser.add_argument("-d", '--directory', type=str)
    parser.add_argument("-a", '--all', type=str)
    args = parser.parse_args()

    assert args.plane in ['xy', 'xz', 'yz']
    if not args.subject:
        print('No subject chosen, so subject default is: 002_S_4171 \n')
        args.subject = '002_S_4171'
    if not args.tesla:
        print('No field strength chosen, default is: 3 \n')
        args.tesla = 3
    if args.tesla != 3:
        args.tesla = 1.5
    if args.all == "True" or args.all == "true" or args.all == "tue" or args.all == "t" or args.all == "T":
        args.all = True
    else:
        args.all = False
    
    
    print(f'Chosen keyword arguments are:\n {args.plane} \n {args.subject} \n {args.tesla} \n {args.all} \n')

    print(f'Clearing folders...\n')
    folder_paths = [
            f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/xy/{args.tesla}/{args.tesla}/',
            f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/xz/{args.tesla}/{args.tesla}/',
            f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/yz/{args.tesla}/{args.tesla}/']

    for folder_path in tqdm(folder_paths):
        import shutil
        shutil.rmtree(folder_path)
        os.mkdir(folder_path )

    models_paths = glob.glob(args.directory + '/*tar')
    models_paths.sort()
    print(f'The directory path chosen is:\n {models_paths} \n')

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G_a = Generator(img_channels=1, num_residuals=9).to(dev)
    G_b = Generator(img_channels=1, num_residuals=9).to(dev)
        
    opti_G = optim.Adam(
        list(G_a.parameters()) + list(G_b.parameters()),
        lr = config.GEN_LEARNING_RATE,
        betas = (0.5, 0.999)
    )

    # Only need generators here
    load_checkpoint(f'{models_paths[2]}', G_a, opti_G, config.GEN_LEARNING_RATE)
    load_checkpoint(f'{models_paths[3]}', G_b, opti_G, config.GEN_LEARNING_RATE)
     
    xy, xz, yz = subject_images(args.subject, tesla = args.tesla) # 002_S_4171 is first subject
    dic = {
        'xy' : xy,
        'xz' : xz,
        'yz' : yz
        }

    idx_black = [min([val[1] for val in dic[args.plane]]), max([val[1] for val in  dic[args.plane]])] # find min and max
    idx_black_xy = [min([val[1] for val in dic["xy"]]), max([val[1] for val in  dic["xy"]])]
    idx_black_xz = [min([val[1] for val in dic["xz"]]), max([val[1] for val in  dic["xz"]])]
    idx_black_yz = [min([val[1] for val in dic["yz"]]), max([val[1] for val in  dic["yz"]])]
    print(f'Image starts and end at index: {idx_black}\n')
                                            
    test_data = datasets.ImageFolder(root=f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/{args.plane}/{args.tesla}/', # In this case we want to construct 1.5T from 3T so from B to A
                            transform=transforms.Compose([
                                transforms.Resize((256,256)),
                                transforms.CenterCrop((256,256)),
                                transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels = 1),
                                transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize(0.5,0.5),
                                ]))
    
    test_data = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    if args.tesla != 3:
        G_a = G_b # Ensure correct model for reconstruction

    # Save all images for each subject into list for reconstructing the 3D image
    # Can we do it in a more efficient way than dataloader with torch
    i = idx_black[0]
    print(f'Starting reconstruction...\n')
    reconstructed_list = []
    for b,_ in tqdm(test_data):
        b = b.to(device)
        #save_image(b*0.5 + 0.5,'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/xy/reconstructed/orig_plot.png')
        fake_a = G_a(b) # Fake 1.5T image
        fake_a = fake_a*0.5 + 0.5
        numpy_arr = fake_a.cpu()
        numpy_arr = numpy_arr.detach().numpy()
        numpy_arr = numpy_arr[0,0,:,:]
        reconstructed_list.append(numpy_arr)
        #save_image(fake_a,f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/xy/reconstructed/model_img_idx_{i}.png')
        i += 1
    print(f'Final img should be of index: {i}\n')

    reconstructed_imgs = np.zeros([256,256,256])
    idx = 0
    
    for i in tqdm(range(256)):
        if i < idx_black[0]:
            reconstructed_imgs[i] = np.zeros([256,256])
        if i > idx_black[1]:
            reconstructed_imgs[i] = np.zeros([256,256])
        if i >= idx_black[0] and i <= idx_black[1]:
            if args.plane == 'xy':
                reconstructed_imgs[:,:,i] = reconstructed_list[idx]
                idx += 1
            elif args.plane == 'xz':
                reconstructed_imgs[:,i,:] = reconstructed_list[idx]
                idx += 1        
            elif args.plane == 'yz':
                reconstructed_imgs[i,:,:] = reconstructed_list[idx]
                idx += 1
                

    im_xy = Image.fromarray((reconstructed_imgs[:,:,sum(idx_black_xy)//2] * 255).astype("uint8"))
    im_xy = im_xy.convert("RGB")
    im_xz = Image.fromarray((reconstructed_imgs[:,sum(idx_black_xz)//2,:] * 255).astype("uint8"))
    im_xz = im_xz.convert("RGB")
    im_yz = Image.fromarray((reconstructed_imgs[sum(idx_black_yz)//2,:,:] * 255).astype("uint8"))
    im_yz = im_yz.convert("RGB")
    
    # Important to note that all below images are created from same cycleGAN, but correspond to the three types of planes that have been used to train the cycleGAN models.
    
    

    

    transformation = np.array([[  -1.,    0.,    0.,  128.],
                               [   0.,    0.,    1., -128.],
                               [   0.,   -1.,    0.,  128.],
                               [   0.,    0.,    0.,    1.]])   # Affine transformation matrix
    # Histogram stretch here
    


    img3d = nib.Nifti1Image(reconstructed_imgs, affine=transformation)
    if not args.all:    
        nib.save(img3d, f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/{args.plane}_{args.tesla}_{args.subject}.nii') #TODO create better naming convention
        im_xy.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/{args.plane.upper()}_model_{args.subject}_xy_{args.tesla}.jpg') 
        im_xz.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/{args.plane.upper()}_model_{args.subject}_xz_{args.tesla}.jpg')
        im_yz.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/{args.plane.upper()}_model_{args.subject}_yz_{args.tesla}.jpg')
        # im_xy.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/xy_{args.tesla}_{args.subject}.jpg') 
        # im_xz.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/xz_{args.tesla}_{args.subject}.jpg')
        # im_yz.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/yz_{args.tesla}_{args.subject}.jpg')
    else:
        im_xy.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/{args.subject}/ALL_model_xy_{args.subject}_{args.tesla}.jpg') 
        im_xz.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/{args.subject}/ALL_model_xz_{args.subject}_{args.tesla}.jpg')
        im_yz.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/{args.subject}/ALL_model_yz_{args.subject}_{args.tesla}.jpg')
        nib.save(img3d, f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/all_{args.plane}_{args.tesla}_{args.subject}.nii') #TODO create better naming convention

    #nib.save(img3d, f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/1.5T_generated_from_3T_ADNI2/{args.plane}_{args.tesla}_{args.subject}.nii') #TODO create better naming convention
    print('Saving done')
# A
##################################################################################################################################################################
# Run using:                                                                                                                                                     #
# reconstruct.py -p xy -s 002_S_4171 -t 3 -d /dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation/models/xy_nearest_lowlearning_final #
# This reconstructs xy from subject 002_S_4171 with field strength 3T                                                                                            #
##################################################################################################################################################################1