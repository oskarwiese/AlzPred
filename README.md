# AlzPred
Official GitHub repository for Oskar &amp; Anders Bachelor project.

The scope of this project will be to train a classifier on MRI images to predict AD as well as training a cycle-generative adversarial network (cycleGAN)
to generate 1.5T images from 3T images in order to construct more usable, and
hopefully unbiased, data. Thereby, the model will have more available training
data, which should lead to a more generalizable classifier, which will be able to aid doctors and humans in need all around the world. This project will also aim to determine possible bias introduced in the model, and discuss how a classifier can be implemented as a tool for doctors, how this might benefit AD prevention, reduce treatment cost and the possible ethical scenarios that might be at play.


The project will revolve around the possibility to predict Alzheimers by utilizing cycleGAN generated data. Furthermore, the questions below will be researched in detail.


- Can a cycleGAN be trained to map between the following domains: 1.5T and 3T MRI
	images? 
  
- Does the data constructed by the cycleGAN then prove useful for classification?
  
- How effective is the prediction model in predicting whether a patient has
	alzheimer’s disease? And can the cyclGAN data remove previous demonstrated bias from the model?
	
Two models have been trained in this project, both are being trained on a server, which in turn means that the code will not be executeable directly by downloading this repository. In the folder /preprocessing_scrips are all data preprocessing scripts, these run freesurfer a software used for medical imaging. In the folder /cycleGAN is all code used in this project to train a cycleGAN. Several different cycleGANs have been trained on various datasets, most importantly on the MRI data from ADNI. In the folder /classifier is code for CNN classifier as well as code to analyze feature representations within the model.
