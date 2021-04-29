# will host the resources here until I set up a new endpoint
base_url=http://latent-composition.csail.mit.edu/other_projects/gan_ensembling/zips

# download network stats (<1M)
echo "downloading network stats ..."
wget $base_url/network_stats.zip
unzip network_stats.zip

# download precomputed evaluations (1.3G)
echo "downloading precomputed evaluations (1.3G) ..."
wget $base_url/precomputed_evaluations.zip
unzip precomputed_evaluations.zip

# download celebahq latents (1.8G)
echo "downloading celebahq latents (1.8G) ..."
wget $base_url/celebahq_latents.zip
unzip celebahq_latents.zip

# download car latents (121M)
echo "downloading car latents (121M) ..."
wget $base_url/car_latents.zip
unzip car_latents.zip

# download cat dataset (291M)
echo "downloading cat dataset  (291M) ..."
wget $base_url/cat_dataset.zip
unzip cat_dataset.zip

# download cifar10 latents(883M)
echo "downloading cifar10 latents (883M) ..."
wget $base_url/cifar10_latents.zip
unzip cifar10_latents.zip

# download pretrained classifiers (subset 592M, all classifiers 7.0G)
echo "downloading a subset of pretrained classifiers (592M) ..."
wget $base_url/pretrained_classifiers_subset.zip
unzip pretrained_classifiers_subset.zip 
# # uncomment the following lines to download all pretrained classifiers
# # it contains 40 binary attribute classifiers and classifiers trained 
# # on different perturb methods for the remaining datasets
# wget $base_url/pretrained_classifiers.zip
# unzip pretrained_classifiers.zip

### dependencies on other repositories:

# stylegan2 ada pytorch
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git resources/stylegan2-ada-pytorch

# stylegan id invert
# NOTE: the ffhq pretrained models need to be downloaded separately. see https://github.com/genforce/idinvert_pytorch for instructions
git clone https://github.com/genforce/idinvert_pytorch.git resources/idinvert_pytorch
# resolves path dependencies
ln -s resources/idinvert_pytorch/models ./

