# celebahq attribute classifiers
python -m scripts.train_classifier.attribute_classifier_image \
    --attribute Smiling 
python -m scripts.train_classifier.attribute_classifier_latent \
    --attribute Smiling 

# cat classifier
python -m scripts.train_classifier.multiclass_classifier_image \
	--domain cat
python -m scripts.train_classifier.multiclass_classifier_latent \
	--domain cat --seed 3
python -m scripts.train_classifier.multiclass_classifier_latent \
	--domain cat --perturb_type stylemix --perturb_layer coarse --seed 3

# car classifier
python -m scripts.train_classifier.multiclass_classifier_image \
	--domain car
python -m scripts.train_classifier.multiclass_classifier_latent \
	--domain car
python -m scripts.train_classifier.multiclass_classifier_latent \
	--domain car --perturb_type stylemix --perturb_layer fine

# cifar 10 classifier
python -m scripts.train_classifier.cifar10_classifier_image
python -m scripts.train_classifier.cifar10_classifier_latent
python -m scripts.train_classifier.cifar10_classifier_latent \
    --stylemix_layer fine

