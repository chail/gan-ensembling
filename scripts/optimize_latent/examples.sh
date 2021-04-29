# cat dataset: preprocess by masking the aligned cat face
python -m scripts.optimize_latent.optimize_cat --partition val --w_path results/optimize_latent/cat --indices 0 10

# car dataset: preprocess by resizing the image and centering the car using the provided bounding box
python -m scripts.optimize_latent.optimize_car --partition val --w_path results/optimize_latent/car --indices 0 10

# celebahq dataset: faces are aligned, no special preprocessing
python -m scripts.optimize_latent.optimize_celebahq --partition val --w_path results/optimize_latent/celebahq --indices 0 10

# cifar10 dataset: uses the mean w of the predicted class label to initialize optimization
python -m scripts.optimize_latent.optimize_cifar10 --partition val --w_path results/optimize_latent/cifar10 --indices 0 10
