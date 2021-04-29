##### celebahq #####
# image crops, test
python -m scripts.eval_ensemble.evaluate_image_ensemble \
	--domain celebahq --classifier_name Smiling --partition test
# stylemix fine layer, val partition
python -m scripts.eval_ensemble.evaluate_gan_ensemble \
	--domain celebahq --classifier_name Smiling \
	--partition val --apply_tensor_transform \
    --aug_type stylemix --aug_layer fine
# stylemix fine layer, test partition
python -m scripts.eval_ensemble.evaluate_gan_ensemble \
	--domain celebahq --classifier_name Smiling \
	--partition test --apply_tensor_transform \
    --aug_type stylemix --aug_layer fine

##### cat #####
# image crops, test
python -m scripts.eval_ensemble.evaluate_image_ensemble \
	--domain cat --classifier_name latentclassifier_stylemix_coarse \
	--partition test
# stylemix coarse layer, val partition
python -m scripts.eval_ensemble.evaluate_gan_ensemble \
	--domain cat --classifier_name latentclassifier_stylemix_coarse \
	--partition val --apply_tensor_transform \
    --aug_type stylemix --aug_layer coarse
# stylemix coarse layer, test partition
python -m scripts.eval_ensemble.evaluate_gan_ensemble \
	--domain cat --classifier_name latentclassifier_stylemix_coarse \
	--partition test --apply_tensor_transform \
    --aug_type stylemix --aug_layer coarse

##### car #####
# image crops, test
python -m scripts.eval_ensemble.evaluate_image_ensemble \
	--domain car --classifier_name latentclassifier_stylemix_fine \
	--partition test
# stylemix fine layer, val partition
python -m scripts.eval_ensemble.evaluate_gan_ensemble \
	--domain car --classifier_name latentclassifier_stylemix_fine \
	--partition val --apply_tensor_transform \
    --aug_type stylemix --aug_layer fine
# stylemix fine layer, test partition
python -m scripts.eval_ensemble.evaluate_gan_ensemble \
	--domain car --classifier_name latentclassifier_stylemix_fine \
	--partition test --apply_tensor_transform \
    --aug_type stylemix --aug_layer fine

##### cifar10 #####
# stylemix fine layer, val partition
python -m scripts.eval_ensemble.evaluate_gan_ensemble \
    --domain cifar10 --generator_type stylegan2-cc \
    --classifier_name imageclassifier  --partition val \
    --aug_type stylemix --aug_layer fine --batch_size 32
# stylemix fine layer, test partition
python -m scripts.eval_ensemble.evaluate_gan_ensemble \
    --domain cifar10 --generator_type stylegan2-cc \
    --classifier_name imageclassifier  --partition test \
    --aug_type stylemix --aug_layer fine --batch_size 32

##### celebahq-idinvert #####
# stylemix fine layer, val partition
python -m scripts.eval_ensemble.evaluate_gan_ensemble \
    --domain celebahq-idinvert --generator_type stylegan-idinvert \
    --classifier_name Smiling --partition val \
    --aug_type stylemix --aug_layer fine --apply_tensor_transform
# stylemix fine layer, test partition
python -m scripts.eval_ensemble.evaluate_gan_ensemble \
    --domain celebahq-idinvert --generator_type stylegan-idinvert \
    --classifier_name Smiling --partition test \
    --aug_type stylemix --aug_layer fine --apply_tensor_transform
