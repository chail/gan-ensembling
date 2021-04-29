import os
import sys
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import folder
import torch.utils.data as data
from utils import util

class ImageDataset(VisionDataset): # data.Dataset): # VisionDataset):
    """
    modified from: https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    uses cached directory listing if available rather than walking directory
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader=folder.default_loader,
                 extensions=folder.IMG_EXTENSIONS, transform=None,
                 target_transform=None, is_valid_file=None, return_path=False):
        super(ImageDataset, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        # self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        cache = self.root.rstrip('/') + '.txt'
        if os.path.isfile(cache):
            print("Using directory list at: %s" % cache)
            with open(cache) as f:
                samples = []
                for line in f:
                    (path, idx) = line.strip().split(';')
                    samples.append((os.path.join(self.root, path), int(idx)))
        else:
            print("Walking directory: %s" % self.root)
            samples = folder.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
            with open(cache, 'w') as f:
                for line in samples:
                    path, label = line
                    f.write('%s;%d\n' % (util.remove_prefix(path, self.root).lstrip('/'), label))

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = samples
        self.return_path = return_path
        # self.transform = transform
        # self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_path:
            return sample, target, path
        return sample, target

    def __len__(self):
        return len(self.samples)
