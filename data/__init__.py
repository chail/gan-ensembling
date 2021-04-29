
def get_transform(dataset_name, transform_type):
    if dataset_name == 'celebahq':
        from . import data_celebahq
        return data_celebahq.get_transform(dataset_name, transform_type)
    elif dataset_name == 'celebahq-idinvert':
        from . import data_celebahq
        return data_celebahq.get_transform(dataset_name, transform_type)
    elif dataset_name == 'car':
        from . import data_cars
        return data_cars.get_transform(dataset_name, transform_type)
    elif dataset_name == 'cat':
        from . import data_cats
        return data_cats.get_transform(dataset_name, transform_type)
    elif dataset_name == 'cifar10':
        from . import data_cifar10
        return data_cifar10.get_transform(dataset_name, transform_type)
    else:
        raise NotImplementedError

def get_dataset(dataset_name, partition, *args, **kwargs):
    if dataset_name == 'celebahq':
        from . import data_celebahq
        return data_celebahq.CelebAHQDataset(partition, *args, **kwargs)
    elif dataset_name == 'celebahq-idinvert':
        from . import data_celebahq
        return data_celebahq.CelebAHQIDInvertDataset(partition, *args, **kwargs)
    elif dataset_name == 'car':
        from . import data_cars
        return data_cars.CarsDataset(partition, *args, **kwargs)
    elif dataset_name == 'cat':
        from . import data_cats
        return data_cats.CatFaceDataset(partition, *args, **kwargs)
    elif dataset_name == 'cifar10':
        from . import data_cifar10
        return data_cifar10.CIFAR10Dataset(partition, *args, **kwargs)
    else:
        raise NotImplementedError
