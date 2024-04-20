#%%

## Datasets for nodepint based on HuggingFace's datasets library

import jax
import numpy as np
from typing import Collection
# from datasets import Dataset, load_dataset

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def load_mnist_dataset_torch(root="./data", train=True) -> Dataset:

    mean, std = (0.1307,), (0.3081,)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean, std)  # Normalize using typical MNIST values
    ])

    return datasets.MNIST(root=root, train=train, transform=transform, download=True)





class ODEDataset(Dataset):
    """ODE dataset."""

    def __init__(self, root_dir, split="train"):
        """
        Arguments:
            root_dir (string): Directory with all the trajectories.
            split (string): train, test (for in-domain), or ood_train, ood_test (for out-of-domain)
        """

        self.root_dir = root_dir if root_dir[-1] == "/" else root_dir+"/"
        self.split = split

        if self.split == "train":
            filename = self.root_dir+"train.npz"
        elif self.split == "test":
            filename = self.root_dir+"test.npz"
        elif self.split == "ood_train":
            filename = self.root_dir+"ood_train.npz"
        elif self.split == "ood_test":
            filename = self.root_dir+"ood_test.npz"

        data = np.load(filename)
        self.X, self.t = data["X"][5], data["t"]        ## TODO: use any environment e here

    def __len__(self):
        return self.X.shape[0]  ## Number of trajectories

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # sample = {'init_state': self.X[idx, 0, ...], 'trajectory': self.X[idx, :, ...]}
        sample = (self.X[idx, 0, ...], self.X[idx, :, ...])

        return sample




def load_lotka_volterra_dataset(root_dir, split) -> Dataset:
    return ODEDataset(root_dir, split)











def make_dataloader_torch(ds: Dataset, batch_size: int = 32, num_workers: int=24, shuffle: bool = True) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)











# def load_jax_dataset_hf(**kwargs) -> Dataset:
#     """
#     The load_jax_dataset function loads a dataset from the HuggingFace Hub or from the file system, converts it to JAX format, and returns it.
    
#     :param **kwargs: Pass a variable number of keyword arguments to the function. The arguments are the same as for HuggingFace's load_dataset().
#     :return: A dataset in the jax format
#     """
#     ds = load_dataset(**kwargs)
#     ds = ds.with_format("jax")
#     return ds


# def get_dataset_features_hf(ds: Dataset) -> Collection[str]:
#     """
#     The get_dataset_features function returns a collection of the features in a dataset.
    
#     :param ds: Dataset: Specify the type of the parameter
#     :return: A collection of strings
#     """
#     # return tuple(ds.features.keys())
#     return ds.column_names


# def reorder_dataset_features_hf(ds: Dataset, features: Collection[str]) -> Dataset:
#     return ds.select_columns(features)

# def extract_all_data_hf(ds:Dataset, features: Collection[str] = None) -> Collection[jax.numpy.ndarray]:
#     """
#     This takes a dataset and returns all of the data in that dataset.
    
#     :param ds:Dataset: Specify the dataset that is being used
#     :param features: Collection[str]: Specify which features to extract
#     :return: A tuple of Jax arrays
#     """
#     if features is None:
#         features = get_dataset_features_hf(ds)
#     return tuple([ds[:][feature] for feature in features])

# # def project_dataset_onto_basis(ds:Dataset, basis:jax.numpy.ndarray) -> Dataset:
# #     """
# #     This takes a dataset and projects it onto a basis. TODO do this piece by pieve, and not all at once.
    
# #     :param ds:Dataset: Specify the dataset that is being used
# #     :param basis:jax.numpy.ndarray: Specify the basis that is being used
# #     :return: A dataset
# #     """

# #     all_data, labels = extract_all_data(ds, features=["image", "label"])
# #     flat_data = jnp.reshape(all_data, (all_data.shape[0], -1))

# #     return flat_data @ basis, labels


# def convert_to_one_hot_encoding_hf(ds:Dataset, feature:str="label") -> Dataset:
#     """
#     This takes a dataset and converts a feature to a one-hot encoding.
    
#     :param ds:Dataset: Specify the dataset that is being used
#     :param feature:str: Specify the feature that is being converted
#     :return: A dataset
#     """
#     num_classes = ds.features[feature].num_classes

#     def one_hot_encoding(batch):
#         newval = jax.nn.one_hot(batch[feature], num_classes)
#         return {"onehotlabels": newval}

#     ds = ds.map(one_hot_encoding, remove_columns=[feature], batched=True)

#     ds = ds.rename_column("onehotlabels", feature)

#     return ds

# def normalise_feature_hf(ds:Dataset, feature:str="image", factor:float=255.) -> Dataset:
#     def divide_by_255(batch):
#         newval = batch[feature] / factor
#         return {"normalised": newval}

#     ds = ds.map(divide_by_255, remove_columns=[feature], batched=True)

#     ds = ds.rename_column("normalised", feature)

#     return ds


# def preprocess_mnist_hf(ds, subset_size="all", seed=None, norm_factor=255., feature_order=["image", "label"]):

#     ds = convert_to_one_hot_encoding_hf(ds, feature="label")
#     ds = normalise_feature_hf(ds, feature="image", factor=norm_factor)

#     if subset_size != "all":
#         if seed is None:
#             seed = time.time_ns()
#             print("WARNING: no seed provided. Using time.time_ns()")
#         if subset_size > ds.num_rows:
#             subset_size = ds.num_rows
#             print("WARNING: subset_size bigger than dataset. Using all datapoints")
#         np.random.seed(seed)
#         ds = ds.select(np.random.randint(0, ds.num_rows, subset_size))

#     ## Warning. Always make sure your datapoints are first, and labels second
#     ds = reorder_dataset_features_hf(ds, feature_order)
#     return ds




# # def preprocess_mnist_conv(ds, subset_size="all", seed=None, norm_factor=255., feature_order=["image", "label"]):
# #     """ Preprocess the MNIST dataset for convolutions """

# #     ds = convert_to_one_hot_encoding(ds, feature="label")
# #     ds = normalise_feature(ds, feature="image", factor=norm_factor) ## TODO use standard normalisation for MNIST

# #     if subset_size != "all":
# #         if seed is None:
# #             seed = time.time_ns()
# #             print("WARNING: no seed provided. Using time.time_ns()")
# #         if subset_size > ds.num_rows:
# #             subset_size = ds.num_rows
# #             print("WARNING: subset_size bigger than dataset. Using all datapoints")
# #         np.random.seed(seed)
# #         ds = ds.select(np.random.randint(0, ds.num_rows, subset_size))

# #     ## Add a channel dimension
# #     ds = ds.map(lambda x: {"image": np.reshape(x["image"], (1, 28, 28))})

# #     ## Warning. Always make sure your datapoints are first, and labels second
# #     ds = reorder_dataset_features(ds, feature_order)
# #     return ds




if __name__ == "__main__":

    import time

    ds = load_jax_dataset_hf(path="mnist", split="train")
    print(f"Features are: {get_dataset_features_hf(ds)}")

    ds = convert_to_one_hot_encoding_hf(ds, feature="label")
    print(ds["image"].shape)

    start = time.time()
    ds = extract_all_data_hf(ds, features=["image", "label"])
    print(type(ds[0]))
    print(f"Time to extract: {time.time() - start}")

# %%
