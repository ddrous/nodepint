#%%

## Datasets for nodepint based on HuggingFace's datasets library

import jax
import numpy as np
from typing import Collection
from datasets import Dataset, load_dataset


def load_jax_dataset(**kwargs) -> Dataset:
    """
    The load_jax_dataset function loads a dataset from the HuggingFace Hub or from the file system, converts it to JAX format, and returns it.
    
    :param **kwargs: Pass a variable number of keyword arguments to the function. The arguments are the same as for HuggingFace's load_dataset().
    :return: A dataset in the jax format
    """
    ds = load_dataset(**kwargs)
    ds = ds.with_format("jax")
    return ds


def get_dataset_features(ds: Dataset) -> Collection[str]:
    """
    The get_dataset_features function returns a collection of the features in a dataset.
    
    :param ds: Dataset: Specify the type of the parameter
    :return: A collection of strings
    """
    # return tuple(ds.features.keys())
    return ds.column_names


def reorder_dataset_features(ds: Dataset, features: Collection[str]) -> Dataset:
    return ds.select_columns(features)

def extract_all_data(ds:Dataset, features: Collection[str] = None) -> Collection[jax.numpy.ndarray]:
    """
    This takes a dataset and returns all of the data in that dataset.
    
    :param ds:Dataset: Specify the dataset that is being used
    :param features: Collection[str]: Specify which features to extract
    :return: A tuple of Jax arrays
    """
    if features is None:
        features = get_dataset_features(ds)
    return tuple([ds[:][feature] for feature in features])

# def project_dataset_onto_basis(ds:Dataset, basis:jax.numpy.ndarray) -> Dataset:
#     """
#     This takes a dataset and projects it onto a basis. TODO do this piece by pieve, and not all at once.
    
#     :param ds:Dataset: Specify the dataset that is being used
#     :param basis:jax.numpy.ndarray: Specify the basis that is being used
#     :return: A dataset
#     """

#     all_data, labels = extract_all_data(ds, features=["image", "label"])
#     flat_data = jnp.reshape(all_data, (all_data.shape[0], -1))

#     return flat_data @ basis, labels


def convert_to_one_hot_encoding(ds:Dataset, feature:str="label") -> Dataset:
    """
    This takes a dataset and converts a feature to a one-hot encoding.
    
    :param ds:Dataset: Specify the dataset that is being used
    :param feature:str: Specify the feature that is being converted
    :return: A dataset
    """
    num_classes = ds.features[feature].num_classes

    def one_hot_encoding(batch):
        newval = jax.nn.one_hot(batch[feature], num_classes)
        return {"onehotlabels": newval}

    ds = ds.map(one_hot_encoding, remove_columns=[feature], batched=True)

    ds = ds.rename_column("onehotlabels", feature)

    return ds

def normalise_feature(ds:Dataset, feature:str="image", factor:float=255.) -> Dataset:
    def divide_by_255(batch):
        newval = batch[feature] / factor
        return {"normalised": newval}

    ds = ds.map(divide_by_255, remove_columns=[feature], batched=True)

    ds = ds.rename_column("normalised", feature)

    return ds


def preprocess_mnist(ds, subset_size="all", seed=None, norm_factor=255., feature_order=["image", "label"]):

    ds = convert_to_one_hot_encoding(ds, feature="label")
    ds = normalise_feature(ds, feature="image", factor=norm_factor)

    if subset_size != "all":
        if seed is None:
            seed = time.time_ns()
            print("WARNING: no seed provided. Using time.time_ns()")
        if subset_size > ds.num_rows:
            subset_size = ds.num_rows
            print("WARNING: subset_size bigger than dataset. Using all datapoints")
        np.random.seed(seed)
        ds = ds.select(np.random.randint(0, ds.num_rows, subset_size))

    ## Warning. Always make sure your datapoints are first, and labels second
    ds = reorder_dataset_features(ds, feature_order)
    return ds




# def preprocess_mnist_conv(ds, subset_size="all", seed=None, norm_factor=255., feature_order=["image", "label"]):
#     """ Preprocess the MNIST dataset for convolutions """

#     ds = convert_to_one_hot_encoding(ds, feature="label")
#     ds = normalise_feature(ds, feature="image", factor=norm_factor) ## TODO use standard normalisation for MNIST

#     if subset_size != "all":
#         if seed is None:
#             seed = time.time_ns()
#             print("WARNING: no seed provided. Using time.time_ns()")
#         if subset_size > ds.num_rows:
#             subset_size = ds.num_rows
#             print("WARNING: subset_size bigger than dataset. Using all datapoints")
#         np.random.seed(seed)
#         ds = ds.select(np.random.randint(0, ds.num_rows, subset_size))

#     ## Add a channel dimension
#     ds = ds.map(lambda x: {"image": np.reshape(x["image"], (1, 28, 28))})

#     ## Warning. Always make sure your datapoints are first, and labels second
#     ds = reorder_dataset_features(ds, feature_order)
#     return ds




if __name__ == "__main__":

    import time

    ds = load_jax_dataset(path="mnist", split="train")
    print(f"Features are: {get_dataset_features(ds)}")

    ds = convert_to_one_hot_encoding(ds, feature="label")
    print(ds["image"].shape)

    start = time.time()
    ds = extract_all_data(ds, features=["image", "label"])
    print(type(ds[0]))
    print(f"Time to extract: {time.time() - start}")

# %%
