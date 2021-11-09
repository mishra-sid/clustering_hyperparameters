import sklearn.decomposition as skd
__all__ = [
    "transform",
]


def transform(data_inp, **kwargs):
    pca_model = skd.PCA(**kwargs)
    pca_model.fit(data_inp)

