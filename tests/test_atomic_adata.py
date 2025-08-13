import pytest
import numpy as np
import pandas as pd
import anndata as ad

from abcoder.backend import atomic_adata


def make_base_adata():
    X = np.array([[1, 2], [3, 4]])
    obs = pd.DataFrame({"group": ["A", "B"]}, index=["c1", "c2"])
    var = pd.DataFrame({"gene": ["g1", "g2"]}, index=["g1", "g2"])
    A = ad.AnnData(X=X, obs=obs, var=var)
    A.obsm["pca"] = np.array([[0.1, 0.2], [0.3, 0.4]])
    A.varm["eig"] = np.array([1.0, 0.5])
    A.layers["raw"] = X.copy()
    A.uns["meta"] = {"version": 1}
    return A


def test_atomic_commit_success():
    adata = make_base_adata()

    with atomic_adata(adata) as box:
        A = box["A"]
        A.X = np.array([[10, 20], [30, 40]])
        A.obs.loc["c1", "group"] = "Z"
        A.var.loc["g2", "gene"] = "g2_new"
        A.obsm["pca"][0, 0] = 9.9
        A.varm["eig"] = np.array([1.5, 0.8])
        A.layers["scaled"] = np.array([[0.1, 0.2], [0.3, 0.4]])
        A.uns["meta"]["note"] = "updated"

    assert (adata.X == np.array([[10, 20], [30, 40]])).all()
    assert adata.obs.loc["c1", "group"] == "Z"
    assert adata.var.loc["g2", "gene"] == "g2_new"
    assert adata.obsm["pca"][0, 0] == 9.9
    assert np.isclose(adata.varm["eig"][1], 0.8)
    assert (adata.layers["scaled"] == np.array([[0.1, 0.2], [0.3, 0.4]])).all()
    assert adata.uns["meta"]["note"] == "updated"


def test_atomic_rollback_on_exception():
    adata = make_base_adata()
    assert adata.n_obs == 2
    assert (adata.X == np.array([[1, 2], [3, 4]])).all()
    with pytest.raises(RuntimeError):
        with atomic_adata(adata) as box:
            A = box["A"]
            A.X = np.array([[999, 999], [999, 999]])
            assert (A.X == np.array([[999, 999], [999, 999]])).all()
            raise RuntimeError("boom")

    assert (adata.X == np.array([[1, 2], [3, 4]])).all()


def test_dimension_change_support():
    adata = make_base_adata()
    assert adata.n_obs == 2
    assert adata.n_vars == 2

    with atomic_adata(adata) as box:
        adata = box["A"][:1, :]

    assert adata.n_obs == 1
    assert adata.n_vars == 2
    assert (adata.X == np.array([[1, 2]])).all()
    assert adata.obs.index.tolist() == ["c1"]
