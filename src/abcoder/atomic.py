from contextlib import contextmanager
from copy import deepcopy
from functools import singledispatch
from typing import Any, Callable, Optional


# Strategy registry: copy and commit
@singledispatch
def _atomic_copy(x: Any) -> Any:
    # prefer object-provided hook
    if hasattr(x, "__atomic_copy__"):
        return x.__atomic_copy__()
    # generic: .copy(deep=True)->.copy()->deepcopy
    try:
        return x.copy(deep=True)
    except Exception:
        try:
            return x.copy()
        except Exception:
            return deepcopy(x)


@singledispatch
def _atomic_commit(dst: Any, src: Any) -> None:
    # prefer object-provided hook
    if hasattr(dst, "__atomic_commit__"):
        dst.__atomic_commit__(dst, src)
        return
    # default: sync __dict__ if available
    if hasattr(dst, "__dict__") and hasattr(src, "__dict__"):
        dst.__dict__.clear()
        dst.__dict__.update(deepcopy(src.__dict__))
        return
    raise TypeError(
        "Unsupported type for inplace commit; use mode='reassign' or register."
    )


def register_atomic_strategy(
    py_type: type, copy_fn: Callable[[Any], Any], commit_fn: Callable[[Any, Any], None]
) -> None:
    _atomic_copy.register(py_type)(copy_fn)
    _atomic_commit.register(py_type)(commit_fn)


# Built-ins
@_atomic_copy.register(dict)
def _copy_dict(x: dict) -> dict:
    return deepcopy(x)


@_atomic_commit.register(dict)
def _commit_dict(dst: dict, src: dict) -> None:
    dst.clear()
    dst.update(deepcopy(src))


@_atomic_copy.register(list)
def _copy_list(x: list) -> list:
    return deepcopy(x)


@_atomic_commit.register(list)
def _commit_list(dst: list, src: list) -> None:
    dst.clear()
    dst.extend(deepcopy(src))


@_atomic_copy.register(set)
def _copy_set(x: set) -> set:
    return deepcopy(x)


@_atomic_commit.register(set)
def _commit_set(dst: set, src: set) -> None:
    dst.clear()
    dst.update(deepcopy(src))


# numpy
try:
    import numpy as np  # type: ignore

    @_atomic_copy.register(np.ndarray)
    def _copy_nd(x: Any) -> Any:
        return x.copy()

    @_atomic_commit.register(np.ndarray)
    def _commit_nd(dst: Any, src: Any) -> None:
        if dst.shape == src.shape:
            dst[...] = src
        else:
            raise ValueError("shape differs; use allow_resize or reassign.")
except Exception:
    np = None  # type: ignore


# scipy.sparse
try:
    import scipy.sparse as sp  # type: ignore

    @_atomic_copy.register(sp.spmatrix)
    def _copy_sp(x: Any) -> Any:
        return x.copy()

    @_atomic_commit.register(sp.spmatrix)
    def _commit_sp(dst: Any, src: Any) -> None:
        if type(dst) is not type(src):
            raise ValueError("sparse type differs; reassign.")
        if hasattr(dst, "data") and hasattr(src, "data"):
            dst.data = src.data.copy()
        if hasattr(dst, "indices") and hasattr(src, "indices"):
            dst.indices = src.indices.copy()
        if hasattr(dst, "indptr") and hasattr(src, "indptr"):
            dst.indptr = src.indptr.copy()
        try:
            dst._shape = src.shape  # type: ignore[attr-defined]
        except Exception:
            pass
except Exception:
    sp = None  # type: ignore


# pandas
try:
    import pandas as pd  # type: ignore

    @_atomic_copy.register(pd.DataFrame)
    def _copy_df(x: Any) -> Any:
        return x.copy(deep=True)

    @_atomic_commit.register(pd.DataFrame)
    def _commit_df(dst: Any, src: Any) -> None:
        dst.drop(index=dst.index, inplace=True)
        for c in src.columns:
            dst[c] = src[c].copy()
        dst.index = src.index.copy()
        dst.columns = src.columns.copy()

    @_atomic_copy.register(pd.Series)
    def _copy_ser(x: Any) -> Any:
        return x.copy(deep=True)

    @_atomic_commit.register(pd.Series)
    def _commit_ser(dst: Any, src: Any) -> None:
        dst.drop(index=dst.index, inplace=True)
        dst.loc[src.index] = src.values
        dst.index = src.index.copy()
        dst.name = src.name
except Exception:
    pd = None  # type: ignore


# AnnData detection and handling
def _is_anndata(x: Any) -> bool:
    return hasattr(x, "_init_as_actual") and all(
        hasattr(x, a)
        for a in ("X", "obs", "var", "uns", "obsm", "varm", "layers", "raw")
    )


@_atomic_copy.register(object)
def _copy_maybe_anndata(x: Any) -> Any:
    if _is_anndata(x):
        return x.copy()
    # fallback to deepcopy for unknown objects
    return deepcopy(x)


@_atomic_commit.register(object)
def _commit_maybe_anndata(dst: Any, src: Any) -> None:
    if _is_anndata(dst) and _is_anndata(src):
        A = src
        dst._init_as_actual(
            X=A.X if getattr(A, "X", None) is not None else None,
            obs=A.obs.copy(deep=True) if getattr(A, "obs", None) is not None else None,
            var=A.var.copy(deep=True) if getattr(A, "var", None) is not None else None,
            uns=deepcopy(getattr(A, "uns", None))
            if getattr(A, "uns", None) is not None
            else None,
            obsm=A.obsm.copy() if getattr(A, "obsm", None) is not None else None,
            varm=A.varm.copy() if getattr(A, "varm", None) is not None else None,
            layers=A.layers.copy() if getattr(A, "layers", None) is not None else None,
            raw=A.raw.copy() if getattr(A, "raw", None) is not None else None,
        )
        return
    # generic fallback: sync __dict__ if present
    if hasattr(dst, "__dict__") and hasattr(src, "__dict__"):
        dst.__dict__.clear()
        dst.__dict__.update(deepcopy(src.__dict__))
        return
    raise TypeError(
        "Unsupported type for inplace commit; use mode='reassign' or register."
    )


@contextmanager
def atomic_object(
    obj: Any,
    mode: str = "auto",
    allow_resize: bool = True,
    commit: Optional[Callable[[Any, Any], None]] = None,
):
    """Generic atomic context manager using strategy registry.

    - mode: "auto" | "inplace" | "reassign"
    - allow_resize: currently advisory for numpy/scipy (use reassign if shape differs)
    - commit: optional callable(dst, src)
    """
    work = _atomic_copy(obj)
    box = {"obj": work}
    try:
        yield box
    except Exception:
        raise
    else:
        if mode == "reassign":
            return
        if callable(commit):
            commit(obj, box["obj"])
            return
        # default strategy-based commit
        try:
            _atomic_commit(obj, box["obj"])
        except ValueError as e:
            if "shape differs" in str(e) and allow_resize:
                # advise reassign path
                raise ValueError(
                    "shape differs; use mode='reassign' and rebind from box['obj']"
                )
            raise


@contextmanager
def atomic_objects(
    objects: Optional[dict] = None,
    *objs: Any,
    mode: str = "auto",
    allow_resize: bool = True,
    commit_map: Optional[dict] = None,
):
    """Atomic context manager for multiple objects.

    Usage:
        with atomic_objects({"a": A, "b": B}) as box:
            box["a"] ...; box["b"] ...

        with atomic_objects(None, A, B) as box:  # positional
            box["0"] ...; box["1"] ...

    On success, commits all. On any exception or commit failure, nothing is changed.
    """
    # Build name -> original map
    if objects is not None and len(objs) > 0:
        raise ValueError(
            "Provide either a dict 'objects' or positional '*objs', not both"
        )
    if objects is None:
        original_map = {str(i): o for i, o in enumerate(objs)}
    else:
        original_map = dict(objects)

    # Make working copies
    work_map = {name: _atomic_copy(obj) for name, obj in original_map.items()}

    try:
        yield work_map
    except Exception:
        # Abort without committing any
        raise
    else:
        if mode == "reassign":
            return

        # Backup originals for rollback
        backups = {name: _atomic_copy(obj) for name, obj in original_map.items()}

        committed = []
        try:
            for name, target in work_map.items():
                src = original_map[name]
                if commit_map and name in commit_map and callable(commit_map[name]):
                    commit_map[name](src, target)
                else:
                    _atomic_commit(src, target)
                committed.append(name)
        except Exception:
            # Rollback previously committed ones
            for name in reversed(committed):
                try:
                    _atomic_commit(original_map[name], backups[name])
                except Exception:
                    # Best-effort rollback; continue
                    pass
            raise


__all__ = [
    "atomic_object",
    "atomic_objects",
    "register_atomic_strategy",
]
