import pytest


def test_atomic_object_dict_inplace():
    from abcoder.atomic import atomic_object

    obj = {"a": 1}
    obj_id = id(obj)
    with atomic_object(obj) as box:
        box["obj"]["b"] = 2
    assert id(obj) == obj_id
    assert obj == {"a": 1, "b": 2}


def test_atomic_object_list_inplace():
    from abcoder.atomic import atomic_object

    obj = [1, 2]
    obj_id = id(obj)
    with atomic_object(obj) as box:
        box["obj"].append(3)
    assert id(obj) == obj_id
    assert obj == [1, 2, 3]


def test_atomic_object_list_no_commit_on_exception():
    from abcoder.atomic import atomic_object

    obj = [1, 2]
    obj_id = id(obj)
    with pytest.raises(RuntimeError):
        with atomic_object(obj) as box:
            box["obj"].append(9)
            raise RuntimeError("boom")
    assert id(obj) == obj_id
    assert obj == [1, 2]


def test_atomic_object_set_inplace():
    from abcoder.atomic import atomic_object

    obj = {1}
    obj_id = id(obj)
    with atomic_object(obj) as box:
        box["obj"]

        def aa(obj):
            obj.add(2)

        aa(box["obj"])

    assert id(obj) == obj_id
    assert obj == {1, 2}


def test_atomic_object_numpy_same_shape():
    np = pytest.importorskip("numpy")
    from abcoder.atomic import atomic_object

    obj = np.array([[1, 2], [3, 4]])
    obj_id = id(obj)
    with atomic_object(obj) as box:
        box["obj"][0, 0] = 9
    assert id(obj) == obj_id
    assert obj[0, 0] == 9


def test_atomic_object_numpy_shape_diff_reassign():
    np = pytest.importorskip("numpy")
    from abcoder.atomic import atomic_object

    obj = np.arange(6).reshape(2, 3)
    obj_id = id(obj)
    with atomic_object(obj, mode="reassign") as box:
        box["obj"] = box["obj"].reshape(3, 2)
    assert id(obj) == obj_id  # original reference unchanged
    obj = obj if False else obj  # no-op to silence linters


def test_atomic_object_pandas_dataframe():
    pd = pytest.importorskip("pandas")
    from abcoder.atomic import atomic_object

    obj = pd.DataFrame({"x": [1, 2]}, index=["a", "b"])
    obj_id = id(obj)
    with atomic_object(obj) as box:
        box["obj"]["y"] = [10, 20]
    assert id(obj) == obj_id
    assert list(obj.columns) == ["x", "y"]
    assert list(obj["y"]) == [10, 20]


def test_atomic_object_scipy_sparse():
    sp = pytest.importorskip("scipy.sparse")
    from abcoder.atomic import atomic_object

    obj = sp.csr_matrix([[1, 0], [0, 2]])
    obj_id = id(obj)
    with atomic_object(obj) as box:
        new = box["obj"].copy()
        new[0, 1] = 3
        box["obj"] = new
    assert id(obj) == obj_id
    assert obj.toarray().tolist() == [[1, 3], [0, 2]]


def test_atomic_object_anndata_like_if_available():
    anndata = pytest.importorskip("anndata")
    import numpy as np
    from abcoder.atomic import atomic_object

    A = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))
    A_id = id(A)
    with atomic_object(A) as box:
        box["obj"].obs["g"] = ["a", "b"]
    assert id(A) == A_id
    assert "g" in A.obs.columns


def test_register_atomic_strategy_custom_class():
    from abcoder.atomic import atomic_object, register_atomic_strategy

    class Foo:
        def __init__(self, x):
            self.x = x

    def cpy(f: Foo) -> Foo:
        return Foo(f.x)

    def cmt(dst: Foo, src: Foo) -> None:
        dst.x = src.x

    register_atomic_strategy(Foo, cpy, cmt)

    f = Foo(1)
    fid = id(f)
    with atomic_object(f) as box:
        box["obj"].x = 42
    assert id(f) == fid
    assert f.x == 42


def test_atomic_object_fallback_generic_object_dict_sync():
    from abcoder.atomic import atomic_object

    class Bar:
        def __init__(self):
            self.a = 1
            self.b = {"k": 2}

    obj = Bar()
    oid = id(obj)
    with atomic_object(obj) as box:
        box["obj"].a = 10
        box["obj"].b["k"] = 20
    assert id(obj) == oid
    assert obj.a == 10
    assert obj.b == {"k": 20}


def test_atomic_object_tuple_immutable_requires_reassign():
    from abcoder.atomic import atomic_object

    tpl = (1, 2)
    with pytest.raises(TypeError):
        with atomic_object(tpl) as box:
            box["obj"] = (1, 2, 3)


def test_atomic_object_no_commit_on_exception_dict():
    from abcoder.atomic import atomic_object

    obj = {"a": 1}
    obj_id = id(obj)
    with pytest.raises(RuntimeError):
        with atomic_object(obj) as box:
            box["obj"]["a"] = 2
            raise RuntimeError("boom")
    # original object must remain unchanged
    assert id(obj) == obj_id
    assert obj == {"a": 1}


def test_atomic_object_no_commit_on_exception_pandas():
    pd = pytest.importorskip("pandas")
    from abcoder.atomic import atomic_object

    obj = pd.DataFrame({"x": [1, 2]})
    obj_id = id(obj)
    with pytest.raises(RuntimeError):
        with atomic_object(obj) as box:
            box["obj"]["y"] = [10, 20]
            raise RuntimeError("boom")
    # original must be unchanged
    assert id(obj) == obj_id
    assert list(obj.columns) == ["x"]


def test_atomic_objects_multi_success_and_rollback():
    from abcoder.atomic import atomic_objects

    a = {"v": 1}
    b = [1]
    aid, bid = id(a), id(b)

    # success path: both commit
    with atomic_objects({"a": a, "b": b}) as box:
        box["a"]["v"] = 2
        box["b"].append(2)
    assert id(a) == aid and id(b) == bid
    assert a == {"v": 2} and b == [1, 2]

    # failure path: neither commit
    a = {"v": 1}
    b = [1]
    aid, bid = id(a), id(b)
    with pytest.raises(RuntimeError):
        with atomic_objects({"a": a, "b": b}) as box:
            box["a"]["v"] = 9
            box["b"].append(9)
            raise RuntimeError("boom")
    assert id(a) == aid and id(b) == bid
    assert a == {"v": 1} and b == [1]


def test_atomic_objects_with_lists_success_and_rollback():
    from abcoder.atomic import atomic_objects

    a = [1]
    b = [10]
    aid, bid = id(a), id(b)

    # success
    with atomic_objects({"a": a, "b": b}) as box:

        def aa(a, b):
            a.append(2)
            b.append(11)

        aa(box["a"], box["b"])

    assert id(a) == aid and id(b) == bid
    assert a == [1, 2] and b == [10, 11]

    # rollback
    a = [1]
    b = [10]
    aid, bid = id(a), id(b)
    with pytest.raises(RuntimeError):
        with atomic_objects({"a": a, "b": b}) as box:
            box["a"].append(9)
            box["b"].append(19)
            raise RuntimeError("boom")
    assert id(a) == aid and id(b) == bid
    assert a == [1] and b == [10]


def test_atomic_object_numpy_no_commit_on_exception():
    np = pytest.importorskip("numpy")
    from abcoder.atomic import atomic_object

    obj = np.array([[1, 2], [3, 4]])
    obj_copy = obj.copy()
    with pytest.raises(RuntimeError):
        with atomic_object(obj) as box:
            box["obj"][0, 0] = 9
            raise RuntimeError("boom")
    assert (obj == obj_copy).all()


def test_atomic_objects_commit_failure_rollback():
    from abcoder.atomic import atomic_objects

    class FailCommit:
        def __init__(self, x):
            self.x = x

    a = {"v": 1}
    b = FailCommit(1)
    aid, bid = id(a), id(b)

    def fail(dst, src):
        raise RuntimeError("commit failed")

    with pytest.raises(RuntimeError):
        with atomic_objects({"a": a, "b": b}, commit_map={"b": fail}) as box:
            box["a"]["v"] = 2
            box["b"].x = 2
    # both unchanged due to rollback
    assert id(a) == aid and id(b) == bid
    assert a == {"v": 1} and b.x == 1


def test_atomic_object_anndata_no_commit_on_exception():
    anndata = pytest.importorskip("anndata")
    import numpy as np
    from abcoder.atomic import atomic_object

    A = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))
    A_copy = A.copy()
    with pytest.raises(RuntimeError):
        with atomic_object(A) as box:
            box["obj"].obs["g"] = ["a", "b"]
            raise RuntimeError("boom")
    # unchanged
    assert list(A_copy.obs.columns) == list(A.obs.columns)
    assert (A_copy.X == A.X).all()


def test_atomic_objects_with_anndata_success_and_rollback():
    anndata = pytest.importorskip("anndata")
    import numpy as np
    from abcoder.atomic import atomic_objects

    A = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))
    B = {"v": 1}

    # success
    A_id, B_id = id(A), id(B)
    with atomic_objects({"A": A, "B": B}) as box:
        box["A"].obs["g"] = ["x", "y"]
        box["B"]["v"] = 2
    assert id(A) == A_id and id(B) == B_id
    assert "g" in A.obs.columns and B["v"] == 2

    # rollback on failure
    A = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))
    B = {"v": 1}
    A_id, B_id = id(A), id(B)
    with pytest.raises(RuntimeError):
        with atomic_objects({"A": A, "B": B}) as box:
            box["A"].obs["g"] = ["x", "y"]
            box["B"]["v"] = 9
            raise RuntimeError("boom")
    assert id(A) == A_id and id(B) == B_id
    assert "g" not in A.obs.columns and B["v"] == 1


def test_atomic_object_restore_to_last_committed_on_exception():
    from abcoder.atomic import atomic_object

    # init and commit once
    obj = [1, 2, 3]
    oid = id(obj)
    with atomic_object(obj) as box:
        box["obj"].append(4)
    assert id(obj) == oid
    assert obj == [1, 2, 3, 4]
    # next transaction fails; object should stay at last committed value
    with pytest.raises(RuntimeError):
        with atomic_object(obj) as box:
            box["obj"]

            def aa(obj):
                obj.append(2)
                raise RuntimeError("trigger error")

            aa(box["obj"])

    assert id(obj) == oid
    assert obj == [1, 2, 3, 4]


def test_my_list():
    from abcoder.atomic import atomic_objects

    my_list = [1, 2, 3]
    my_list_id = id(my_list)

    # success
    with atomic_objects({"my_list": my_list}) as box:

        def aa(my_list):
            my_list.append(4)

        aa(box["my_list"])

    assert id(my_list) == my_list_id
    assert my_list == [1, 2, 3, 4]

    # success
    with atomic_objects({"my_list": my_list}) as box:

        def aa(my_list):
            my_list.append(5)

        aa(box["my_list"])

    assert id(my_list) == my_list_id
    assert my_list == [1, 2, 3, 4, 5]

    # # rollback
    # my_list = [1, 2, 3]
    # my_list_id = id(my_list)
    # with pytest.raises(RuntimeError):
    #     with atomic_objects({"a": a, "b": b}) as box:
    #         box["a"].append(9)
    #         box["b"].append(19)
    #         raise RuntimeError("boom")
    # assert id(a) == aid and id(b) == bid
    # assert a == [1] and b == [10]
