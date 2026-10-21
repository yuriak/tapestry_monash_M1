import pytest
from tapestry_m1.core.io import ContractError
from tapestry_m1.core.artifacts import atomic_bytes, identity


def test_atomic_outputs(tmp_path):
    f=tmp_path/'out';atomic_bytes(f,b'one');atomic_bytes(f,b'one')
    with pytest.raises(ContractError):atomic_bytes(f,b'two')
    assert identity({'a':1,'b':2})==identity({'b':2,'a':1})
