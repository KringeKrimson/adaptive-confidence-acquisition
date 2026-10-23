import numpy as np
import pytest
from aca.frozen import calibration_data
from aca.routing import tune_policies


def test_method_selection_never_consults_assessment():
    rows=[dict(id=f'q{i}',partition='calibration' if i<8 else 'assessment',
        small_correct=float(i%2),strong_correct=1.,cheap=.2 if i%2==0 else .8,
        ic=.1 if i%2==0 else .9,small_seconds=.1,strong_seconds=1.,ic_seconds=.2,
        probe_seconds=.001) for i in range(16)]
    ids=[f'q{i}' for i in range(8)]
    first=calibration_data(rows,ids)
    for row in rows[8:]:
        for key in ['small_correct','strong_correct','cheap','ic','strong_seconds']:
            row[key]=float('nan')
    second=calibration_data(rows,ids)
    assert all(np.array_equal(first[k],second[k]) for k in first)
    assert tune_policies(first,[.1])==tune_policies(second,[.1])
    with pytest.raises(ValueError,match='calibration'):
        calibration_data(rows,['q8'])
