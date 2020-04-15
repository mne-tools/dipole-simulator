import numpy as np
from functools import partial
import pandas as pd
import mne


def gen_forward_solution(pos, bem, info, trans, verbose=True):
    # NOTE:
    # Both, dipole amplitude and orientation, are set to arbitrary values
    # here: They are merely required to instantiate a Dipole object, which
    # we then use to conveniently retrieve a forward solution via
    # `make_forward_dipole()`. This forward solution is returned in "fixed"
    # orientation. We then convert it back to "free" orientation mode before
    # returning the forward object.

    amplitude = np.array([1]).reshape(1,)  # Arbitraty amplitude.
    ori = np.array([1., 1., 1.]).reshape(1, 3)  # Arbitrary orientation.
    ori /= np.linalg.norm(ori)
    pos = pos.reshape(1, 3)
    gof = np.array([100]).reshape(1,)
    dip = mne.Dipole(times=[0], pos=pos, ori=ori,
                     amplitude=amplitude, gof=gof)
    fwd, _ = mne.make_forward_dipole(dip, bem=bem, info=info, trans=trans,
                                     verbose=verbose)
    fwd = mne.convert_forward_solution(fwd, force_fixed=False, verbose=verbose)
    return fwd


def _create_format_coord(axis):
    if axis == 'topomap':

        def format_coord(x, y):
            x *= 1000
            y *= 1000
            x = int(round(x))
            y = int(round(y))
            return f'x={x} mm, y={y} mm'

        return format_coord

    if axis == 'x':
        x_label = 'y'
        y_label = 'z'
    elif axis == 'y':
        x_label = 'x'
        y_label = 'z'
    elif axis == 'z':
        x_label = 'x'
        y_label = 'y'

    # FIXME
    def format_coord(x, y, x_label, y_label):
        return f'{x_label}={x:.1f} mm, {y_label}={y:.1f} mm'

    return partial(format_coord, x_label=x_label, y_label=y_label)


def load_fwd_lookup_table(fwd_path):
    lookup_table_fname = 'fwd_lookup_table.csv'
    lookup_table_path = fwd_path / lookup_table_fname

    # Convert to str dtype to work around floating point precision issues
    lookup_table = pd.read_csv(lookup_table_path,
                               dtype=dict(x=str, y=str, z=str))
    lookup_table = lookup_table.rename(columns={'success': 'fwd_exists'})
    lookup_table = lookup_table.set_index(['x', 'y', 'z'])
    return lookup_table
