import pathlib
import numpy as np
import itertools
from joblib import Parallel, delayed
import mne

data_path = pathlib.Path('data')
subjects_dir = data_path / 'subjects'
subject = 'sample'
fwd_dir = data_path / 'fwd'

# Transformation matrix.
trans_fname = data_path / f'{subject}-trans.fif'
head_to_mri_t = mne.read_trans(trans_fname)
del trans_fname

# BEM solution.
bem_fname = data_path / f'{subject}-bem-sol.fif'
bem = mne.read_bem_solution(bem_fname, verbose=False)
del bem_fname

# Info object.
evoked_fname = data_path / f'{subject}-ave.fif'
evoked = mne.read_evokeds(evoked_fname, verbose='warning')[0]
evoked.pick_types(meg=True, eeg=True)
info = evoked.info
info['projs'] = []
info['bads'] = []
del evoked_fname, evoked


def _gen_forward_solution(pos, bem, info, trans):
    """Invoked by gen_forward_solution."""
    ori = np.eye(3)
    pos = np.tile(pos, (3, 1))
    dip = mne.Dipole(times=np.arange(3), pos=pos, amplitude=3 * [10e-9],
                     ori=ori, gof=3 * [100])
    fwd, _ = mne.make_forward_dipole(dip, bem=bem, info=info, trans=trans,
                                     verbose='warning')

    return fwd


def gen_forward_solution(x, y, z, bem, info, trans):
    msg = (f'Processing forward solution for dipole location: '
           f'x={x}, y={y}, z={z} [m, MNE Head]')
    print(msg)
    pos = np.array([x, y, z])

    try:
        fwd = _gen_forward_solution(pos=pos, bem=bem, info=info,
                                    trans=head_to_mri_t)
    except RuntimeError as e:
        if 'No points left in source space ' in str(e):
            print('… skipping (location outside skull)')
            return
        else:
            raise e

    fwd_fname = fwd_dir / f'{subject}-{pos[0]}-{pos[1]}-{pos[2]}-fwd.fif'
    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)

    print('… done.')


def find_head_dims(info):
    xmin, xmax = None, None
    ymin, ymax = None, None
    zmin, zmax = None, None

    for dig in info['dig']:
        x, y, z = dig['r']

        if xmin is None:
            xmin = x
            xmax = x
        elif x < xmin:
            xmin = x
        elif x > xmax:
            xmax = x

        if ymin is None:
            ymin = y
            ymax = y
        elif y < ymin:
            ymin = y
        elif y > ymax:
            ymax = y

        if zmin is None:
            zmin = z
            zmax = z
        elif z < zmin:
            zmin = z
        elif z > zmax:
            zmax = z

    return dict(x=(xmin, xmax),
                y=(ymin, ymax),
                z=(zmin, zmax))


def main():
    head_dims = find_head_dims()
    grid_steps = 5
    x_grid = np.linspace(start=head_dims['x'][0],
                         stop=head_dims['x'][1],
                         num=grid_steps)
    y_grid = np.linspace(start=head_dims['y'][0],
                         stop=head_dims['y'][1],
                         num=grid_steps)
    z_grid = np.linspace(start=head_dims['z'][0],
                         stop=head_dims['z'][1],
                         num=grid_steps)

    x_grid = np.round(x_grid, 3)
    y_grid = np.round(y_grid, 3)
    z_grid = np.round(z_grid, 3)

    d = delayed(gen_forward_solution)
    xyz = itertools.product(x_grid, y_grid, z_grid)
    Parallel(n_jobs=8)(d(x, y, z, bem=bem, info=info, trans=head_to_mri_t)
                       for x, y, z in xyz)


if __name__ == '__main__':
    main()
