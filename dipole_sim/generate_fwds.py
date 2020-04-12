import pathlib
import numpy as np
import itertools
from joblib import Parallel, delayed
import mne

from forward import gen_forward_solution
from utils import create_head_grid


data_path = pathlib.Path('../data')
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


def gen_forward_files(x, y, z, bem, info, trans):
    msg = (f'Processing forward solution for dipole location: '
           f'x={x}, y={y}, z={z} [m, MNE Head]')
    print(msg)
    pos = np.array([x, y, z])

    try:
        fwd = gen_forward_solution(pos=pos, bem=bem, info=info,
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


def main():
    head_grid_steps = 25
    head_grid = create_head_grid(info=info, grid_steps=head_grid_steps)

    x_grid = np.round(head_grid[0].squeeze(), 3)
    y_grid = np.round(head_grid[1].squeeze(), 3)
    z_grid = np.round(head_grid[2].squeeze(), 3)

    d = delayed(gen_forward_files)
    xyz = itertools.product(x_grid, y_grid, z_grid)
    Parallel(n_jobs=8)(d(x, y, z, bem=bem, info=info, trans=head_to_mri_t)
                       for x, y, z in xyz)


if __name__ == '__main__':
    main()
