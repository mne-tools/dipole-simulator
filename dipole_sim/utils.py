import numpy as np
from functools import partial
from mne.transforms import combine_transforms, invert_transform, Transform


def create_head_grid(info, grid_steps=50):
    """Find max. extensoion of the head in either dimension, and create a
    grid corresponding to our pre-computed forward solutions.
    """
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

    x_grid = np.linspace(start=xmin, stop=xmax, num=grid_steps).round(3)
    y_grid = np.linspace(start=ymin, stop=ymax, num=grid_steps).round(3)
    z_grid = np.linspace(start=zmin, stop=zmax, num=grid_steps).round(3)
    grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij', sparse=True)
    return grid


def find_closest(a, x):
    """Find the element in the array a that's closest to the scalar x.
    """
    a = a.squeeze()
    idx = np.abs(a - x).argmin()
    return a[idx]


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


def gen_ras_to_head_trans(head_to_mri_t, t1_img):
    mri_to_head_t = invert_transform(head_to_mri_t)

    # RAS <> VOXEL
    ras_to_vox_t = Transform(fro='ras', to='mri_voxel',
                             trans=t1_img.header.get_ras2vox())
    vox_to_mri_t = Transform(fro='mri_voxel', to='mri',
                             trans=t1_img.header.get_vox2ras_tkr())

    # RAS <> MRI
    ras_to_mri_t = combine_transforms(ras_to_vox_t,
                                      vox_to_mri_t,
                                      fro='ras', to='mri')

    ras_to_head_t = combine_transforms(ras_to_mri_t,
                                       mri_to_head_t,
                                       fro='ras', to='head')

    return ras_to_head_t


def get_axis_names_from_slice(slice_view, all_axes):
    if slice_view == 'x':
        x_idx = 'y'
        y_idx = 'z'
    elif slice_view == 'y':
        x_idx = 'x'
        y_idx = 'z'
    elif slice_view == 'z':
        x_idx = 'x'
        y_idx = 'y'

    return x_idx, y_idx
