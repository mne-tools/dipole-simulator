import numpy as np
import warnings
from nilearn.plotting import plot_anat
import matplotlib.pyplot as plt
import xarray as xr

from forward import _create_format_coord


def plot_slice(widget, state, axis, pos, img_data):
    if axis == 'x':
        coord = dict(x=pos)  # Used for xarray slicing below
        x_axis = 'y'
        y_axis = 'z'
    elif axis == 'y':
        coord = dict(y=pos)
        x_axis = 'x'
        y_axis = 'z'
    elif axis == 'z':
        coord = dict(z=pos)
        x_axis = 'x'
        y_axis = 'y'
    else:
        raise ValueError('plane must be x, y, or z')

    fig = widget['fig'][axis]
    ax = fig.axes[0]
    ax.images = []

    kwargs = dict(x=x_axis, y=y_axis,
                  cmap='gray', vmin=0, vmax=127,
                  add_colorbar=False, add_labels=False,
                  ax=ax)

    xr.plot.imshow(img_data.sel(**coord, method='nearest'), **kwargs)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.format_coord = _create_format_coord(axis)
    draw_crosshairs(widget=widget, state=state)
    fig.canvas.draw()

    label_text = state['label_text']
    label_text['x'] = (f'sagittal '
                       f'(x = {round(state["slice_coord"]["x"]["val"])} mm)')
    label_text['y'] = (f'coronal '
                       f'(y = {round(state["slice_coord"]["y"]["val"])} mm)')
    label_text['z'] = (f'axial '
                       f'(z = {round(state["slice_coord"]["z"]["val"])} mm)')

    for axis in (x_axis, y_axis):
        _update_axis_label(widget=widget, state=state, axis=axis)


def create_slice_fig(handle_click, handle_enter, handle_leave):
    fig, ax = plt.subplots(1, figsize=(2, 2))
    fig.canvas.mpl_connect('button_press_event', handle_click)
    fig.canvas.mpl_connect('figure_enter_event', handle_enter)
    fig.canvas.mpl_connect('figure_leave_event', handle_leave)
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.resizable = False
    ax.set_position([0, 0, 1, 1])
    fig.tight_layout()
    fig.set_tight_layout(True)
    return fig


def _update_axis_label(widget, state, axis):
    label = widget['label']['axis'][axis]
    label_text = state['label_text'][axis]
    label.value = label_text


def draw_crosshairs(widget, state):
    kwargs = dict(color='white', label='crosshair', lw=0.5)
    # Remove potentially existing crosshairs.
    for axis in widget['fig'].keys():
        if not widget['fig'][axis].axes:
            continue

        ax = widget['fig'][axis].axes[-1]
        lines_to_keep = [line for line in ax.lines
                         if line.get_label() != 'crosshair']
        ax.lines = lines_to_keep

        x_axis, y_axis = get_axis_names_from_slice(
            slice_view=axis, all_axes=widget['fig'].keys())

        ax.axvline(state['crosshair_pos'][x_axis], **kwargs)
        ax.axhline(state['crosshair_pos'][y_axis], **kwargs)

    widget['fig']['x'].canvas.draw()
    widget['fig']['y'].canvas.draw()
    widget['fig']['z'].canvas.draw()


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
