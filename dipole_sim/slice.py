import warnings
from nilearn.plotting import plot_anat

import matplotlib.pyplot as plt

from utils import _create_format_coord, get_axis_names_from_slice


def plot_slice(widget, state, axis, pos, t1_img):
    old_label_text = state['label_text'][axis]
    new_label_text = old_label_text + ' [updating]'
    state['label_text'][axis] = new_label_text
    _update_axis_label(widget, state, axis)

    fig = widget['fig'][axis]

    with warnings.catch_warnings():  # Suppress DeprecationWarning
        warnings.simplefilter("ignore")
        img = plot_anat(t1_img, display_mode=axis, cut_coords=(pos,),
                        figure=fig, dim=-0.5)

    img.axes[pos].ax.format_coord = _create_format_coord(axis)
    draw_crosshairs(widget=widget, state=state)
    fig.canvas.draw()
    state['label_text'][axis] = old_label_text
    _update_axis_label(widget, state, axis)


def create_slice_fig(handle_click, handle_enter, handle_leave):
    fig = plt.figure(figsize=(2, 2))
    fig.canvas.mpl_connect('button_press_event', handle_click)
    fig.canvas.mpl_connect('figure_enter_event', handle_enter)
    fig.canvas.mpl_connect('figure_leave_event', handle_leave)
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.resizable = False
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
