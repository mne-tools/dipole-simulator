from matplotlib.backend_bases import MouseButton
from mne.transforms import apply_trans

from evoked_field import plot_evoked, reset_topomaps
from slice import plot_slice, draw_crosshairs
from dipole import (draw_dipole_arrows, remove_dipole_arrows,
                    plot_dipole_pos_marker, remove_dipole_pos_markers,
                    plot_dipole_ori_marker, remove_dipole_ori_markers)
from cursor import enable_crosshair_cursor
from utils import get_axis_names_from_slice


def handle_click(event, widget, markers, state, evoked, ras_to_head_t,
                 fwd_path, subject, info, t1_img):
    if event.button != MouseButton.LEFT:
        return

    in_ax = event.inaxes
    if in_ax.figure in widget['topomap_fig'].values():
        return

    for axis, fig in widget['fig'].items():
        if fig is in_ax.figure:
            break

    x = event.xdata
    y = event.ydata

    x_idx, y_idx = get_axis_names_from_slice(axis)
    remaining_idx = axis

    if state['mode'] == 'slice_browser':
        handle_click_in_slice_browser_mode(widget, markers, state, x, y, x_idx,
                                           y_idx, evoked, t1_img)
    elif state['mode'] == 'set_dipole_pos':
        handle_click_in_set_dipole_pos_mode(widget, state, x_idx, y_idx,
                                            remaining_idx, x, y, ras_to_head_t)
    elif state['mode'] == 'set_dipole_ori':
        # Construct the 3D coordinates of the clicked-on point
        handle_click_in_set_dipole_ori_mode(widget, state, x_idx, y_idx,
                                            remaining_idx, x, y, ras_to_head_t)

    if state['dipole_pos']['x'] is not None:
        plot_dipole_pos_marker(widget, markers, state)

    if state['dipole_ori']['x'] is not None:
        plot_dipole_ori_marker(widget, markers, state)

    if (state['dipole_pos']['x'] is not None and
            state['dipole_ori']['x'] is not None and
            state['dipole_pos'] != state['dipole_ori']):
        draw_dipole_arrows(widget, state)

    draw_crosshairs(widget=widget, state=state)

    if (state['dipole_pos']['x'] is not None and
            state['dipole_ori']['x'] is not None and
            state['dipole_pos'] != state['dipole_ori']):

        try:
            plot_evoked(widget, state, fwd_path, subject, info, ras_to_head_t)
        except RuntimeError as e:
            msg = f'Error while calculating generated fields:\n\n{e}'
            print(msg)


def handle_leave(event):
    pass


def handle_enter(event):
    pass


def enter_set_dipole_pos_mode():
    pass


def leave_set_dipole_pos_mode():
    pass


def enter_set_dipole_ori_mode():
    pass


def leave_set_dipole_ori_mode():
    pass


def handle_click_in_slice_browser_mode(widget, markers, state, x, y, x_idx,
                                       y_idx, evoked, t1_img):
    state['slice_coord'][x_idx]['val'] = x
    state['slice_coord'][y_idx]['val'] = y
    state['crosshair_pos'][x_idx] = x
    state['crosshair_pos'][y_idx] = y

    remove_dipole_arrows(widget)
    remove_dipole_pos_markers(widget, markers, state)
    remove_dipole_ori_markers(widget, markers, state)

    widget['label']['dipole_pos'].value = 'Not set'
    widget['label']['dipole_ori'].value = 'Not set'

    state['dipole_pos']['x'] = None
    state['dipole_pos']['y'] = None
    state['dipole_pos']['z'] = None
    state['dipole_ori']['x'] = None
    state['dipole_ori']['y'] = None
    state['dipole_ori']['z'] = None

    # widget['fig'][x_idx].axes[0].clear()
    # widget['fig'][y_idx].axes[0].clear()
    widget['fig'][x_idx].axes[0].images = []
    widget['fig'][x_idx].axes[0].texts = []
    widget['fig'][y_idx].axes[0].images = []
    widget['fig'][y_idx].axes[0].texts = []

    plot_slice(widget, state, x_idx, x, t1_img)
    plot_slice(widget, state, y_idx, y, t1_img)

    enable_crosshair_cursor(widget)
    reset_topomaps(widget, evoked)


def handle_click_in_set_dipole_pos_mode(widget, state, x_idx, y_idx,
                                        remaining_idx, x, y, ras_to_head_t):
    # for axis in state['dipole_pos'].keys():
    #     state['dipole_pos'][axis] = state['slice_coord'][axis]['val']

    # Construct the 3D coordinates of the clicked-on point
    dipole_pos_ras = dict()
    dipole_pos_ras[x_idx] = x
    dipole_pos_ras[y_idx] = y
    dipole_pos_ras[remaining_idx] = state['slice_coord'][remaining_idx]['val']

    state['dipole_pos'] = dipole_pos_ras
    dipole_pos_head = apply_trans(trans=ras_to_head_t,
                                  pts=(dipole_pos_ras['x'],
                                       dipole_pos_ras['y'],
                                       dipole_pos_ras['z']))
    dipole_pos_head /= 1000
    dipole_pos_head = dict(x=dipole_pos_head[0], y=dipole_pos_head[1],
                           z=dipole_pos_head[2])

    label_text = (f"x={int(round(dipole_pos_ras['x']))}, "
                  f"y={int(round(dipole_pos_ras['y']))}, "
                  f"z={int(round(dipole_pos_ras['z']))} [mm, MRI RAS] ⟶ "
                  f"x={round(dipole_pos_head['x'], 3)}, "
                  f"y={round(dipole_pos_head['y'], 3)}, "
                  f"z={round(dipole_pos_head['z'], 3)} [m, MNE Head]")
    widget['label']['dipole_pos'].value = label_text

    leave_set_dipole_pos_mode()


def handle_click_in_set_dipole_ori_mode(widget, state, x_idx, y_idx,
                                        remaining_idx, x, y, ras_to_head_t):
    dipole_ori_ras = dict()
    dipole_ori_ras[x_idx] = x
    dipole_ori_ras[y_idx] = y
    dipole_ori_ras[remaining_idx] = state['slice_coord'][remaining_idx]['val']

    state['dipole_ori'] = dipole_ori_ras

    dipole_ori_head = apply_trans(trans=ras_to_head_t,
                                  pts=(dipole_ori_ras['x'],
                                       dipole_ori_ras['y'],
                                       dipole_ori_ras['z']))
    dipole_ori_head /= 1000
    dipole_ori_head = dict(x=dipole_ori_head[0], y=dipole_ori_head[1],
                           z=dipole_ori_head[2])

    label_text = (f"x={int(round(dipole_ori_ras['x']))}, "
                  f"y={int(round(dipole_ori_ras['y']))}, "
                  f"z={int(round(dipole_ori_ras['z']))} [mm, MRI RAS] ⟶ "
                  f"x={round(dipole_ori_head['x'], 3)}, "
                  f"y={round(dipole_ori_head['y'], 3)}, "
                  f"z={round(dipole_ori_head['z'], 3)} [m, MNE Head]")
    widget['label']['dipole_ori'].value = label_text
