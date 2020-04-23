from matplotlib.backend_bases import MouseButton
from mne.transforms import apply_trans

from evoked_field import reset_topomaps
from slice import plot_slice, draw_crosshairs, get_axis_names_from_slice
from dipole import (draw_dipole_arrows, remove_dipole_arrows,
                    plot_dipole_pos_marker, remove_dipole_pos_markers,
                    plot_dipole_ori_marker, remove_dipole_ori_markers,
                    update_dipole_pos, update_dipole_ori,
                    draw_dipole_if_necessary)
from cursor import enable_crosshair_cursor


def handle_click(event, widget, markers, state, evoked, ras_to_head_t,
                 fwd_path, subject, info, img_data, t1_img):
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
                                           y_idx, evoked, img_data)
    elif state['mode'] == 'set_dipole_pos':
        handle_click_in_set_dipole_pos_mode(widget, state, x_idx, y_idx,
                                            remaining_idx, x, y, ras_to_head_t)
    elif state['mode'] == 'set_dipole_ori':
        # Construct the 3D coordinates of the clicked-on point
        handle_click_in_set_dipole_ori_mode(widget, state, x_idx, y_idx,
                                            remaining_idx, x, y, ras_to_head_t)

    draw_dipole_if_necessary(state, widget, markers)
    # draw_crosshairs(widget=widget, state=state)


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
                                       y_idx, evoked, img_data):
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
    # widget['fig'][x_idx].axes[0].images = []
    # widget['fig'][x_idx].axes[0].texts = []
    # widget['fig'][y_idx].axes[0].images = []
    # widget['fig'][y_idx].axes[0].texts = []

    plot_slice(widget, state, x_idx, x, img_data)
    plot_slice(widget, state, y_idx, y, img_data)

    enable_crosshair_cursor(widget)
    reset_topomaps(widget=widget, evoked=evoked)


def handle_click_in_set_dipole_pos_mode(widget, state, x_idx, y_idx,
                                        remaining_idx, x, y, ras_to_head_t,
                                        evoked):
    # for axis in state['dipole_pos'].keys():
    #     state['dipole_pos'][axis] = state['slice_coord'][axis]['val']

    # Construct the 3D coordinates of the clicked-on point
    dipole_pos_ras = dict()
    dipole_pos_ras[x_idx] = x
    dipole_pos_ras[y_idx] = y
    dipole_pos_ras[remaining_idx] = state['slice_coord'][remaining_idx]['val']

    state['dipole_pos'] = dipole_pos_ras
    update_dipole_pos(dipole_pos_ras=dipole_pos_ras,
                      ras_to_head_t=ras_to_head_t,
                      widget=widget, evoked=evoked)
    leave_set_dipole_pos_mode()


def handle_click_in_set_dipole_ori_mode(widget, state, x_idx, y_idx,
                                        remaining_idx, x, y, ras_to_head_t,
                                        evoked):
    dipole_ori_ras = dict()
    dipole_ori_ras[x_idx] = x
    dipole_ori_ras[y_idx] = y
    dipole_ori_ras[remaining_idx] = state['slice_coord'][remaining_idx]['val']

    state['dipole_ori'] = dipole_ori_ras
    update_dipole_ori(dipole_ori_ras=dipole_ori_ras,
                      ras_to_head_t=ras_to_head_t,
                      widget=widget, evoked=evoked)
    leave_set_dipole_ori_mode()
