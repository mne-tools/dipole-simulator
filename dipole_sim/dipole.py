from mne.transforms import apply_trans
from evoked_field import reset_topomaps


def remove_dipole_arrows(widget):
    for axis, fig in widget['fig'].items():
        ax = widget['fig'][axis].axes[0]
        for child in ax.get_children():
            if child.get_label() == 'dipole_arrow':
                child.remove()

        fig.canvas.draw()


def draw_dipole_arrows(widget, state):
    remove_dipole_arrows(widget)
    for axis, fig in widget['fig'].items():
        ax = widget['fig'][axis].axes[0]

        if axis == 'x':
            x_idx = 'y'
            y_idx = 'z'
        elif axis == 'y':
            x_idx = 'x'
            y_idx = 'z'
        elif axis == 'z':
            x_idx = 'x'
            y_idx = 'y'

        x = state['dipole_pos'][x_idx]
        y = state['dipole_pos'][y_idx]
        dx = state['dipole_ori'][x_idx] - state['dipole_pos'][x_idx]
        dy = state['dipole_ori'][y_idx] - state['dipole_pos'][y_idx]

        ax.arrow(x=x, y=y, dx=dx, dy=dy, facecolor='white', edgecolor='black',
                 width=5, head_width=15, length_includes_head=True,
                 label='dipole_arrow')

        fig.canvas.draw()


def plot_dipole_pos_marker(widget, markers, state):
    remove_dipole_pos_markers(widget, markers, state)
    for axis in state['dipole_pos'].keys():
        if axis == 'x':
            x_idx = 'y'
            y_idx = 'z'
        elif axis == 'y':
            x_idx = 'x'
            y_idx = 'z'
        elif axis == 'z':
            x_idx = 'x'
            y_idx = 'y'

        x = state['dipole_pos'][x_idx]
        y = state['dipole_pos'][y_idx]

        ax = widget['fig'][axis].axes[0]
        markers['dipole_pos'][axis] = ax.scatter(x, y, marker='o', s=50,
                                                 facecolors='r',
                                                 edgecolors='r',
                                                 label='dipole_pos_marker')
        widget['fig'][axis].canvas.draw()
        # FIXME there must be a public function fore this?
        ax.figure.canvas._cursor = 'crosshair'


def remove_dipole_pos_markers(widget, markers, state):
    for axis in state['dipole_pos'].keys():
        ax = widget['fig'][axis].axes[0]
        if markers['dipole_pos'][axis] is not None:
            markers['dipole_pos'][axis].remove()
            markers['dipole_pos'][axis] = None
            widget['fig'][axis].canvas.draw()
        # FIXME there must be a public function fore this?
        ax.figure.canvas._cursor = 'crosshair'


def plot_dipole_ori_marker(widget, markers, state):
    remove_dipole_ori_markers(widget, markers, state)
    for axis in state['dipole_ori'].keys():
        if axis == 'x':
            x_idx = 'y'
            y_idx = 'z'
        elif axis == 'y':
            x_idx = 'x'
            y_idx = 'z'
        elif axis == 'z':
            x_idx = 'x'
            y_idx = 'y'

        x = state['dipole_ori'][x_idx]
        y = state['dipole_ori'][y_idx]

        ax = widget['fig'][axis].axes[0]
        markers['dipole_ori'][axis] = ax.scatter(x, y, marker='x', s=50,
                                                 facecolors='r',
                                                 label='dipole_ori_marker')
        widget['fig'][axis].canvas.draw()
        # FIXME there must be a public function fore this?
        ax.figure.canvas._cursor = 'crosshair'


def remove_dipole_ori_markers(widget, markers, state):
    for axis in state['dipole_ori'].keys():
        ax = widget['fig'][axis].axes[0]
        if markers['dipole_ori'][axis] is not None:
            markers['dipole_ori'][axis].remove()
            markers['dipole_ori'][axis] = None
            widget['fig'][axis].canvas.draw()
        # FIXME there must be a public function fore this?
        ax.figure.canvas._cursor = 'crosshair'


def update_dipole_pos(dipole_pos_ras, ras_to_head_t, widget, evoked):
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
    reset_topomaps(widget=widget, evoked=evoked)


def update_dipole_ori(dipole_ori_ras, ras_to_head_t, widget, evoked):
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
    reset_topomaps(widget=widget, evoked=evoked)


def draw_dipole_if_necessary(state, widget, markers):
    if state['dipole_pos']['x'] is not None:
        plot_dipole_pos_marker(widget, markers, state)

    if state['dipole_ori']['x'] is not None:
        plot_dipole_ori_marker(widget, markers, state)

    if (state['dipole_pos']['x'] is not None and
            state['dipole_ori']['x'] is not None and
            state['dipole_pos'] != state['dipole_ori']):
        draw_dipole_arrows(widget, state)
