def remove_dipole_arrows(widget):
    for axis, fig in widget['fig'].items():
        ax = widget['fig'][axis].axes[0]
        artists_to_keep = [artist for artist in ax.artists
                           if artist.get_label() != 'dipole']
        ax.artists = artists_to_keep
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

        ax.arrow(x=x, y=y, dx=dx, dy=dy, color='white',
                 width=3, head_width=15, length_includes_head=True,
                 label='dipole')

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
                                                 edgecolors='r',
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
