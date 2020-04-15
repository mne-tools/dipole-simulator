from ipywidgets import (Accordion, Label, Checkbox, Output, VBox, HBox,
                        ToggleButtons, IntSlider)
import IPython.display
import pathlib
from matplotlib.backend_bases import MouseButton

from slice import create_slice_fig, plot_slice, get_axis_names_from_slice
from evoked_field import create_topomap_fig, plot_sensors, plot_evoked
from cursor import enable_crosshair_cursor
from transforms import gen_ras_to_head_trans
from callbacks import (handle_click_in_slice_browser_mode,
                       handle_click_in_set_dipole_pos_mode,
                       handle_click_in_set_dipole_ori_mode)
from dipole import (plot_dipole_pos_marker, plot_dipole_ori_marker,
                    draw_dipole_arrows)


# This widget will capture the MNE output.
# Create it here so we can use it as a function decorator.
output_widget = Output()


class App:
    def __init__(self,
                 evoked,
                 info=None,
                 trans=None,
                 t1_img=None,
                 subject='sample',
                 data_path='data'):
        self._evoked = evoked
        self._info = evoked.info if info is None else info
        self._trans = trans
        self._t1_img = t1_img
        self._subject = subject
        self._data_path = (pathlib.Path('data') if data_path is None
                           else pathlib.Path(data_path))
        self._fwd_path = self._data_path / 'fwd'
        self._subjects_dir = self._data_path / 'subjects'
        self._bem_path = self._data_path / f'{subject}-bem-sol.fif'

        self._exact_solution = False
        self._state = self._init_state()
        self._widget = self._init_widget()
        self._markers = self._init_markers()

        self._plot_slice(axis='all')

        self._plot_sensors()
        self._enable_crosshair_cursor()

        self._ras_to_head_t = gen_ras_to_head_trans(head_to_mri_t=self._trans,
                                                    t1_img=self._t1_img)

        self._gen_app_layout()

    def _init_state(self):
        state = dict()
        state['slice_coord'] = dict(x=dict(val=0, min=-60, max=60),
                                    y=dict(val=0, min=-70, max=70),
                                    z=dict(val=0, min=-20, max=60))
        state['crosshair_pos'] = dict(x=0, y=0, z=0)
        state['dipole_pos'] = dict(x=None, y=None, z=None)
        state['dipole_ori'] = dict(x=None, y=None, z=None)
        state['dipole_amplitude'] = 50e-9  # Am
        state['label_text'] = dict(x='sagittal',
                                   y='coronal',
                                   z='axial',
                                   topomap_mag='Evoked magnetometer field',
                                   topomap_grad='Evoked gradiometer field',
                                   topomap_eeg='Evoked EEG field')
        state['dipole_arrows'] = []
        state['mode'] = 'slice_browser'
        return state

    def _toggle_exact_solution(self, change):
        self._exact_solution = not self._exact_solution

    def _init_widget(self):
        state = self._state
        widget = dict()
        fig = dict(x=self._create_slice_fig(),
                   y=self._create_slice_fig(),
                   z=self._create_slice_fig())
        widget['fig'] = fig

        topomap_fig = dict(mag=create_topomap_fig(),
                           grad=create_topomap_fig(),
                           eeg=create_topomap_fig())
        widget['topomap_fig'] = topomap_fig

        label = dict()
        label['axis'] = dict(x=Label(state['label_text']['x']),
                             y=Label(state['label_text']['y']),
                             z=Label(state['label_text']['z']))
        label['topomap_mag'] = Label(state['label_text']['topomap_mag'])
        label['topomap_grad'] = Label(state['label_text']['topomap_grad'])
        label['topomap_eeg'] = Label(state['label_text']['topomap_eeg'])
        label['dipole_pos'] = Label('Not set')
        label['dipole_ori'] = Label('Not set')
        label['dipole_pos_'] = Label('Dipole origin:')
        label['dipole_ori_'] = Label('Dipole orientation:')
        widget['label'] = label

        toggle_buttons = dict(mode_selector=ToggleButtons(
            options=['Slice Browser', 'Set Dipole Origin',
                     'Set Dipole Orientation']))
        toggle_buttons['mode_selector'].observe(self._handle_view_mode_change,
                                                'value')
        widget['toggle_buttons'] = toggle_buttons

        checkbox = dict(exact_solution=Checkbox(
            value=self._exact_solution,
            description='Exact solution (slow!)',
            tooltip='Calculate an exact forward projection. This is SLOW!'))
        checkbox['exact_solution'].observe(self._toggle_exact_solution,
                                           'value')
        widget['checkbox'] = checkbox

        widget['amplitude_slider'] = IntSlider(
            value=int(self._state['dipole_amplitude'] * 1e9),
            min=5, max=100, step=5, continuous_update=False)
        widget['amplitude_slider'].observe(self._handle_amp_change,
                                           names='value')
        widget['label']['amplitude_slider'] = Label('Dipole amplitude in nAm')

        widget['output'] = output_widget
        accordion = Accordion(
            children=[widget['output'],
                      Label("now this isn't too helpful now, is it")],
            titles=('MNE Output', 'Help'))
        accordion.set_title(0, 'MNE Output')
        accordion.set_title(1, 'Help')
        widget['accordion'] = accordion

        return widget

    def _init_markers(self):
        markers = dict()
        markers['dipole_pos'] = dict(x=None, y=None, z=None)
        markers['dipole_ori'] = dict(x=None, y=None, z=None)
        return markers

    def _create_slice_fig(self):
        fig = create_slice_fig(handle_click=self._handle_slice_click,
                               handle_enter=self._handle_slice_mouse_enter,
                               handle_leave=self._handle_slice_mouse_leave)
        return fig

    @output_widget.capture(clear_output=True)
    def _handle_slice_click(self, event):
        if event.button != MouseButton.LEFT:
            return

        widget, markers, state = self._widget, self._markers, self._state
        in_ax = event.inaxes
        x, y = event.xdata, event.ydata

        # Which slice (axis) was clicked in?
        for axis, fig in widget['fig'].items():
            if fig is in_ax.figure:
                break

        x_idx, y_idx = get_axis_names_from_slice(slice_view=axis,
                                                 all_axes=widget['fig'].keys())
        remaining_idx = axis

        if state['mode'] == 'slice_browser':
            handle_click_in_slice_browser_mode(widget, markers, state, x, y,
                                               x_idx, y_idx, self._evoked,
                                               self._t1_img)
        elif state['mode'] == 'set_dipole_pos':
            handle_click_in_set_dipole_pos_mode(widget, state, x_idx, y_idx,
                                                remaining_idx, x, y,
                                                self._ras_to_head_t)
        elif state['mode'] == 'set_dipole_ori':
            # Construct the 3D coordinates of the clicked-on point
            handle_click_in_set_dipole_ori_mode(widget, state, x_idx, y_idx,
                                                remaining_idx, x, y,
                                                self._ras_to_head_t)

        self._plot_dipole_markers_and_arrow()
        self._enable_crosshair_cursor()

        if (state['dipole_pos']['x'] is not None and
                state['dipole_ori']['x'] is not None and
                state['dipole_pos'] != state['dipole_ori']):
            plot_evoked(widget, state, fwd_path=self._fwd_path,
                        subject=self._subject, info=self._info,
                        ras_to_head_t=self._ras_to_head_t,
                        exact_solution=self._exact_solution,
                        bem_path=self._bem_path, head_to_mri_t=self._trans)

    def _handle_slice_mouse_enter(self, event):
        pass

    def _handle_slice_mouse_leave(self, event):
        pass

    def _handle_amp_change(self, change):
        state = self._state
        widget = self._widget

        new_amp = change['new'] * 1e-9
        self._state['dipole_amplitude'] = new_amp

        if (state['dipole_pos']['x'] is not None and
                state['dipole_ori']['x'] is not None and
                state['dipole_pos'] != state['dipole_ori']):
            plot_evoked(widget, state, fwd_path=self._fwd_path,
                        subject=self._subject, info=self._info,
                        ras_to_head_t=self._ras_to_head_t,
                        exact_solution=self._exact_solution,
                        bem_path=self._bem_path, head_to_mri_t=self._trans)

    def _plot_dipole_markers_and_arrow(self):
        state = self._state
        widget = self._widget
        markers = self._markers

        if state['dipole_pos']['x'] is not None:
            plot_dipole_pos_marker(widget, markers, state)

        if state['dipole_ori']['x'] is not None:
            plot_dipole_ori_marker(widget, markers, state)

        if (state['dipole_pos']['x'] is not None and
                state['dipole_ori']['x'] is not None and
                state['dipole_pos'] != state['dipole_ori']):
            draw_dipole_arrows(widget, state)

    def _set_view_mode(self, new_mode):
        state = self._state

        if new_mode == 'Slice Browser':
            state['mode'] = 'slice_browser'
        elif new_mode == 'Set Dipole Origin':
            state['mode'] = 'set_dipole_pos'
        elif new_mode == 'Set Dipole Orientation':
            state['mode'] = 'set_dipole_ori'

    def _handle_view_mode_change(self, change):
        new_mode = change['new']
        self._set_view_mode(new_mode)

    def _enable_crosshair_cursor(self):
        enable_crosshair_cursor(self._widget)

    def _plot_sensors(self, ch_type=None):
        if ch_type is None:
            ch_types = ('mag', 'grad', 'eeg')
        else:
            ch_types = (ch_type,)

        for ch_type in ch_types:
            plot_sensors(widget=self._widget, evoked=self._evoked,
                         ch_type=ch_type)

    def _plot_slice(self, axis):
        if axis == 'all':
            axes = ('x', 'y', 'z')
        else:
            axes = (axis,)

        for axis in axes:
            pos = self._state['slice_coord'][axis]['val']
            plot_slice(widget=self._widget, state=self._state, axis=axis,
                       pos=pos, t1_img=self._t1_img)

    def _gen_app_layout(self):
        toggle_buttons = self._widget['toggle_buttons']
        checkbox = self._widget['checkbox']
        label = self._widget['label']
        fig = self._widget['fig']
        topomap_fig = self._widget['topomap_fig']
        accordion = self._widget['accordion']
        dipole_amp_slider = self._widget['amplitude_slider']

        dipole_props_col = VBox(
            [HBox([label['dipole_pos_'], label['dipole_pos']]),
             HBox([label['dipole_ori_'], label['dipole_ori']])])

        dipole_amp_and_exact_sol_col = VBox(
            [label['amplitude_slider'],
             dipole_amp_slider,
             checkbox['exact_solution']])

        app = VBox([HBox([toggle_buttons['mode_selector'],
                          dipole_amp_and_exact_sol_col]),
                    HBox([VBox([label['axis']['x'], fig['x'].canvas]),
                          VBox([label['axis']['y'], fig['y'].canvas]),
                          VBox([label['axis']['z'], fig['z'].canvas])]),
                    HBox([VBox([label['topomap_mag'],
                                topomap_fig['mag'].canvas]),
                          VBox([label['topomap_grad'],
                                topomap_fig['grad'].canvas]),
                          VBox([label['topomap_eeg'],
                                topomap_fig['eeg'].canvas])]),
                    dipole_props_col,
                    accordion])

        self._app_layout = app

    def display(self):
        IPython.display.display(self._app_layout)


if __name__ == '__main__':
    import mne
    import nilearn

    data_path = pathlib.Path('data')
    fwd_path = data_path / 'fwd'
    subjects_dir = data_path / 'subjects'
    subject = 'sample'

    evoked_fname = data_path / 'sample-ave.fif'
    evoked = mne.read_evokeds(evoked_fname, verbose='warning')[0]
    evoked.pick_types(meg=True, eeg=True)

    info = evoked.info
    info['projs'] = []
    info['bads'] = []
    del evoked_fname

    t1_fname = str(subjects_dir / subject / 'mri' / 'T1.mgz')
    t1_img = nilearn.image.load_img(t1_fname)
    del t1_fname

    trans_fname = data_path / 'sample-trans.fif'
    head_to_mri_t = mne.read_trans(trans_fname)

    app = App(evoked=evoked,
              trans=head_to_mri_t,
              t1_img=t1_img)
    app.display()
