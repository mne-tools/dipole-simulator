from ipywidgets import (Accordion, Label, Checkbox, Output, VBox, HBox,
                        ToggleButtons, IntSlider, Tab, Layout, Button,
                        Accordion, HTML, Dropdown, GridspecLayout)
import IPython.display
import pathlib
from matplotlib.backend_bases import MouseButton
import nibabel as nib
import numpy as np
import xarray as xr

from slice import create_slice_fig, plot_slice, get_axis_names_from_slice
from evoked_field import (create_topomap_fig, plot_sensors, plot_evoked,
                          reset_topomaps)
from cursor import enable_crosshair_cursor
from transforms import gen_ras_to_head_trans
from callbacks import (handle_click_in_slice_browser_mode,
                       handle_click_in_set_dipole_pos_mode,
                       handle_click_in_set_dipole_ori_mode)
from dipole import (plot_dipole_pos_marker, plot_dipole_ori_marker,
                    draw_dipole_arrows, remove_dipole_arrows,
                    remove_dipole_pos_markers, remove_dipole_ori_markers,
                    update_dipole_ori, update_dipole_pos,
                    draw_dipole_if_necessary)
from forward import load_fwd_lookup_table


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

        img, img_canonical, data_canonical_mm = self._init_mr_image(t1_img)
        self._t1_img = img
        self._t1_img_canonical = img_canonical
        self._t1_img_canonical_data = data_canonical_mm
        del img, img_canonical, data_canonical_mm

        self._subject = subject
        self._data_path = (pathlib.Path('data') if data_path is None
                           else pathlib.Path(data_path))
        self._fwd_path = self._data_path / 'fwd'
        self._subjects_dir = self._data_path / 'subjects'
        self._bem_path = self._data_path / f'{subject}-bem-sol.fif'

        self._fwd_lookup_table = load_fwd_lookup_table(fwd_path=self._fwd_path)

        self._exact_solution = False
        self._state = self._init_state()
        self._widget = self._init_widget()
        self._markers = self._init_markers()

        self._plot_slice(axis='all')

        self._plot_sensors()

        self._ras_to_head_t = gen_ras_to_head_trans(head_to_mri_t=self._trans,
                                                    t1_img=self._t1_img)

        self._preset_coords = {
            'Preset 1': dict(pos=[2.94, -76.54, -0.38],
                             ori=[1, 1, 1,]),
            'Preset 2': dict(pos=[-50.70, -23.55, 53.65],
                             ori=[0, 1, 0]),
            'Preset 3': dict(pos=[21.60, 80.03, 34.01],
                             ori=[1, 0, 0])}

        self._gen_app_layout()
        self._enable_crosshair_cursor()

    @staticmethod
    def _init_mr_image(img):
        img_canonical = nib.as_closest_canonical(img)      
        vox_grid = np.c_[np.arange(img_canonical.dataobj.shape[0]),
                         np.arange(img_canonical.dataobj.shape[1]),
                         np.arange(img_canonical.dataobj.shape[2])]

        coords_mm = nib.affines.apply_affine(img_canonical.affine,
                                             pts=vox_grid)
        data_canonical_mm = xr.DataArray(data=img_canonical.dataobj,
                                         dims=('x', 'y', 'z'),
                                         coords=(coords_mm[:, 0],
                                                 coords_mm[:, 1],
                                                 coords_mm[:, 2]))

        return img, img_canonical, data_canonical_mm

    def _init_state(self):
        state = dict()
        state['slice_coord'] = dict(x=dict(val=0, min=-60, max=60),
                                    y=dict(val=0, min=-70, max=70),
                                    z=dict(val=0, min=-20, max=60))
        state['crosshair_pos'] = dict(x=0, y=0, z=0)
        state['dipole_pos'] = dict(x=None, y=None, z=None)
        state['dipole_ori'] = dict(x=None, y=None, z=None)
        state['dipole_amplitude'] = 50e-9  # Am
        state['label_text'] = dict(
            x=(f'sagittal (x = {round(state["slice_coord"]["x"]["val"])} mm)'),
            y=(f'coronal (y = {round(state["slice_coord"]["y"]["val"])} mm)'),
            z=(f'axial (z = {round(state["slice_coord"]["z"]["val"])} mm)'),
            topomap_mag='Evoked magnetometer field',
            topomap_grad='Evoked gradiometer field',
            topomap_eeg='Evoked EEG field')
        state['dipole_arrows'] = []
        state['mode'] = 'slice_browser'
        state['updating'] = False
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
        label['axis'] = dict(x=HTML(f"<b>{state['label_text']['x']}</b>"),
                             y=HTML(f"<b>{state['label_text']['y']}</b>"),
                             z=HTML(f"<b>{state['label_text']['z']}</b>"))
        label['topomap_mag'] = HTML(f"<b>{state['label_text']['topomap_mag']}</b>")
        label['topomap_grad'] = HTML(f"<b>{state['label_text']['topomap_grad']}</b>")
        label['topomap_eeg'] = HTML(f"<b>{state['label_text']['topomap_eeg']}</b>")

        label['dipole_pos'] = Label('Not set')
        label['dipole_ori'] = Label('Not set')
        label['dipole_pos_'] = Label('Dipole origin:')
        label['dipole_ori_'] = Label('Dipole orientation:')
        label['status'] = Label('Status:')
        label['updating'] = Label('Ready.')
        widget['label'] = label
        widget['tab'] = Tab(layout=Layout(width='700'))

        toggle_buttons = dict(
            mode_selector=ToggleButtons(
                options=['Slice Browser', 'Set Dipole Origin',
                         'Set Dipole Orientation'],
                button_style='primary',
                layout=Layout(width='auto'))
            )
        toggle_buttons['mode_selector'].observe(self._handle_view_mode_change,
                                                'value')
        widget['toggle_buttons'] = toggle_buttons
        widget['reset_button'] = Button(description='Reset',
                                        button_style='danger',
                                        layout=Layout(width='auto'))
        widget['reset_button'].on_click(self._handle_reset_button_click)

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

        widget['quickstart_text'] = HTML(
            value=('<ul>'
                   '<li>Select the desired brain slices in the '
                   '<b>Slice Browser.</b></li>'
                   '<li>Choose the location of the dipole via '
                   '<b>Set Dipole Origin.</b></li>'
                   '<li>Orient the dipole via '
                   '<b>Set Dipole Orientation.</b></li>'
                   '<li>Adjust the <b>dipole amplitude</b> '
                   'using the slider below the topographic maps.</li>'
                   '</ul>'))
        widget['quickstart_accordion'] = Accordion(
            children=[widget['quickstart_text']])
        widget['quickstart_accordion'].set_title(0, 'Quickstart')

        widget['preset_dropdown'] = Dropdown(
            options=['Select Preset…', 'Preset 1', 'Preset 2', 'Preset 3'],
            value='Select Preset…',
            layout=Layout(width='auto'))
        widget['preset_dropdown'].observe(self._handle_preset_selection_change,
                                          'value')

        widget['title'] = HTML(value='<h2>Dipole Simulator</h2>')

        widget['output'] = output_widget
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

    def _toggle_updating_state(self):
        self._state['updating'] = not self._state['updating']

        if self._state['updating']:
            self._widget['label']['updating'].value = 'Updating …'
        else:
            self._widget['label']['updating'].value = 'Ready.'

    @output_widget.capture(clear_output=True)
    def _handle_slice_click(self, event):
        if event.button != MouseButton.LEFT:
            return

        self._toggle_updating_state()
        widget, markers, state = self._widget, self._markers, self._state
        in_ax = event.inaxes
        if in_ax is None:  # User clicked into the figure, but outside an axes
            self._toggle_updating_state()
            return

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
                                               self._t1_img_canonical_data)
        elif state['mode'] == 'set_dipole_pos':
            handle_click_in_set_dipole_pos_mode(widget, state, x_idx, y_idx,
                                                remaining_idx, x, y,
                                                self._ras_to_head_t,
                                                evoked=self._evoked)
        elif state['mode'] == 'set_dipole_ori':
            # Construct the 3D coordinates of the clicked-on point
            handle_click_in_set_dipole_ori_mode(widget, state, x_idx, y_idx,
                                                remaining_idx, x, y,
                                                self._ras_to_head_t,
                                                evoked=self._evoked)

        self._plot_dipole_markers_and_arrow()
        self._enable_crosshair_cursor()

        if (state['dipole_pos']['x'] is not None and
                state['dipole_ori']['x'] is not None and
                state['dipole_pos'] != state['dipole_ori']):
            plot_evoked(widget, state, fwd_path=self._fwd_path,
                        subject=self._subject, info=self._info,
                        ras_to_head_t=self._ras_to_head_t,
                        exact_solution=self._exact_solution,
                        bem_path=self._bem_path, head_to_mri_t=self._trans,
                        fwd_lookup_table=self._fwd_lookup_table,
                        t1_img=self._t1_img)

        self._toggle_updating_state()

    def _handle_slice_mouse_enter(self, event):
        pass

    def _handle_slice_mouse_leave(self, event):
        pass

    def _handle_amp_change(self, change):
        self._toggle_updating_state()

        state = self._state
        widget = self._widget

        widget['amplitude_slider'].disabled = True

        new_amp = change['new'] * 1e-9
        self._state['dipole_amplitude'] = new_amp

        if (state['dipole_pos']['x'] is not None and
                state['dipole_ori']['x'] is not None and
                state['dipole_pos'] != state['dipole_ori']):
            plot_evoked(widget, state, fwd_path=self._fwd_path,
                        subject=self._subject, info=self._info,
                        ras_to_head_t=self._ras_to_head_t,
                        exact_solution=self._exact_solution,
                        bem_path=self._bem_path, head_to_mri_t=self._trans,
                        fwd_lookup_table=self._fwd_lookup_table,
                        t1_img=self._t1_img)
        self._toggle_updating_state()
        widget['amplitude_slider'].disabled = False

    def _handle_reset_button_click(self, button):
        self._toggle_updating_state()
        widget = self._widget
        markers = self._markers
        state = self._state

        widget['preset_dropdown'].value = 'Select Preset…'
        remove_dipole_arrows(widget=widget)
        remove_dipole_pos_markers(widget=widget, markers=markers, state=state)
        remove_dipole_ori_markers(widget=widget, markers=markers, state=state)

        self._state = self._init_state()
        self._plot_slice(axis='all')
        reset_topomaps(widget=widget, evoked=self._evoked)
        widget['label']['dipole_pos'].value = 'Not set'
        widget['label']['dipole_ori'].value = 'Not set'
        widget['amplitude_slider'].value = (self
                                            ._state['dipole_amplitude'] * 1e9)
        self._enable_crosshair_cursor()
        self._toggle_updating_state()

    @output_widget.capture(clear_output=True)
    def _handle_preset_selection_change(self, change):
        self._toggle_updating_state()

        preset_name = change['new']
        if preset_name == 'Select Preset…':
            self._toggle_updating_state()
            return

        preset = self._preset_coords[preset_name]

        pos = np.array(preset['pos']).astype(float)
        ori = np.array(preset['ori']).astype(float)
        ori /= np.linalg.norm(ori)

        pos = dict(x=pos[0], y=pos[1], z=pos[2])
        ori = dict(x=ori[0], y=ori[1], z=ori[2])
        self._state['dipole_pos'] = pos
        self._state['dipole_ori'] = ori
        
        for plane, coord in pos.items():
            self._state['slice_coord'][plane]['val'] = coord

        state = self._state
        widget = self._widget

        update_dipole_pos(dipole_pos_ras=pos,
                          ras_to_head_t=self._ras_to_head_t,
                          widget=self._widget, evoked=self._evoked)
        update_dipole_ori(dipole_ori_ras=ori,
                          ras_to_head_t=self._ras_to_head_t,
                          widget=self._widget, evoked=self._evoked)
        self._plot_slice(axis='all')
        draw_dipole_if_necessary(state=self._state, widget=self._widget,
                                 markers=self._markers)

        if (state['dipole_pos']['x'] is not None and
                state['dipole_ori']['x'] is not None and
                state['dipole_pos'] != state['dipole_ori']):
            plot_evoked(widget, state, fwd_path=self._fwd_path,
                        subject=self._subject, info=self._info,
                        ras_to_head_t=self._ras_to_head_t,
                        exact_solution=self._exact_solution,
                        bem_path=self._bem_path, head_to_mri_t=self._trans,
                        fwd_lookup_table=self._fwd_lookup_table,
                        t1_img=self._t1_img)
        
        self._toggle_updating_state()

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
                       pos=pos, img_data=self._t1_img_canonical_data)

    def _gen_app_layout(self):
        title = self._widget['title']
        toggle_buttons = self._widget['toggle_buttons']
        # checkbox = self._widget['checkbox']
        label = self._widget['label']
        fig = self._widget['fig']
        topomap_fig = self._widget['topomap_fig']
        dipole_amp_slider = self._widget['amplitude_slider']
        tab = self._widget['tab']
        output = self._widget['output']
        reset_button = self._widget['reset_button']
        quickstart = self._widget['quickstart_accordion']
        preset = self._widget['preset_dropdown']

        dipole_props_col = VBox(
            [HBox([label['dipole_pos_'], label['dipole_pos']]),
             HBox([label['dipole_ori_'], label['dipole_ori']])])

        # dipole_amp_and_exact_sol_col = VBox(
        #     [label['amplitude_slider'],
        #      dipole_amp_slider,
        #      checkbox['exact_solution']])

        dipole_amp_col = VBox(
            [dipole_amp_slider,
             label['amplitude_slider']])

        grid = GridspecLayout(3, 3, grid_gap='0', width="95%")
        grid[0, 0] = HBox([label['status'], label['updating']])
        grid[1, 0] = preset
        grid[1, -1] = reset_button
        grid[2, :3] = HBox([toggle_buttons['mode_selector']],
                           layout=Layout(display='flex',
                                         flex_flow='column',
                                         align_items='center',
                                         width='100%'))

        main_tab = VBox([grid,
                         #   dipole_amp_and_exact_sol_col]),
                         HBox([VBox([label['axis']['x'], fig['x'].canvas],
                                    layout=Layout(align_items='center')),
                               VBox([label['axis']['y'], fig['y'].canvas],
                                    layout=Layout(align_items='center')),
                               VBox([label['axis']['z'], fig['z'].canvas],
                                    layout=Layout(align_items='center'))]),
                         HBox([VBox([label['topomap_mag'],
                                    topomap_fig['mag'].canvas],
                                    layout=Layout(align_items='center')),
                               VBox([label['topomap_grad'],
                                    topomap_fig['grad'].canvas],
                                    layout=Layout(align_items='center')),
                               VBox([label['topomap_eeg'],
                                    topomap_fig['eeg'].canvas],
                                    layout=Layout(align_items='center'))]),
                         dipole_amp_col,
                         dipole_props_col],
                        layout=Layout(align_items='center'))

        mne_output_tab = VBox([output])
        help_tab = VBox([Label('Whoops. Somebody was lazy here.')])

        about_text = ('<h3>M/EEG Dipole Simulator</h3>\n'
                      '<p>An interactive demonstration of how dipole location '
                      'and orientation affect MEG and EEG sensor signals.'
                      '</p>\n'
                      '<p><b>This is still '
                      '<a href="https://github.com/hoechenberger/dipoles_demo/issues/26">'
                      'work in progress</a></b></p>\n'
                      '<h4>Contributors</h4>\n'
                      '<ul>\n'
                      '<li>Idea & Conceptualization: '
                      '<b>Maximilien Chaumon</b> '
                      '(<a href="https://github.com/dnacombo">GitHub</a>, '
                      '<a href="https://twitter.com/@cuttingEEG">Twitter)</a>'
                      '</li>\n'
                      '<li>Conceptualization & Supervision: '
                      '<b>Alexandre Gramfort</b> '
                      '(<a href="https://github.com/agramfort">GitHub</a>, '
                      '<a href="https://twitter.com/agramfort">Twitter</a>)'
                      '</li>\n'
                      '<li>Testing & Feedback: '
                      '<b>Sophie Herbst</b> '
                      '(<a href="https://github.com/SophieHerbst">GitHub</a>, '
                      '<a href="https://twitter.com/herbstso">Twitter</a>) & '
                      '<b>Virginie van Wassenhove</b> '
                      '(<a href="https://github.com/virvw">GitHub</a>, '
                      '<a href="https://twitter.com/virginie_vw">Twitter</a>)'
                      '</li>\n'
                      '<li>Implementation: '
                      '<b>Richard Höchenberger</b> '
                      '(<a href="https://github.com/hoechenberger">GitHub</a>, '
                      '<a href="https://twitter.com/RHoechenberger">Twitter</a>)'
                      '</li>\n'
                      '</ul>\n')

        about_tab = VBox([HTML(about_text)])
        del about_text

        tab.children = [main_tab, mne_output_tab, help_tab, about_tab]
        tab.set_title(0, 'Dipole Simulator')
        tab.set_title(1, 'MNE-Python Output')
        tab.set_title(2, 'Help')
        tab.set_title(3, 'About')

        app = VBox([title, quickstart, tab], layout=Layout(align_items='center'))

        self._app_layout = app

    def display(self):
        IPython.display.display(self._app_layout)


if __name__ == '__main__':
    import mne

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
    t1_img = nib.load(t1_fname)
    del t1_fname

    trans_fname = data_path / 'sample-trans.fif'
    head_to_mri_t = mne.read_trans(trans_fname)

    app = App(evoked=evoked,
              trans=head_to_mri_t,
              t1_img=t1_img)
    app.display()
