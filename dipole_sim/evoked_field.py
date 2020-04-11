import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.transforms import apply_trans

from utils import create_head_grid, find_closest
from download_fwd import download_from_github


def _update_topomap_label(widget, state, ch_type):
    label = widget['label']['topomap_' + ch_type]
    label_text = state['label_text']['topomap_' + ch_type]
    label.value = label_text


def gen_evoked(pos, ori, info, fwd):
    leadfield = fwd['sol']['data']
    meeg_data = np.dot(leadfield, ori.T)  # compute forward
    evoked = mne.EvokedArray(meeg_data, info)
    return evoked


def plot_evoked(widget, state, fwd_path, subject, info, ras_to_head_t):
    old_topomap_mag_label_text = state['label_text']['topomap_mag']
    new_topomap_mag_label_text = old_topomap_mag_label_text + ' [updating]'
    state['label_text']['topomap_mag'] = new_topomap_mag_label_text

    old_topomap_grad_label_text = state['label_text']['topomap_grad']
    new_topomap_grad_label_text = old_topomap_grad_label_text + ' [updating]'
    state['label_text']['topomap_grad'] = new_topomap_grad_label_text

    old_topomap_eeg_label_text = state['label_text']['topomap_eeg']
    new_topomap_eeg_label_text = old_topomap_eeg_label_text + ' [updating]'
    state['label_text']['topomap_eeg'] = new_topomap_eeg_label_text

    for ch_type in ['mag', 'grad', 'eeg']:
        _update_topomap_label(widget, state, ch_type)

    dipole_pos = (state['dipole_pos']['x'],
                  state['dipole_pos']['y'],
                  state['dipole_pos']['z'])
    dipole_ori = (state['dipole_ori']['x'],
                  state['dipole_ori']['y'],
                  state['dipole_ori']['z'])

    dipole_pos = apply_trans(trans=ras_to_head_t, pts=dipole_pos)
    dipole_pos /= 1000

    dipole_ori = apply_trans(trans=ras_to_head_t, pts=dipole_ori)
    dipole_ori /= 1000
    dipole_ori /= np.linalg.norm(dipole_ori)

    dipole_pos = np.array(dipole_pos).reshape(1, 3).round(3)
    dipole_ori = np.array(dipole_ori).reshape(1, 3).round(3)

#     fwd = gen_forward_solution(dipole_pos, bem=bem, info=info,
# trans=head_to_mri_t)

    # Retrieve the dipole pos closest to the one we have a pre-calculated fwd
    # for.
    pos_head_grid = create_head_grid(info=info)
    dipole_pos_for_fwd = (find_closest(pos_head_grid[0], dipole_pos[0, 0]),
                          find_closest(pos_head_grid[1], dipole_pos[0, 1]),
                          find_closest(pos_head_grid[2], dipole_pos[0, 2]))

    print(f'Requested calculations for dipole located at:\n'
          f'    x={dipole_pos[0, 0]}, y={dipole_pos[0, 1]}, '
          f'z={dipole_pos[0, 2]} [m, MNE Head]\n'
          f'Using a forward solution for the following location:\n'
          f'    x={dipole_pos_for_fwd[0]}, y={dipole_pos_for_fwd[1]}, '
          f'z={dipole_pos_for_fwd[2]} [m, MNE Head]\n')

    fwd_fname = (f'{subject}-{dipole_pos_for_fwd[0]}-{dipole_pos_for_fwd[1]}-'
                 f'{dipole_pos_for_fwd[2]}-fwd.fif')
    if (fwd_path / fwd_fname).exists():
        print(f'\nUsing existing forward solution: {fwd_fname}')
    else:
        print('Retrieving forward solution from GitHub.')
        try:
            download_from_github(fwd_path=fwd_path, subject=subject,
                                 dipole_pos=dipole_pos_for_fwd)
        except RuntimeError as e:
            msg = (f'Failed to retrieve pre-calculated forward solution. '
                   f'The error was: {e}\n\n'
                   f'Please try again with another dipole origin inside the '
                   f'brain.')
            raise RuntimeError(msg)

    fwd = mne.read_forward_solution(fwd_path / fwd_fname)
    del fwd_fname, pos_head_grid, dipole_pos_for_fwd

    fwd = mne.forward.convert_forward_solution(fwd=fwd, force_fixed=True)
    evoked = gen_evoked(pos=dipole_pos, ori=dipole_ori, info=info, fwd=fwd)

    for ch_type, fig in widget['topomap_fig'].items():
        ax_topomap = fig.axes[0]
        ax_colorbar = fig.axes[1]

        ax_topomap.clear()
        ax_colorbar.clear()

        if ch_type == 'eeg':
            outlines = 'head'
        else:
            outlines = 'skirt'

        evoked.plot_topomap(ch_type=ch_type,
                            colorbar=False,
                            outlines=outlines,
                            contours=0,
                            times=evoked.times[-1],
                            res=256,
                            show=False,
                            axes=ax_topomap)
        ax_topomap.set_title(None)
#             ax_topomap.format_coord = _create_format_coord('topomap')
        cb = fig.colorbar(ax_topomap.images[-1], cax=ax_colorbar,
                          orientation='horizontal')

        if ch_type == 'mag':
            label = 'fT'
        elif ch_type == 'grad':
            label = 'fT/cm'
        elif ch_type == 'eeg':
            label = 'µV'

        cb.set_label(label, fontweight='bold')
        fig.canvas.draw()

    state['label_text']['topomap_mag'] = old_topomap_mag_label_text
    state['label_text']['topomap_grad'] = old_topomap_grad_label_text
    state['label_text']['topomap_eeg'] = old_topomap_eeg_label_text

    for ch_type in ['mag', 'grad', 'eeg']:
        _update_topomap_label(widget, state, ch_type)


def create_topomap_fig():
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 0.1]},
                           figsize=(2, 2.5))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.resizable = False
    fig.canvas.callbacks.callbacks.clear()
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    fig.set_tight_layout(True)
    return fig


def reset_topomaps(widget, evoked):
    for ch_type in ['mag', 'grad', 'eeg']:
        widget['topomap_fig'][ch_type].axes[0].clear()
        plot_sensors(widget=widget, evoked=evoked, ch_type=ch_type)
        widget['topomap_fig'][ch_type].canvas.draw()


def plot_sensors(widget, evoked, ch_type):
    ax = widget['topomap_fig'][ch_type].axes[0]
    evoked.plot_sensors(ch_type=ch_type,
                        title='',
                        show=False,
                        axes=ax)
    ax.collections[0].set_sizes([0.05])
    ax.figure.canvas.draw()
