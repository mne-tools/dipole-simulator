import numpy as np
import mne


def gen_forward_solution(widget, pos, bem, info, trans):
    ori = np.eye(3)
    pos = np.tile(pos, (3, 1))
    dip = mne.Dipole(times=np.arange(3), pos=pos, amplitude=3 * [10e-9],
                     ori=ori, gof=3 * [100])
    with widget['output']:
        fwd, _ = mne.make_forward_dipole(dip, bem=bem, info=info, trans=trans)

    return fwd
