import numpy as np
import mne


def gen_forward_solution(pos, bem, info, trans):
    ori = np.array([1.0, 1.0, 1.0]).reshape(1, 3)  # Arbitrary orientation.
    ori /= np.linalg.norm(ori)
    pos = pos.reshape(1, 3)
    gof = np.array([100]).reshape(1,)
    amplitude = np.array([10e-9]).reshape(1,)  # Arbitraty amplitude.
    dip = mne.Dipole(times=[0], pos=pos, ori=ori,
                     amplitude=amplitude, gof=gof)
    fwd, _ = mne.make_forward_dipole(dip, bem=bem, info=info, trans=trans)
    return fwd
