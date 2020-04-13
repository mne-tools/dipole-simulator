import numpy as np
import mne


def gen_forward_solution(pos, bem, info, trans, verbose=True):
    # NOTE:
    # Both, dipole amplitude and orientation, are set to arbitrary values
    # here: They are merely required to instantiate a Dipole object, which
    # we then use to conveniently retrieve a forward solution via
    # `make_forward_dipole()`. This forward solution is returned in "fixed"
    # orientation. We then convert it back to "free" orientation mode before
    # returning the forward object.

    amplitude = np.array([1]).reshape(1,)  # Arbitraty amplitude.
    ori = np.array([1., 1., 1.]).reshape(1, 3)  # Arbitrary orientation.
    ori /= np.linalg.norm(ori)
    pos = pos.reshape(1, 3)
    gof = np.array([100]).reshape(1,)
    dip = mne.Dipole(times=[0], pos=pos, ori=ori,
                     amplitude=amplitude, gof=gof)
    fwd, _ = mne.make_forward_dipole(dip, bem=bem, info=info, trans=trans,
                                     verbose=verbose)
    fwd = mne.convert_forward_solution(fwd, force_fixed=False, verbose=verbose)
    return fwd
