import numpy as np
import nibabel as nib
from mne.transforms import combine_transforms, invert_transform, Transform


def gen_ras_to_head_trans(head_to_mri_t, t1_img):
    # RAS -> VOXEL
    ras_to_vox_t = Transform(fro='ras', to='mri_voxel',
                             trans=np.linalg.inv(t1_img.header.get_vox2ras()))

    # VOXEL -> MRI
    vox_to_mri_t = Transform(fro='mri_voxel', to='mri',
                             trans=t1_img.header.get_vox2ras_tkr())

    # MRI -> HEAD
    mri_to_head_t = invert_transform(head_to_mri_t)
    
    # Now we have generated all the required transformations
    # to go from RAS to MNE Head coordinates. Let's combine
    # the transforms into a single transform. This requires
    # two calls to `combine_transforms()`.

    # RAS -> MRI
    ras_to_mri_t = combine_transforms(ras_to_vox_t,
                                      vox_to_mri_t,
                                      fro='ras', to='mri')

    # RAS -> HEAD
    ras_to_head_t = combine_transforms(ras_to_mri_t,
                                       mri_to_head_t,
                                       fro='ras', to='head')

    return ras_to_head_t
