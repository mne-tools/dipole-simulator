import numpy as np
import nibabel as nib
from mne.transforms import combine_transforms, invert_transform, Transform


def gen_ras_to_head_trans(head_to_mri_t, t1_img):
    mri_to_head_t = invert_transform(head_to_mri_t)

    # RAS <> VOXEL
    ras_to_vox_t = Transform(fro='ras', to='mri_voxel',
                             trans=np.linalg.inv(t1_img.header.get_vox2ras()))
    #  trans=t1_img.header.get_ras2vox())
    vox_to_mri_t = Transform(fro='mri_voxel', to='mri',
                             trans=t1_img.header.get_vox2ras_tkr())

    # RAS <> MRI
    ras_to_mri_t = combine_transforms(ras_to_vox_t,
                                      vox_to_mri_t,
                                      fro='ras', to='mri')

    ras_to_head_t = combine_transforms(ras_to_mri_t,
                                       mri_to_head_t,
                                       fro='ras', to='head')

    return ras_to_head_t
