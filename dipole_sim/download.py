import requests


def download_fwd_from_zenodo(fwd_path, subject, dipole_pos,
                             zenodo_files, overwrite=False):
    x, y, z = dipole_pos
    fname = f'{subject}-{x}-{y}-{z}-fwd.fif'

    if (fwd_path / fname).exists() and not overwrite:
        return

    for file in zenodo_files:
        if file['key'] == fname:
            fwd_url = file['links']['self']
            fwd_request = requests.get(fwd_url)
            with open(fwd_path / fname, 'wb') as f:
                f.write(fwd_request.content)

            return

    raise RuntimeError('Could not find the requested forward solution online!')


def download_fwd_from_github(fwd_path, subject, dipole_pos, overwrite=False):
    x, y, z = dipole_pos
    fname = f'{subject}-{x}-{y}-{z}-fwd.fif'

    if (fwd_path / fname).exists() and not overwrite:
        return

    fwd_url = (f'https://github.com/hoechenberger/dipoles_demo_data/'
               f'raw/master/data/fwd/{fname}')
    fwd_request = requests.get(fwd_url, allow_redirects=True)
    if fwd_request.status_code != 200:
        msg = (f'Could not download the requested forward solution from '
               f'GitHub!\nDownload URL was: {fwd_url}')
        raise RuntimeError(msg)

    with open(fwd_path / fname, 'wb') as f:
        f.write(fwd_request.content)


def download_bem_from_github(data_path, subject, overwrite=False):
    fname = f'{subject}-bem-sol.fif'

    if (data_path / fname).exists() and not overwrite:
        return

    bem_url = (f'https://github.com/hoechenberger/dipoles_demo_data/'
               f'raw/master/data/{fname}')
    bem_request = requests.get(bem_url, allow_redirects=True)
    if bem_request.status_code != 200:
        msg = (f'Could not download the requested BEM solution from '
               f'GitHub!\nDownload URL was: {bem_url}')
        raise RuntimeError(msg)

    with open(data_path / fname, 'wb') as f:
        f.write(bem_request.content)
