"""
Step 1 of 3 for the Orbit Explorer: download the data.

Everything comes from NASA/JPL's Solar System Dynamics group and is public:

  * Sentry risk list         https://ssd-api.jpl.nasa.gov/doc/sentry.html
      - summary of every object on the list (one row per asteroid)
      - every "virtual impactor" (each possible impact date + its probability)
      - objects that have been removed from the list, with the removal date
  * Small-Body Database (SBDB) Query API: orbits of all near-Earth objects
                             https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html
  * Close-Approach Data (CAD) API: JPL's own predicted close passes of Earth
                             https://ssd-api.jpl.nasa.gov/doc/cad.html
  * DE440s planetary ephemeris: precise positions of Sun, planets and Moon
                             https://ssd.jpl.nasa.gov/planets/eph_export.html

Files are saved in data/07_explorer/ (git-ignored) and only downloaded once.
Run from the repository root:   python3 interactive/07_orbit_explorer/step1_download.py
"""
from pathlib import Path
import json
import requests

DATA = Path('data/07_explorer')
DATA.mkdir(parents=True, exist_ok=True)

API = 'https://ssd-api.jpl.nasa.gov'

# The two removed objects we feature in the explorer (they are no longer on Sentry's list)
FEATURED = {'2024 YR4': '2024 YR4', '99942': '99942 Apophis'}

DOWNLOADS = {
    # file name                     : (url, query parameters)
    'sentry_summary.json':      (f'{API}/sentry.api', {}),
    'sentry_vi_all.json':       (f'{API}/sentry.api', {'all': 1}),
    'sentry_removed.json':      (f'{API}/sentry.api', {'removed': 1}),
    'sbdb_neo_elements.json':   (f'{API}/sbdb_query.api', {
        'fields': 'pdes,full_name,epoch,e,a,i,om,w,ma,H,diameter,condition_code,data_arc',
        'sb-group': 'neo',
        'full-prec': 'true',            # without this, orbits are rounded to 4 digits
    }),
    'cad_earth_2000_2130.json': (f'{API}/cad.api', {
        'date-min': '2000-01-01', 'date-max': '2130-12-31', 'dist-max': '0.05'}),
}
for des in FEATURED:
    for body in ['Earth', 'Moon']:
        name = f"cad_{des.replace(' ', '_')}_{body.lower()}.json"
        DOWNLOADS[name] = (f'{API}/cad.api', {'des': des, 'body': body,
                                              'date-min': '2000-01-01', 'date-max': '2130-12-31',
                                              'dist-max': '0.05'})


def download_json(name, url, params):
    path = DATA / name
    if path.exists():
        print(f'  already have {path}')
        return
    print(f'  downloading {name} ...')
    r = requests.get(url, params=params, timeout=300)
    r.raise_for_status()                      # stop with a clear error if it failed
    reply = r.json()
    if 'error' in reply or 'message' in reply and 'data' not in reply:
        raise RuntimeError(f'{url} returned an error: {reply}')
    path.write_text(json.dumps(reply))
    print(f'    saved {path} ({path.stat().st_size / 1e6:.1f} MB)')


def download_file(name, url):
    path = DATA / name
    if path.exists():
        print(f'  already have {path}')
        return
    print(f'  downloading {name} (about 32 MB) ...')
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f'    saved {path} ({path.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    print('Downloading JPL data into', DATA)
    for name, (url, params) in DOWNLOADS.items():
        download_json(name, url, params)
    download_file('de440s.bsp', 'https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440s.bsp')
    print('Done.')
