"""
Step 2b: swap in JPL's own path for the few asteroids our simulation can't follow.

Step 3 compares our simulation with every close approach JPL predicts. Almost all
match within a few percent. A few dozen objects don't, for two reasons:

  * After a very close flyby (Apophis in 2029, for example), a tiny difference before
    the flyby grows into a completely different path afterwards. JPL's orbit model
    includes effects ours leaves out, such as the faint push of sunlight on the
    asteroid (the Yarkovsky effect) and the exact shape of Earth's gravity.
  * Temporary "mini-moons" (2006 RH120, 2020 CD3) that orbited Earth for months.

For those objects (and for the two featured objects, 2024 YR4 and Apophis), we
download JPL Horizons' positions instead and turn them into the same kind of
Kepler segments. We pick an object when our distance is off by more than 10% for
some close approach AND falls outside JPL's own uncertainty range for it.

JPL Horizons API: https://ssd-api.jpl.nasa.gov/doc/horizons.html

Run from the repository root after step 2:
    python3 interactive/07_orbit_explorer/step2b_jpl_tracks.py
Input:  data/07_explorer/segments_sim.npz       Output: data/07_explorer/segments.npz
"""
from pathlib import Path
import json
import sys
import numpy as np
import requests
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).parent))
from step2_compute_orbits import (DATA, AU_KM, LD_KM, JD0, FEATURED, TOLERANCE, TOL_FLOOR,
                                  state_to_elements, kepler_position, as_float32, MU_SUN)
import step3_build_page as s3

HORIZONS = 'https://ssd.jpl.nasa.gov/api/horizons.api'
CACHE = DATA / 'horizons'
CACHE.mkdir(exist_ok=True)


def horizons_vectors(des, jd_start, jd_stop, step):
    """Heliocentric ecliptic-J2000 position (AU) and velocity (AU/day) from JPL Horizons."""
    name = f"{des.replace(' ', '_')}_{jd_start:.4f}_{jd_stop:.4f}_{step.replace(' ', '')}.json"
    path = CACHE / name
    if not path.exists():
        command = f"'{des};'" if des.isdigit() else f"'DES={des};'"   # numbered vs provisional
        params = {
            'format': 'json', 'COMMAND': command, 'OBJ_DATA': 'NO', 'MAKE_EPHEM': 'YES',
            'EPHEM_TYPE': 'VECTORS', 'CENTER': "'500@10'", 'REF_PLANE': 'ECLIPTIC',
            'REF_SYSTEM': 'ICRF', 'OUT_UNITS': 'AU-D', 'VEC_TABLE': '2', 'CSV_FORMAT': 'YES',
            'VEC_LABELS': 'NO', 'TIME_TYPE': 'TDB',
            'START_TIME': f"'{Time(jd_start, format='jd', scale='tdb').iso[:16]}'",
            'STOP_TIME': f"'{Time(jd_stop, format='jd', scale='tdb').iso[:16]}'",
            'STEP_SIZE': f"'{step}'",
        }
        r = requests.get(HORIZONS, params=params, timeout=300)
        r.raise_for_status()
        result = r.json().get('result', '')
        if '$$SOE' not in result:
            raise RuntimeError(f'Horizons gave no data for {des}:\n{result[:800]}')
        path.write_text(json.dumps({'result': result}))
    result = json.loads(path.read_text())['result']
    rows = result.split('$$SOE')[1].split('$$EOE')[0].strip().splitlines()
    values = np.array([[float(x) for x in (row.split(',')[0], *row.split(',')[2:8])] for row in rows])
    return values[:, 0], values[:, 1:4], values[:, 4:7]


def windows(jd, dist, limit, pad):
    """Time ranges where dist < limit, padded and merged."""
    near = jd[dist < limit]
    spans = []
    for t in near:
        if spans and t - pad <= spans[-1][1]:
            spans[-1][1] = t + pad
        else:
            spans.append([t - pad, t + pad])
    return spans


def jpl_track(des, jd_start, jd_end):
    """Positions every 2 days, every hour within 0.1 au, every 10 minutes within 0.005 au."""
    jd, r, v = horizons_vectors(des, jd_start, jd_end, '2 d')
    for limit, pad, step in [(0.1, 3.0, '1 h'), (0.005, 0.5, '10 m')]:
        earth, _ = s3.earth_moon(jd)
        dist = np.linalg.norm(r - earth, axis=1)
        for a, b in windows(jd, dist, limit, pad):
            a, b = max(a, jd_start), min(b, jd_end)
            j2, r2, v2 = horizons_vectors(des, a, b, step)
            keep = (jd < a) | (jd > b)
            jd, r, v = np.concatenate([jd[keep], j2]), np.concatenate([r[keep], r2]), np.concatenate([v[keep], v2])
            order = np.argsort(jd)
            jd, r, v = jd[order], r[order], v[order]
    return jd, r, v


def segments_from_track(jd, r, v):
    """Same rule as step 2: start a new ellipse when the old one drifts more than 2%."""
    earth, moon = s3.earth_moon(jd)
    d_near = np.minimum(np.linalg.norm(r - earth, axis=1), np.linalg.norm(r - earth - moon, axis=1))
    tol = np.maximum(TOLERANCE * d_near, TOL_FLOOR)
    t = jd - JD0

    def start_ellipse(i):
        # The page stores start times as 32-bit numbers, which round times near 2030
        # by up to ~40 seconds (= ~1,000 km of travel!). So we round the start time
        # first and move the ellipse's starting point to match the rounded time.
        t32 = float(np.float32(t[i]))
        el = state_to_elements(r[i:i + 1], v[i:i + 1])
        el[0, 5] += np.sqrt(MU_SUN / el[0, 0]**3) * (t32 - t[i])
        return t32, as_float32(el)

    t0, el = start_ellipse(0)
    out = []
    for i in range(1, len(t)):
        if np.linalg.norm(kepler_position(el, np.array([t[i] - t0]))[0] - r[i]) > tol[i]:
            out.append((t0, *el[0]))
            t0, el = start_ellipse(i - 1)
    out.append((t0, *el[0]))
    seg = np.array(out)
    seg[0, 0] = float(np.float32(t[0]))
    return seg


def objects_to_replace(orbits):
    index = {d: k for k, d in enumerate(orbits.des)}
    rows = [r for r in s3.table(s3.load('cad_earth_2000_2130.json')) if r['des'] in index]
    bad = set(FEATURED)
    for r in rows:
        k, jd = index[r['des']], float(r['jd'])
        jpl, lo, hi = (float(r[c]) for c in ('dist', 'dist_min', 'dist_max'))
        model, _ = s3.closest_approach(orbits, k, jd)
        if abs(model / jpl - 1) > 0.10 and not (lo <= model <= hi):
            bad.add(r['des'])
    return sorted(bad)


if __name__ == '__main__':
    z = dict(np.load(DATA / 'segments_sim.npz'))
    orbits = s3.Orbits('segments_sim.npz')
    print('Finding objects whose simulated path disagrees with JPL ...')
    replace = objects_to_replace(orbits)
    print(f'  {len(replace)} objects: {", ".join(replace)}')

    des = [str(d) for d in z['des']]
    keep = ~np.isin(z['obj'], [des.index(d) for d in replace])
    obj, t_start, elements = [z['obj'][keep]], [z['t_start'][keep]], [z['elements'][keep]]
    for d in replace:
        jd, r, v = jpl_track(d, float(z['jd_start']), float(z['jd_end']))
        seg = segments_from_track(jd, r, v)
        print(f'  {d:12s} {len(jd):6d} JPL positions -> {len(seg):4d} segments')
        obj.append(np.full(len(seg), des.index(d)))
        t_start.append(seg[:, 0])
        elements.append(seg[:, 1:])
    obj, t_start, elements = np.concatenate(obj), np.concatenate(t_start), np.concatenate(elements)
    order = np.lexsort((t_start, obj))
    np.savez_compressed(DATA / 'segments.npz', des=z['des'], obj=obj[order], t_start=t_start[order],
                        elements=elements[order], jd0=z['jd0'], jd_start=z['jd_start'],
                        jd_end=z['jd_end'], jpl_tracks=np.array(replace))
    print(f'Saved {len(obj)} segments to {DATA / "segments.npz"}')
