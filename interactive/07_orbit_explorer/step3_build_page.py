"""
Step 3 of 3 for the Orbit Explorer: check the orbits against JPL and build the web page.

  1. Collect, for every asteroid: its name, its single most likely possible impact
     (date + probability), its other possible impact dates, and JPL's predicted
     close approaches.
  2. Work out Earth's and the Moon's positions for every day from JPL's DE440s.
  3. CHECK our orbits: for every close approach JPL predicts for these asteroids
     (2000 to 2130, closer than 0.05 au), compute the distance from our model and
     compare. The result is printed and shown on the page.
  4. Pack all of that into orbit_explorer.html (template.html + the data).

Run from the repository root after steps 1 and 2:
    python3 interactive/07_orbit_explorer/step3_build_page.py
"""
from pathlib import Path
import base64
import json
import sys
from datetime import date, datetime, timezone
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from step2_compute_orbits import (DATA, AU_KM, LD_KM, JD0, MU_SUN, EQ_TO_ECL, EPH,
                                  kepler_solve, orbit_axes, FEATURED)

HERE = Path(__file__).parent
OUT_HTML = HERE / 'orbit_explorer.html'
OUT_FRAGMENT = DATA / 'orbit_explorer_fragment.html'   # same page, without <html>/<head> wrapper


# ---------------------------------------------------------------- small helpers
def jd_from_date(text):
    """'2032-12-22.35' or '2032-Dec-22 14:57' -> Julian date (works for any year)."""
    text = text.strip()
    if ' ' in text:                                    # CAD style: 2032-Dec-22 14:57
        d = datetime.strptime(text, '%Y-%b-%d %H:%M')
        y, m, day, frac = d.year, d.month, d.day, (d.hour + d.minute / 60) / 24
    else:                                              # Sentry style: 2032-12-22.35
        y, m, rest = text.split('-')
        y, m, day, frac = int(y), int(m), int(float(rest)), float(rest) % 1
    a = (14 - m) // 12                                 # standard Gregorian-calendar formula
    yy, mm = y + 4800 - a, m + 12 * a - 3
    jdn = day + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045
    return jdn - 0.5 + frac


def b64(array, dtype):
    """Pack a numeric array as base64 text so it can sit inside the HTML file."""
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


def load(name):
    return json.load(open(DATA / name))


def table(reply):
    """JPL APIs return either a list of dicts or 'fields' + 'data' lists."""
    if 'fields' in reply:
        return [dict(zip(reply['fields'], row)) for row in reply.get('data', [])]
    return reply.get('data', [])


def num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- orbits (same maths as the page)
class Orbits:
    def __init__(self):
        z = np.load(DATA / 'segments.npz')
        self.des = [str(d) for d in z['des']]
        self.obj = z['obj']
        self.t = z['t_start'].astype(np.float32)          # stored as 32-bit floats on the page
        self.el = z['elements'].astype(np.float32)
        self.start = np.searchsorted(self.obj, np.arange(len(self.des) + 1)).astype(np.int32)
        self.jd_start, self.jd_end = float(z['jd_start']), float(z['jd_end'])

    def helio(self, k, t):
        """Heliocentric position (AU) of object k at times t (days since 2000-01-01)."""
        lo, hi = self.start[k], self.start[k + 1]
        s = lo + np.searchsorted(self.t[lo:hi], t, side='right') - 1
        s = np.clip(s, lo, hi - 1)
        a, e, i, om, w, M0 = self.el[s].astype(np.float64).T
        n = np.sqrt(MU_SUN / a**3)
        E = kepler_solve(M0 + n * (t - self.t[s].astype(np.float64)), e)
        P, Q = orbit_axes(i, om, w)
        x, y = a * (np.cos(E) - e), a * np.sqrt(1 - e**2) * np.sin(E)
        return x[:, None] * P + y[:, None] * Q


def earth_moon(jd):
    """Heliocentric Earth (AU) and geocentric Moon (AU), ecliptic J2000, from DE440s."""
    jd = np.atleast_1d(jd)
    emb = EPH[0, 3].compute(jd)
    earth = emb + EPH[3, 399].compute(jd)
    sun = EPH[0, 10].compute(jd)
    moon = EPH[3, 301].compute(jd) - EPH[3, 399].compute(jd)
    return (EQ_TO_ECL @ (earth - sun)).T / AU_KM, (EQ_TO_ECL @ moon).T / AU_KM


def closest_approach(orbits, k, jd_guess, body='earth', window=3.0):
    """Model's minimum distance (AU) to Earth or Moon within +-window days of jd_guess."""
    best_jd = jd_guess
    for half, step in [(window, 1 / 24), (2 / 24, 1 / 720)]:   # hourly, then every 2 minutes
        jd = best_jd + np.arange(-half, half + step / 2, step)
        earth, moon = earth_moon(jd)
        pos = orbits.helio(k, jd - JD0) - earth
        if body == 'moon':
            pos = pos - moon
        d = np.linalg.norm(pos, axis=1)
        best_jd = jd[np.argmin(d)]
    return d.min(), best_jd


# ---------------------------------------------------------------- build
def main():
    orbits = Orbits()
    des = orbits.des
    index = {d: k for k, d in enumerate(des)}
    t_min, t_max = orbits.jd_start - JD0, orbits.jd_end - JD0

    summary = {str(r['des']): r for r in table(load('sentry_summary.json'))}
    removed = {str(r['des']): r['removed'] for r in table(load('sentry_removed.json'))}
    sb = load('sbdb_neo_elements.json')
    sbdb = {row[0]: dict(zip(sb['fields'], row)) for row in sb['data']}

    # --- every possible impact (virtual impactor), grouped by object
    vis = [r for r in table(load('sentry_vi_all.json')) if str(r['des']) in index]
    vi_obj = np.array([index[str(r['des'])] for r in vis])
    vi_jd = np.array([jd_from_date(r['date']) for r in vis])
    vi_ip = np.array([float(r['ip']) for r in vis])
    order = np.lexsort((vi_jd, vi_obj))
    vi_obj, vi_jd, vi_ip = vi_obj[order], vi_jd[order], vi_ip[order]
    vis = [vis[i] for i in order]

    # --- JPL close approaches (Earth) for our objects, plus Moon approaches for featured ones
    cad_rows = [r for r in table(load('cad_earth_2000_2130.json')) if r['des'] in index]
    cad_obj = np.array([index[r['des']] for r in cad_rows])
    cad_jd = np.array([float(r['jd']) for r in cad_rows])
    cad_ld = np.array([float(r['dist']) for r in cad_rows]) * AU_KM / LD_KM
    featured_cad = {}
    for d in FEATURED:
        for body in ['earth', 'moon']:
            rows = table(load(f"cad_{d.replace(' ', '_')}_{body}.json"))
            featured_cad.setdefault(d, []).extend(
                {'body': body.capitalize(), 'jd': float(r['jd']), 'date': r['cd'],
                 'dist_km': float(r['dist']) * AU_KM} for r in rows)

    # --- one record per object for the page
    objects = []
    for k, d in enumerate(des):
        o = sbdb.get(d, {})
        rec = {'des': d, 'name': (o.get('full_name') or d).strip(),
               'H': num(o.get('H')), 'U': o.get('condition_code'),
               'arc_days': num(o.get('data_arc'))}
        if d in summary:
            s = summary[d]
            mine = np.where(vi_obj == k)[0]
            best = mine[np.argmax(vi_ip[mine])] if len(mine) else None
            rec.update({
                'featured': False,
                'ip_cum': num(s['ip']), 'ps_cum': num(s['ps_cum']), 'ts_max': num(s['ts_max']),
                'n_imp': int(s['n_imp']), 'range': s['range'], 'diam_km': num(s['diameter']),
                'v_inf': num(s['v_inf']), 'last_obs': s['last_obs'],
            })
            if best is not None:
                rec.update({'ip_max': float(vi_ip[best]), 'vi_date': vis[best]['date'],
                            'vi_t': float(vi_jd[best] - JD0), 'vi_ps': num(vis[best]['ps']),
                            'vi_ts': num(vis[best]['ts'])})
                # JPL's best-guess (nominal) pass within 30 days of that possible impact
                near = np.where((cad_obj == k) & (np.abs(cad_jd - vi_jd[best]) < 30))[0]
                if len(near):
                    j = near[np.argmin(cad_ld[near])]
                    rec['cad'] = {'t': float(cad_jd[j] - JD0), 'date': cad_rows[j]['cd'],
                                  'ld': float(cad_ld[j]),
                                  'ld_min': float(cad_rows[j]['dist_min']) * AU_KM / LD_KM,
                                  'ld_max': float(cad_rows[j]['dist_max']) * AU_KM / LD_KM}
        else:
            approaches = sorted(featured_cad.get(d, []), key=lambda a: a['jd'])
            rec.update({'featured': True, 'removed': removed.get(d),
                        'approaches': [{**a, 't': a['jd'] - JD0} for a in approaches]})
        objects.append(rec)

    # --- Earth (every 4 days) and Moon (every day), with a margin for interpolation
    e_t = np.arange(t_min - 12, t_max + 12, 4.0)
    m_t = np.arange(t_min - 3, t_max + 3, 1.0)
    earth, _ = earth_moon(e_t + JD0)
    _, moon = earth_moon(m_t + JD0)
    moon_ld = moon * AU_KM / LD_KM

    # --- the check against JPL
    print(f'Checking {len(cad_rows)} JPL close approaches against our model ...')
    check = []
    for k, jd, ld in zip(cad_obj, cad_jd, cad_ld):
        d, best = closest_approach(orbits, k, jd)
        check.append((ld, d * AU_KM / LD_KM, (best - jd) * 24, (jd - 2461200.5) / 365.25))
    check = np.array(check)
    rel = np.abs(check[:, 1] / check[:, 0] - 1)
    stats = {
        'n': int(len(check)),
        'median_pct': float(np.median(rel) * 100),
        'within5_pct': float(np.mean(rel < 0.05) * 100),
        'within20_pct': float(np.mean(rel < 0.20) * 100),
        'median_hours': float(np.median(np.abs(check[:, 2]))),
    }
    print(f"  median distance error {stats['median_pct']:.2f}%, "
          f"{stats['within5_pct']:.0f}% within 5%, {stats['within20_pct']:.0f}% within 20%, "
          f"median timing error {stats['median_hours']:.2f} h")
    for lo, hi in [(-30, 0), (0, 10), (10, 30), (30, 60), (60, 110)]:
        m = (check[:, 3] >= lo) & (check[:, 3] < hi)
        if m.any():
            print(f'    {lo:+4d} to {hi:+4d} years from 2026: median error '
                  f'{np.median(rel[m]) * 100:6.2f}%  ({m.sum()} approaches)')
    for d in FEATURED:
        k = index[d]
        for a in featured_cad[d]:
            dist, best = closest_approach(orbits, k, a['jd'], body=a['body'].lower())
            print(f"  {d:9s} {a['body']:5s} {a['date']}: JPL {a['dist_km']:>11,.0f} km, "
                  f"model {dist * AU_KM:>11,.0f} km, timing {(best - a['jd']) * 24:+.2f} h")

    data = {
        'meta': {
            'jd0': JD0, 't_min': t_min, 't_max': t_max, 'mu_sun': MU_SUN,
            'au_km': AU_KM, 'ld_km': LD_KM,
            'today_t': float(jd_from_date(date.today().isoformat() + '.5') - JD0),
            'built': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'check': stats, 'n_segments': int(len(orbits.t)),
        },
        'objects': objects,
        'seg_start': b64(orbits.start, '<i4'),
        'seg_t': b64(orbits.t, '<f4'),
        'seg_el': b64(orbits.el, '<f4'),
        'earth': {'t0': float(e_t[0]), 'step': 4.0, 'xyz': b64(earth, '<f4')},
        'moon': {'t0': float(m_t[0]), 'step': 1.0, 'xyz': b64(moon_ld, '<f4')},
        'vi': {'obj': b64(vi_obj, '<u2'), 't': b64(vi_jd - JD0, '<f4'), 'ip': b64(vi_ip, '<f4')},
        'cad': {'obj': b64(cad_obj, '<u2'), 't': b64(cad_jd - JD0, '<f4'), 'ld': b64(cad_ld, '<f4')},
    }
    payload = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    page = (HERE / 'template.html').read_text().replace('/*__EXPLORER_DATA__*/', payload)
    OUT_FRAGMENT.write_text(page)
    OUT_HTML.write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                        '<meta name="viewport" content="width=device-width, initial-scale=1, '
                        'viewport-fit=cover">\n</head>\n<body>\n' + page + '\n</body>\n</html>\n')
    print(f'Wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
