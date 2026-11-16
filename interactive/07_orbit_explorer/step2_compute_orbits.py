"""
Step 2 of 3 for the Orbit Explorer: work out where every asteroid is, 2000 to 2130.

Why not just use each asteroid's orbit ellipse?
  An orbit ellipse ("two-body" or Kepler motion) only includes the Sun's gravity.
  We tested it against JPL's own close-approach predictions: it is excellent for
  a couple of years, but decades later it can be off by hundreds of lunar distances,
  because Earth and the other planets keep nudging these asteroids.

So we run a proper gravity simulation (an "N-body" simulation) with the REBOUND
package. It includes the Sun, all eight planets and the Moon, starting from JPL's
DE440 planetary positions. Each asteroid starts from its JPL orbit (SBDB).

Storing 2,253 asteroids x every hour for 131 years would be gigabytes. Instead we
store each asteroid's path as a chain of short Kepler ellipses ("segments"). A new
segment starts whenever the simple ellipse would drift away from the real simulated
path by more than 2% of the asteroid's distance from Earth (or the Moon, whichever
is closer), with a 300 km floor. 2% is too small to see on the explorer's log-scale
map, but it lets the web page compute any moment instantly and keeps the file small.

Important: this is the *best-guess* (nominal) orbit. Many of these asteroids were
only seen for a few days, so where they really are is uncertain, sometimes very much
so. Sentry's possible impacts come from that uncertainty, not from the best guess.

Run from the repository root (takes several minutes):
    python3 interactive/07_orbit_explorer/step2_compute_orbits.py
Output: data/07_explorer/segments_sim.npz
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import rebound
from jplephem.spk import SPK

DATA = Path('data/07_explorer')

# ---------------------------------------------------------------- constants
AU_KM = 149597870.7
LD_KM = 384400.0                         # mean Earth-Moon distance ("lunar distance")
DAY_S = 86400.0
JD0 = 2451544.5                          # 2000-01-01 00:00 TDB. Time in this project = days since JD0
JD_START = JD0                           # start of the explorer's timeline
JD_END = pd.Timestamp('2131-01-01').to_julian_date()
JD_REF = 2461200.5                       # 2026-Jun-09: the date most JPL orbits are given at

# Gravity strength (GM) of each body from JPL DE440, km^3/s^2, converted to AU^3/day^2
GM_KM3S2 = {
    'sun': 132712440041.279419, 'mercury': 22031.868551, 'venus': 324858.592000,
    'earth': 398600.435507, 'moon': 4902.800118, 'mars': 42828.375816,
    'jupiter': 126712764.100000, 'saturn': 37940584.841800,
    'uranus': 5794556.400000, 'neptune': 6836527.100580,
}
GM = {k: v * DAY_S**2 / AU_KM**3 for k, v in GM_KM3S2.items()}
MU_SUN = GM['sun']
BODIES = list(GM_KM3S2)                  # order of the massive bodies in the simulation
I_SUN, I_EARTH, I_MOON = BODIES.index('sun'), BODIES.index('earth'), BODIES.index('moon')

# JPL's orbits are in "ecliptic J2000" coordinates; the DE440 ephemeris is equatorial (ICRF).
OBLIQUITY = np.radians(84381.448 / 3600)
EQ_TO_ECL = np.array([[1, 0, 0],
                      [0, np.cos(OBLIQUITY), np.sin(OBLIQUITY)],
                      [0, -np.sin(OBLIQUITY), np.cos(OBLIQUITY)]])

TOLERANCE = 0.02                         # 2% of distance to Earth/Moon
TOL_FLOOR = 300.0 / AU_KM                # but never tighter than 300 km
FEATURED = ['2024 YR4', '99942']         # removed from Sentry, but famous; shown separately

EPH = SPK.open(str(DATA / 'de440s.bsp'))


# ---------------------------------------------------------------- planets from DE440
def body_state(name, jd):
    """Barycentric position (AU) and velocity (AU/day) of a body, ecliptic J2000."""
    seg = {'sun': [(0, 10)], 'mercury': [(0, 1)], 'venus': [(0, 2)],
           'earth': [(0, 3), (3, 399)], 'moon': [(0, 3), (3, 301)], 'mars': [(0, 4)],
           'jupiter': [(0, 5)], 'saturn': [(0, 6)], 'uranus': [(0, 7)], 'neptune': [(0, 8)]}[name]
    pos, vel = np.zeros(3), np.zeros(3)
    for a, b in seg:
        p, v = EPH[a, b].compute_and_differentiate(jd)    # km and km/day
        pos += p
        vel += v
    return EQ_TO_ECL @ pos / AU_KM, EQ_TO_ECL @ vel / AU_KM


# ---------------------------------------------------------------- orbit maths
def elements_to_state(a, e, i, om, w, M, mu=MU_SUN):
    """Orbital elements (angles in radians) -> heliocentric position & velocity arrays (N,3)."""
    E = kepler_solve(M, e)
    cosE, sinE = np.cos(E), np.sin(E)
    b = a * np.sqrt(1 - e**2)
    n = np.sqrt(mu / a**3)
    x, y = a * (cosE - e), b * sinE                    # position in the orbit plane
    vx = -a * n * sinE / (1 - e * cosE)
    vy = b * n * cosE / (1 - e * cosE)
    P, Q = orbit_axes(i, om, w)
    return x[:, None] * P + y[:, None] * Q, vx[:, None] * P + vy[:, None] * Q


def orbit_axes(i, om, w):
    """Unit vectors P (toward perihelion) and Q (90 degrees ahead), shape (N,3)."""
    cO, sO, ci, si, cw, sw = np.cos(om), np.sin(om), np.cos(i), np.sin(i), np.cos(w), np.sin(w)
    P = np.stack([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si], axis=-1)
    Q = np.stack([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si], axis=-1)
    return P, Q


def kepler_solve(M, e):
    """Solve Kepler's equation M = E - e sin E for E (Newton's method)."""
    M = np.mod(M, 2 * np.pi)
    E = np.where(e > 0.8, np.pi, M)
    for _ in range(30):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def state_to_elements(r, v, mu=MU_SUN):
    """Heliocentric position & velocity (N,3) -> elements a, e, i, om, w, M (N,) each."""
    rn = np.linalg.norm(r, axis=1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    evec = np.cross(v, h) / mu - r / rn[:, None]
    e = np.linalg.norm(evec, axis=1)
    a = 1 / (2 / rn - np.einsum('ij,ij->i', v, v) / mu)
    i = np.arctan2(np.hypot(h[:, 0], h[:, 1]), h[:, 2])
    om = np.arctan2(h[:, 0], -h[:, 1])
    node = np.stack([np.cos(om), np.sin(om), np.zeros_like(om)], axis=1)
    hhat = h / hn[:, None]
    w = np.arctan2(np.einsum('ij,ij->i', np.cross(node, evec), hhat), np.einsum('ij,ij->i', node, evec))
    nu = np.arctan2(np.einsum('ij,ij->i', np.cross(evec, r), hhat), np.einsum('ij,ij->i', evec, r))
    E = np.arctan2(np.sqrt(1 - e**2) * np.sin(nu), e + np.cos(nu))
    M = E - e * np.sin(E)
    if np.any(e >= 1):
        raise ValueError('found an unbound (hyperbolic) orbit; this code assumes ellipses')
    return np.stack([a, e, np.mod(i, 2 * np.pi), np.mod(om, 2 * np.pi), np.mod(w, 2 * np.pi),
                     np.mod(M, 2 * np.pi)], axis=1)


def as_float32(el):
    """The web page stores numbers as 32-bit floats; round the same way here."""
    return el.astype(np.float32).astype(np.float64)


def kepler_position(el, dt):
    """Position from elements el (N,6; M at the anchor time) after dt days (N,)."""
    a, e, i, om, w, M0 = el.T
    n = np.sqrt(MU_SUN / a**3)
    E = kepler_solve(M0 + n * dt, e)
    P, Q = orbit_axes(i, om, w)
    x, y = a * (np.cos(E) - e), a * np.sqrt(1 - e**2) * np.sin(E)
    return x[:, None] * P + y[:, None] * Q


# ---------------------------------------------------------------- simulation helpers
def new_simulation(jd):
    sim = rebound.Simulation()
    sim.G = 1.0                          # masses below are GM values, so G = 1
    sim.integrator = 'ias15'             # adaptive integrator, handles close encounters well
    for name in BODIES:
        x, v = body_state(name, jd)
        sim.add(m=GM[name], x=x[0], y=x[1], z=x[2], vx=v[0], vy=v[1], vz=v[2])
    sim.N_active = len(BODIES)           # asteroids are massless "test particles"
    sim.testparticle_type = 0
    sim.t = jd - JD0
    return sim


def add_asteroids(sim, r_helio, v_helio):
    # Copy the Sun's position into plain numbers first. (Keeping sim.particles[I_SUN]
    # around is a trap: adding particles can move REBOUND's memory, so it goes stale.)
    sun = sim.particles[I_SUN]
    sun_x = np.array([sun.x, sun.y, sun.z])
    sun_v = np.array([sun.vx, sun.vy, sun.vz])
    for r, v in zip(r_helio + sun_x, v_helio + sun_v):
        sim.add(m=0, x=r[0], y=r[1], z=r[2], vx=v[0], vy=v[1], vz=v[2])


def read_state(sim):
    """Heliocentric asteroid positions/velocities, plus Earth and Moon (heliocentric)."""
    xyz = np.zeros((sim.N, 3))
    vel = np.zeros((sim.N, 3))
    sim.serialize_particle_data(xyz=xyz, vxvyvz=vel)
    xyz -= xyz[I_SUN]
    vel -= vel[I_SUN]
    nb = len(BODIES)
    return xyz[nb:], vel[nb:], xyz[I_EARTH], vel[I_EARTH], xyz[I_MOON]


def bring_to_reference_epoch(epochs, r, v):
    """Some orbits are given at other dates. Simulate each one to JD_REF."""
    out_r, out_v = r.copy(), v.copy()
    for sign in (+1, -1):                # +1: epochs before JD_REF, -1: epochs after
        idx = np.where(sign * (JD_REF - epochs) > 0)[0]
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(sign * epochs[idx])]        # in the order we will reach them
        sim = new_simulation(epochs[idx[0]])
        sim.dt = sign * 0.1
        order = []
        for ep in np.unique(epochs[idx])[::sign]:
            sim.integrate(ep - JD0)
            group = idx[epochs[idx] == ep]
            add_asteroids(sim, r[group], v[group])
            order.extend(group)
        sim.integrate(JD_REF - JD0)
        rr, vv, *_ = read_state(sim)
        out_r[order], out_v[order] = rr, vv
        print(f'  moved {len(idx)} orbits {"forward" if sign > 0 else "backward"} to the reference date')
    return out_r, out_v


# ---------------------------------------------------------------- the main run
def run(r0, v0, direction):
    """Simulate from JD_REF to the end (direction=+1) or start (-1), building segments."""
    sim = new_simulation(JD_REF)
    sim.dt = direction * 0.1
    add_asteroids(sim, r0, v0)
    n_obj = len(r0)
    t_end = (JD_END if direction > 0 else JD_START) - JD0

    t = JD_REF - JD0
    r_prev, v_prev = r0.copy(), v0.copy()
    anchor_t = np.full(n_obj, t)
    anchor_el = as_float32(state_to_elements(r0, v0))
    bound = np.full(n_obj, t)            # the segment boundary on the JD_REF side
    segments = []                        # rows: (object, lower bound, anchor time, 6 elements)
    n_steps, n_coarse = 0, 0
    clock = time.time()

    _, _, earth, v_earth, moon = read_state(sim)
    while direction * (t_end - t) > 1e-9:
        # Choose the output step: 12 hours normally, shorter when something is very close
        # to Earth or the Moon (down to 22.5 minutes), so fast flybys are tracked closely.
        d_near = np.minimum(np.linalg.norm(r_prev - earth, axis=1), np.linalg.norm(r_prev - moon, axis=1))
        speed = np.linalg.norm(v_prev - v_earth, axis=1)
        dt = np.clip(0.05 * np.min(d_near / speed), 1 / 64, 0.5)
        dt = 2.0 ** np.floor(np.log2(dt))                 # 1/64, 1/32, ... 1/2 day
        t_next = t + direction * dt
        if direction * (t_next - t_end) > 0:
            t_next = t_end
        sim.integrate(t_next)
        r, v, earth, v_earth, moon = read_state(sim)

        # Does each asteroid's current ellipse still match the simulation?
        predicted = kepler_position(anchor_el, t_next - anchor_t)
        err = np.linalg.norm(predicted - r, axis=1)
        d_near = np.minimum(np.linalg.norm(r - earth, axis=1), np.linalg.norm(r - moon, axis=1))
        tol = np.maximum(TOLERANCE * d_near, TOL_FLOOR)
        bad = np.where(err > tol)[0]
        if len(bad):
            # Close each failing segment at the previous step, and start a new ellipse there
            for k in bad:
                lower = bound[k] if direction > 0 else t
                segments.append((k, lower, anchor_t[k], *anchor_el[k]))
                bound[k] = t
            anchor_t[bad] = t
            anchor_el[bad] = as_float32(state_to_elements(r_prev[bad], v_prev[bad]))
            # If even the brand-new ellipse misses at this step, our steps were too coarse
            err2 = np.linalg.norm(kepler_position(anchor_el[bad], t_next - t) - r[bad], axis=1)
            n_coarse += int(np.sum(err2 > tol[bad]))

        r_prev, v_prev = r, v
        t = t_next
        n_steps += 1
        if n_steps % 20000 == 0:
            print(f'    reached {pd.Timestamp("2000-01-01") + pd.Timedelta(days=t):%Y-%m-%d}'
                  f'  ({n_steps} steps, {len(segments)} segments, {time.time() - clock:.0f}s)')

    # Close the final segment of every asteroid
    for k in range(n_obj):
        lower = bound[k] if direction > 0 else t_end
        segments.append((k, lower, anchor_t[k], *anchor_el[k]))
    print(f'  {"forward" if direction > 0 else "backward"} run: {n_steps} steps, '
          f'{len(segments)} segments, {n_coarse} too-coarse steps, {time.time() - clock:.0f}s')
    return np.array(segments)


def load_objects():
    """Sentry objects (sorted by risk) plus the featured removed objects, with their orbits."""
    sentry = pd.DataFrame(json.load(open(DATA / 'sentry_summary.json'))['data'])
    s = json.load(open(DATA / 'sbdb_neo_elements.json'))
    orbits = pd.DataFrame(s['data'], columns=s['fields']).set_index('pdes')
    des = list(sentry['des'].astype(str)) + FEATURED
    missing = [d for d in des if d not in orbits.index]
    if missing:
        raise RuntimeError(f'no JPL orbit found for {missing}')
    o = orbits.loc[des]
    el = o[['a', 'e', 'i', 'om', 'w', 'ma']].astype(float).to_numpy()
    el[:, 2:] = np.radians(el[:, 2:])
    return des, o['epoch'].astype(float).to_numpy(), el


def run_group(r0, v0, members, direction):
    """Run one group of asteroids; return segments with the asteroids' overall index."""
    seg = run(r0[members], v0[members], direction)
    seg[:, 0] = members[seg[:, 0].astype(int)]
    return seg


if __name__ == '__main__':
    from multiprocessing import Pool
    des, epochs, el = load_objects()
    print(f'{len(des)} objects ({len(des) - len(FEATURED)} on Sentry + {len(FEATURED)} featured)')
    r, v = elements_to_state(*el.T)
    r, v = bring_to_reference_epoch(epochs, r, v)

    # Asteroids don't affect each other, so we can simulate them in separate groups,
    # several at once (one per processor core). Fewer asteroids per group also means
    # fewer slow-down steps for close flybys.
    groups = np.array_split(np.arange(len(des)), 8)
    tasks = [(r, v, g, d) for d in (+1, -1) for g in groups]
    print(f'Simulating 2000 to 2131 in {len(tasks)} pieces ...')
    with Pool() as pool:
        pieces = pool.starmap(run_group, tasks)
    seg = np.concatenate(pieces)
    seg = seg[np.lexsort((seg[:, 1], seg[:, 0]))]      # sort by object, then by start time
    # Re-express each ellipse's mean anomaly at the segment's start time,
    # so the web page only needs (start time + 6 elements) per segment.
    obj, lo, anchor = seg[:, 0].astype(int), seg[:, 1], seg[:, 2]
    elements = seg[:, 3:].copy()
    n = np.sqrt(MU_SUN / elements[:, 0]**3)
    elements[:, 5] = np.mod(elements[:, 5] + n * (lo - anchor), 2 * np.pi)
    np.savez_compressed(DATA / 'segments_sim.npz', des=np.array(des), obj=obj, t_start=lo,
                        elements=elements, jd0=JD0, jd_start=JD_START, jd_end=JD_END)
    print(f'Saved {len(seg)} segments to {DATA / "segments_sim.npz"}')
