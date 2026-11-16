# Sentry Orbit Explorer (project 07)

An interactive page showing every asteroid on NASA/JPL's Sentry impact-risk list around
Earth and the Moon, for any moment from 2000 to 2130.

- **Near Earth view:** Earth at the centre, the Sun always at the top, distance on a log scale
  (each ring is 10× farther). The Moon's orbit, geostationary satellites, 0.05 au and 1 au are marked.
- **Inner solar system view:** a top-down view with the Sun at the centre.
- **Timeline:** drag the slider or click the strip, play at different speeds, or step by hour, day,
  month or year. The grey bars show how many objects have their most likely possible impact in each
  year. Selected objects show every possible impact date (ticks) and JPL's predicted close passes (▲).
- **Object list:** name, colour, highest single impact chance and its date. Tick boxes to select
  several objects, click a date to jump there, and use "Map shows selected only" to hide the rest.
- **Featured for comparison:** 2024 YR4 (passes 23,000 km from the Moon on 2032-12-22) and
  99942 Apophis (passes 38,000 km from Earth on 2029-04-13). Both have been removed from the risk list.

## Open it

Download `orbit_explorer.html` and open it in a web browser. It needs an internet connection the
first time to load the plotting library (Plotly). GitHub shows the file's code, not the page.

## Rebuild it (about 15 minutes)

Run from the repository root:

```bash
python3 interactive/07_orbit_explorer/step1_download.py
```

```bash
python3 interactive/07_orbit_explorer/step2_compute_orbits.py
```

```bash
python3 interactive/07_orbit_explorer/step2b_jpl_tracks.py
```

```bash
python3 interactive/07_orbit_explorer/step3_build_page.py
```

| Step | What it does |
|---|---|
| 1 | Downloads the Sentry list, every possible impact, the orbits, JPL's close approaches and the DE440s planet positions into `data/07_explorer/` (about 60 MB). |
| 2 | Runs a REBOUND gravity simulation (Sun, 8 planets, Moon) for all 2,253 objects from 2000 to 2130, about 10 minutes on 4 cores. Paths are stored as short Kepler ellipses that stay within 2% of the simulation. |
| 2b | For the 23 objects the simulation can't follow (after very close flybys, and two temporary "mini-moons"), downloads JPL Horizons positions instead (about 200 MB, cached). |
| 3 | Checks the simulation against 3,854 JPL close-approach predictions (median distance error 0.21%), then writes `orbit_explorer.html`. |

Needs `rebound` (`pip install rebound`) on top of the course packages.

## What to keep in mind

- Dots follow each asteroid's **best-guess** orbit. Sentry's possible impacts come from the
  *uncertainty* in the orbit, so on a possible impact date the best-guess dot is usually not at Earth.
  The detail card shows how close JPL's best-guess orbit passes on that date.
- Many Sentry objects were only seen for a few days, so their real positions decades ahead are very
  uncertain. The detail card shows JPL's uncertainty code (0 = very well known, 9 = extremely uncertain).
- Times are UTC to within about a minute.

## Data sources

- NASA/JPL CNEOS Sentry API: https://ssd-api.jpl.nasa.gov/doc/sentry.html
- JPL Small-Body Database Query API: https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html
- JPL Close-Approach Data API: https://ssd-api.jpl.nasa.gov/doc/cad.html
- JPL Horizons API: https://ssd-api.jpl.nasa.gov/doc/horizons.html
- JPL DE440s planetary ephemeris: https://ssd.jpl.nasa.gov/planets/eph_export.html
- REBOUND N-body code (Rein & Liu 2012): https://rebound.readthedocs.io
