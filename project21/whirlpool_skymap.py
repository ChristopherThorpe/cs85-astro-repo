"""Sky map of X-ray sources in the Whirlpool Galaxy (M51), drawn over a visible-light photo.

Data sources:
  - X-ray sources: Chandra Source Catalog 2.2, Chandra X-ray Center (CfA), https://cxc.cfa.harvard.edu/csc/
  - Background photo: Digitized Sky Survey 2 (red light), from NASA's SkyView service,
    https://skyview.gsfc.nasa.gov/

Run from the repo's top folder:  python3 project21/whirlpool_skymap.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvo
import requests
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS

# --- Settings -------------------------------------------------------------
NAME = "Whirlpool Galaxy (M51)"
RA, DEC = 202.4696, 47.1952   # centre of M51, in degrees
HALF_SIZE = 0.12              # half-width of the box, in degrees (the full Moon is about 0.5 deg across)
IMAGE_PIXELS = 1000           # width and height of the background photo

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# --- 1. Download the X-ray sources (cached in data/ after the first run) --
catalog_file = "data/csc22_m51.csv"
if os.path.exists(catalog_file):
    sources = pd.read_csv(catalog_file)
else:
    # Lines of right ascension (RA) bunch together away from the sky's equator,
    # so the box needs to be wider in RA to cover the same patch of sky.
    ra_half = HALF_SIZE / np.cos(np.radians(DEC))
    query = f"""SELECT m.name, m.ra, m.dec, m.significance, m.flux_aper_b, m.hard_hm
                FROM csc22.master_source m
                WHERE m.ra BETWEEN {RA - ra_half} AND {RA + ra_half}
                  AND m.dec BETWEEN {DEC - HALF_SIZE} AND {DEC + HALF_SIZE}"""
    tap = pyvo.dal.TAPService("https://cda.cfa.harvard.edu/csc22tap")
    sources = tap.search(query, maxrec=20000).to_table().to_pandas()
    sources.to_csv(catalog_file, index=False)
print(len(sources), "X-ray sources in the box")

# --- 2. Download a visible-light photo of the same patch of sky -----------
# SkyView returns a FITS file: the image plus a header saying which sky position each pixel shows.
photo_file = "data/dss2red_m51.fits"
if not os.path.exists(photo_file):
    params = {"Position": f"{RA},{DEC}", "Coordinates": "J2000", "Survey": "DSS2 Red",
              "Pixels": IMAGE_PIXELS, "Size": 2 * HALF_SIZE, "Return": "FITS"}
    r = requests.get("https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl", params=params, timeout=180)
    r.raise_for_status()
    with open(photo_file, "wb") as f:
        f.write(r.content)

with fits.open(photo_file) as hdul:
    photo = hdul[0].data
    wcs = WCS(hdul[0].header)  # the pixel -> sky-position "map" stored in the header

# --- 3. Draw the map ------------------------------------------------------
fig = plt.figure(figsize=(9, 8.5))
ax = fig.add_subplot(projection=wcs)

# The galaxy's core is far brighter than its faint arms. An "asinh stretch" squeezes the
# bright end so both show up at once (like adjusting the contrast on a phone photo).
norm = ImageNormalize(photo, interval=PercentileInterval(99.7), stretch=AsinhStretch(0.05))
ax.imshow(photo, origin="lower", cmap="gray", norm=norm)

# Dot size grows with X-ray brightness (on a log scale, since brightness spans 10,000x).
# Some very faint sources have a flux of zero or no value at all; they get the smallest size.
flux = sources["flux_aper_b"]
faintest = flux[flux > 0].min()
flux = flux.where(flux > 0, faintest)
size = 6 + 22 * (np.log10(flux) - np.log10(faintest))

# Dot colour shows hardness: blue = softer (lower-energy) X-rays, red = harder (higher-energy).
# Sources with no hardness value are drawn as yellow outlines.
sky = ax.get_transform("icrs")  # tells matplotlib the dot positions are RA/Dec, not pixels
has_hr = sources["hard_hm"].notna()
dots = ax.scatter(sources.loc[has_hr, "ra"], sources.loc[has_hr, "dec"], s=size[has_hr],
                  c=sources.loc[has_hr, "hard_hm"], cmap="coolwarm", vmin=-1, vmax=1, alpha=0.85,
                  edgecolor="black", linewidth=0.4, transform=sky)
ax.scatter(sources.loc[~has_hr, "ra"], sources.loc[~has_hr, "dec"], s=size[~has_hr],
           facecolor="none", edgecolor="yellow", linewidth=0.8, transform=sky,
           label="no hardness measured")

# Keep the view to the photo's edges (a few catalog sources sit just outside it).
ax.set_xlim(-0.5, photo.shape[1] - 0.5)
ax.set_ylim(-0.5, photo.shape[0] - 0.5)

ax.coords[0].set_axislabel("Right ascension (RA)")
ax.coords[1].set_axislabel("Declination (Dec)")
ax.set_title(f"X-ray sources in the {NAME}\n"
             "Chandra Source Catalog 2.2 over a DSS2 red-light image (NASA SkyView)")
fig.colorbar(dots, ax=ax, fraction=0.046, pad=0.04,
             label="X-ray hardness  (blue = softer, red = harder)")
ax.legend(loc="upper left", fontsize=8, facecolor="black", labelcolor="white", framealpha=0.6)
ax.text(0.99, 0.01, "Bigger dot = brighter in X-rays", transform=ax.transAxes, ha="right", va="bottom",
        color="white", fontsize=9, bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))

out = "plots/21_whirlpool_m51_skymap.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)
