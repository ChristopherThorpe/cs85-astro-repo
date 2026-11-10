"""Sky map of Chandra X-ray sources for a chosen target, drawn over a visible-light photo.

Data sources:
  - X-ray sources: Chandra Source Catalog 2.2, Chandra X-ray Center (CfA), https://cxc.cfa.harvard.edu/csc/
  - Background photo: Digitized Sky Survey 2 (red light), from NASA's SkyView service,
    https://skyview.gsfc.nasa.gov/

Run from the repo's top folder, naming a target from the TARGETS list below:
    python3 project21/skymap.py whirlpool_m51
    python3 project21/skymap.py 47tuc
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvo
import requests
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS

# --- Targets ----------------------------------------------------------------
# name:      the title shown on the plot
# ra, dec:   the centre of the map, in degrees
# half_size: half-width of the box, in degrees (for scale, the full Moon is about 0.5 degrees across)
# contrast:  how strongly to squeeze the photo's bright end; smaller = more squeezing.
#            Use a small value (0.05) for a galaxy on dark sky, a larger one (0.3) for a field full of stars.
# zoom:      optional half-width, in degrees, of a zoomed-in second panel for a crowded centre (None = no zoom)
TARGETS = {
    "whirlpool_m51": dict(name="Whirlpool Galaxy (M51)", ra=202.4696, dec=47.1952,
                          half_size=0.12, contrast=0.05, zoom=None),
    "47tuc": dict(name="globular cluster 47 Tucanae", ra=6.0236, dec=-72.0813,
                  half_size=0.10, contrast=0.3, zoom=0.017),
}
IMAGE_PIXELS = 1000  # width and height of the background photo


def get_sources(key, ra, dec, half_size):
    """Download the Chandra sources in a box around (ra, dec), caching them in data/."""
    catalog_file = f"data/csc22_{key}.csv"
    if os.path.exists(catalog_file):
        return pd.read_csv(catalog_file)
    # Lines of right ascension (RA) bunch together away from the sky's equator,
    # so the box needs to be wider in RA to cover the same patch of sky.
    ra_half = half_size / np.cos(np.radians(dec))
    query = f"""SELECT m.name, m.ra, m.dec, m.significance, m.flux_aper_b, m.hard_hm
                FROM csc22.master_source m
                WHERE m.ra BETWEEN {ra - ra_half} AND {ra + ra_half}
                  AND m.dec BETWEEN {dec - half_size} AND {dec + half_size}"""
    tap = pyvo.dal.TAPService("https://cda.cfa.harvard.edu/csc22tap")
    sources = tap.search(query, maxrec=20000).to_table().to_pandas()
    sources.to_csv(catalog_file, index=False)
    return sources


def get_photo(key, ra, dec, half_size):
    """Download a DSS2 red-light photo from SkyView (cached in data/). Returns the image and its WCS.

    The FITS file holds the image plus a header (the "WCS") saying which sky position each pixel shows.
    """
    photo_file = f"data/dss2red_{key}.fits"
    if not os.path.exists(photo_file):
        params = {"Position": f"{ra},{dec}", "Coordinates": "J2000", "Survey": "DSS2 Red",
                  "Pixels": IMAGE_PIXELS, "Size": 2 * half_size, "Return": "FITS"}
        r = requests.get("https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl", params=params, timeout=180)
        r.raise_for_status()
        with open(photo_file, "wb") as f:
            f.write(r.content)
    with fits.open(photo_file) as hdul:
        return hdul[0].data, WCS(hdul[0].header)


def draw_panel(ax, photo, sources, contrast):
    """Draw the photo and the X-ray dots on one panel. Returns the dots, for the colour bar."""
    # Bright centres are far brighter than faint outskirts. An "asinh stretch" squeezes the
    # bright end so both show up at once (like adjusting the contrast on a phone photo).
    norm = ImageNormalize(photo, interval=PercentileInterval(99.99), stretch=AsinhStretch(contrast))
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
    ax.coords[0].set_axislabel("Right ascension (RA)")
    ax.coords[1].set_axislabel("Declination (Dec)")
    return dots


def make_skymap(key):
    t = TARGETS[key]
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    sources = get_sources(key, t["ra"], t["dec"], t["half_size"])
    print(len(sources), "X-ray sources in the box around", t["name"])
    photo, wcs = get_photo(key, t["ra"], t["dec"], t["half_size"])

    zoom = t["zoom"]
    fig = plt.figure(figsize=(16, 8.5) if zoom else (9, 8.5))

    # Left (or only) panel: the whole field.
    ax = fig.add_subplot(1, 2 if zoom else 1, 1, projection=wcs)
    dots = draw_panel(ax, photo, sources, t["contrast"])
    # Keep the view to the photo's edges (a few catalog sources sit just outside it).
    ax.set_xlim(-0.5, photo.shape[1] - 0.5)
    ax.set_ylim(-0.5, photo.shape[0] - 0.5)
    ax.set_title(f"X-ray sources in the {t['name']}\n"
                 "Chandra Source Catalog 2.2 over a DSS2 red-light image (NASA SkyView)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, facecolor="black", labelcolor="white", framealpha=0.6)
    ax.text(0.99, 0.01, "Bigger dot = brighter in X-rays", transform=ax.transAxes, ha="right",
            va="bottom", color="white", fontsize=9,
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))

    # Right panel: the same picture, zoomed in on the centre.
    if zoom:
        # Convert the zoom size from degrees to photo pixels, and centre it on the target.
        cx, cy = wcs.world_to_pixel_values(t["ra"], t["dec"])
        half_px = zoom / abs(wcs.wcs.cdelt[1])
        ax.add_patch(plt.Rectangle((cx - half_px, cy - half_px), 2 * half_px, 2 * half_px,
                                   fill=False, edgecolor="yellow", linewidth=1.2, linestyle="--"))
        zax = fig.add_subplot(1, 2, 2, projection=wcs)
        draw_panel(zax, photo, sources, t["contrast"])
        zax.set_xlim(cx - half_px, cx + half_px)
        zax.set_ylim(cy - half_px, cy + half_px)
        n_in = int(((abs(sources["ra"] - t["ra"]) * np.cos(np.radians(t["dec"])) < zoom)
                    & (abs(sources["dec"] - t["dec"]) < zoom)).sum())
        zax.set_title(f"Zoom on the centre (yellow box): {n_in} sources", fontsize=10)

    fig.subplots_adjust(wspace=0.3)  # room between the panels for the axis labels
    fig.colorbar(dots, ax=fig.axes, fraction=0.025 if zoom else 0.046, pad=0.03,
                 label="X-ray hardness  (blue = softer, red = harder)")

    out = f"plots/21_{key}_skymap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in TARGETS:
        sys.exit(f"Usage: python3 project21/skymap.py TARGET   (TARGET is one of: {', '.join(TARGETS)})")
    make_skymap(sys.argv[1])
