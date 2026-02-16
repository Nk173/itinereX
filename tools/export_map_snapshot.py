#!/usr/bin/env python3
"""Export a JS-driven Folium/Leaflet HTML map to a static PDF/PNG snapshot.

Why this exists:
- GitHub READMEs cannot embed iframes.
- Folium maps are JavaScript-driven, so HTML->PDF tools that don't run JS won't work.

This script uses Playwright (headless Chromium) to render the page, wait for Leaflet
tiles to load, then exports:
- PDF via Chromium's print-to-PDF
- PNG screenshot (optional)

Examples:
  python tools/export_map_snapshot.py \
    --url https://nk173.github.io/itinereX/roads_after_cleaning.html \
    --out-pdf docs/roads_after_cleaning.pdf \
    --out-png docs/roads_after_cleaning.png

  python tools/export_map_snapshot.py \
    --html docs/roads_after_cleaning.html \
    --out-pdf docs/roads_after_cleaning.pdf

Setup:
  pip install playwright
  playwright install chromium
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional


def _as_file_url(path: pathlib.Path) -> str:
    # file:// URLs must be absolute.
    return path.resolve().as_uri()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="Remote URL to render")
    src.add_argument("--html", help="Path to local HTML file to render")

    p.add_argument("--out-pdf", required=True, help="Output PDF path")
    p.add_argument("--out-png", help="Optional output PNG path")

    p.add_argument("--viewport-width", type=int, default=1600)
    p.add_argument("--viewport-height", type=int, default=1000)

    p.add_argument(
        "--clip-selector",
        default=".leaflet-container",
        help="CSS selector to clip PNG screenshots to (default: Leaflet map container). Use empty string to disable.",
    )
    p.add_argument(
        "--pdf-landscape",
        action="store_true",
        help="Export PDF in landscape orientation (useful for wide maps).",
    )

    p.add_argument(
        "--wait-seconds",
        type=float,
        default=6.0,
        help="Extra wait after initial load (gives Leaflet tiles time to render)",
    )
    p.add_argument(
        "--tile-min",
        type=int,
        default=8,
        help="Minimum number of rendered Leaflet tile <img> to consider 'loaded'",
    )
    p.add_argument(
        "--timeout-seconds",
        type=float,
        default=90.0,
        help="Overall timeout for page load and tile wait",
    )

    # Optional re-centering of the Leaflet map after load.
    # Useful to create a consistent static snapshot (e.g., centered on the UK).
    p.add_argument("--center-lat", type=float, default=None, help="Latitude to center the map on")
    p.add_argument("--center-lon", type=float, default=None, help="Longitude to center the map on")
    p.add_argument("--zoom", type=int, default=None, help="Leaflet zoom level to set")

    return p.parse_args(argv)


def _ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _export(url: str, out_pdf: pathlib.Path, out_png: Optional[pathlib.Path], args: argparse.Namespace) -> None:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Playwright is not installed. Run: pip install playwright && playwright install chromium"
        ) from e

    _ensure_parent(out_pdf)
    if out_png is not None:
        _ensure_parent(out_png)

    timeout_ms = int(args.timeout_seconds * 1000)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport={"width": args.viewport_width, "height": args.viewport_height},
            device_scale_factor=2,
        )
        page = context.new_page()

        # Load the page.
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

        # Wait for Leaflet container to exist.
        page.wait_for_selector(".leaflet-container", timeout=timeout_ms)

        # Optional: re-center the Leaflet map if requested.
        if args.center_lat is not None and args.center_lon is not None:
            page.evaluate(
                """
                ({lat, lon, zoom}) => {
                  function isLeafletMap(obj) {
                    return obj && typeof obj.setView === 'function' && obj._container && obj._container.classList;
                  }

                  const candidates = [];
                  for (const k of Object.keys(window)) {
                    try {
                      const v = window[k];
                      if (isLeafletMap(v)) candidates.push(v);
                    } catch (e) {}
                  }

                  const map = candidates.find(m => m._container.classList.contains('leaflet-container')) || candidates[0];
                  if (!map) return false;

                  const z = (typeof zoom === 'number' && !Number.isNaN(zoom)) ? zoom : map.getZoom();
                  map.setView([lat, lon], z, { animate: false });
                  return true;
                }
                """,
                {"lat": float(args.center_lat), "lon": float(args.center_lon), "zoom": args.zoom},
            )

            # Give tiles time to refill after the view changes.
            page.wait_for_timeout(1500)

        # Best-effort: wait for some tiles to load (OSM tiles are <img class="leaflet-tile">).
        # Some Folium maps may render few/no tiles if a custom basemap is missing; in that case
        # we still export after a short grace period.
        try:
            page.wait_for_function(
                """
                (minCount) => {
                  const imgs = Array.from(document.querySelectorAll('img.leaflet-tile'));
                  const loaded = imgs.filter(i => i.complete && i.naturalWidth > 0 && i.naturalHeight > 0);
                  return loaded.length >= minCount;
                }
                """,
                arg=args.tile_min,
                timeout=min(timeout_ms, 45_000),
            )
        except Exception:
            pass

        if args.wait_seconds > 0:
            page.wait_for_timeout(int(args.wait_seconds * 1000))

        # Export PDF.
        page.pdf(
            path=str(out_pdf),
            format="A4",
            landscape=bool(args.pdf_landscape),
            print_background=True,
            margin={"top": "10mm", "right": "10mm", "bottom": "10mm", "left": "10mm"},
        )

        # Optional PNG screenshot.
        if out_png is not None:
            sel = (args.clip_selector or "").strip()
            if sel:
                locator = page.locator(sel)
                # Wait for the map element to be visible before capturing.
                locator.wait_for(state="visible", timeout=timeout_ms)
                locator.screenshot(path=str(out_png))
            else:
                page.screenshot(path=str(out_png), full_page=True)

        context.close()
        browser.close()


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    out_pdf = pathlib.Path(args.out_pdf)
    out_png = pathlib.Path(args.out_png) if args.out_png else None

    if args.url:
        url = args.url
    else:
        html_path = pathlib.Path(args.html)
        if not html_path.exists():
            raise FileNotFoundError(str(html_path))
        url = _as_file_url(html_path)

    _export(url=url, out_pdf=out_pdf, out_png=out_png, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
