# BLISS project page (`gh-pages`)

Static project page (Bulma + Plotly loaded from CDN, no build step), served
by GitHub Pages from this branch once the repository is public.

## Enabling the page

In **Settings → Pages**, choose:

- **Source:** Deploy from a branch
- **Branch:** `gh-pages`
- **Folder:** `/ (root)`

## Where the data comes from

All data is bundled in `static/data/`:

- **Dataset Explorer:** `catalog.json` (every shell of the review subset with
  its list of defects), `geometry.json` and `grid.json` (the 8,192-node
  subset of the shared mesh and the hemisphere grid) and `shells/<id>.json`
  (surface deviation, load path, displacement at the peak and buckling mode
  of each shell). The shape of every shell is drawn in the browser from its
  list of defects with the generator of the dataset.
- **Leaderboard and Results:** `leaderboard.json` (the full tables of
  protocols B1, B2 and B3 of the paper) and `predictions.json` (the
  predictions of every model on the dense shells shared by B1 and B3).

## Updating the bundled data

Regenerate `static/data/` from the dataset release and the paper tables:

```bash
python scripts/build_explorer.py --sample <review subset>
python scripts/build_data.py --sample <review subset> \
    --tables <paper tables dir> --runs <finished runs dir>
```

## Anonymized copy for review

`review/` is an anonymized copy of the page (data embedded as small scripts
under `review/static/data/`), served through anonymous.4open.science.
Regenerate it after any change to the site with
`python scripts/build_review.py`; the script fails if an identifying term
is left.
