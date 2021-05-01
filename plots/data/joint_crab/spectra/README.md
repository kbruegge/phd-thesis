# Spectra

This folder contains 1-D spectra for each dataset / observation
in the OGIP spectrum formats:
http://gamma-astro-data-formats.readthedocs.io/en/latest/ogip/index.html

- `pha` - counts on the on region
- `bkg` - counts in the off region plus on/off area factor info
- `arf` - effective area plus livetime info
- `rfm` - energy redistribution matrix

They are kind of intermediate results files, generated from the
input input data in the `data` folder by the `extract-spectra` step.
