# Third-party components

The Docker image bundles these Python packages (exact versions in `requirements.txt`), installed from PyPI
wheels on `python:3.12.16-slim-bookworm` (Debian bookworm with CPython 3.12, PSF licence). No datasets or model
weights are bundled. Left out on purpose: aiohttp/requests (network access, never used), soundfile (FLAC-coded
WFDB signal files only; it bundles libsndfile, LGPL), docopt (bioread's command-line tools only).

| Component | Version | Authors / copyright holder | Licence | URL |
|---|---|---|---|---|
| Neo | 0.14.6 | Neo authors and contributors (NeuralEnsemble) | BSD-3-Clause | https://github.com/NeuralEnsemble/python-neo |
| quantities | 0.16.4 | Darren Dale and contributors | BSD-3-Clause | https://github.com/python-quantities/python-quantities |
| WFDB for Python | 4.3.1 | MIT Laboratory for Computational Physiology | MIT | https://github.com/MIT-LCP/wfdb-python |
| pyEDFlib | 0.1.42 | Holger Nahrstaedt, Chris Lee-Messer, Teunis van Beelen | BSD-3-Clause | https://github.com/holgern/pyedflib |
| PyNWB | 4.2.0 | The Regents of the University of California (LBNL), NWB developers | BSD-3-Clause-LBNL | https://github.com/NeurodataWithoutBorders/pynwb |
| HDMF | 6.2.0 | The Regents of the University of California (LBNL), HDMF developers | BSD-3-Clause-LBNL | https://github.com/hdmf-dev/hdmf |
| NWB Inspector | 0.7.2 | The Regents of the University of California (LBNL); Cody Baker, Ben Dichter and contributors | BSD-3-Clause-LBNL | https://github.com/NeurodataWithoutBorders/nwbinspector |
| h5py | 3.16.0 | Andrew Collette and contributors (bundles HDF5, The HDF Group, BSD-3-Clause) | BSD-3-Clause | https://github.com/h5py/h5py |
| pymatreader | 1.3.2 | Dirk Gütlin & Thomas Hartmann | BSD-2-Clause | https://gitlab.com/obob/pymatreader |
| xmltodict | 1.0.4 | Martin Blech | MIT | https://github.com/martinblech/xmltodict |
| bioread | 2025.5.2 | Board of Regents of the University of Wisconsin System (Nate Vack, John Ollinger, Dan Fitch, Mike Davison) | MIT | https://github.com/uwmadison-chm/bioread |
| openpyxl | 3.1.5 | openpyxl authors (Charlie Clark, maintainer) | MIT | https://foss.heptapod.net/openpyxl/openpyxl |
| et_xmlfile | 2.0.0 | et_xmlfile authors (Charlie Clark, maintainer) | MIT | https://foss.heptapod.net/openpyxl/et_xmlfile |
| pandas | 3.1.1 | The pandas development team | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| NumPy | 2.5.4 | NumPy Developers | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| SciPy | 1.18.1 | SciPy Developers | BSD-3-Clause | https://scipy.org |
| Matplotlib | 3.11.2 | Matplotlib Development Team | Matplotlib License (PSF-based) | https://matplotlib.org |
| PyYAML | 6.0.3 | Ingy döt Net, Kirill Simonov and contributors | MIT | https://github.com/yaml/pyyaml |
| attrs | 26.1.0 | Hynek Schlawack | MIT | https://github.com/python-attrs/attrs |
| click | 8.5.1 | Pallets | BSD-3-Clause | https://github.com/pallets/click |
| contourpy | 1.4.0 | ContourPy Developers | BSD-3-Clause | https://github.com/contourpy/contourpy |
| cycler | 0.12.1 | Matplotlib project | BSD-3-Clause | https://github.com/matplotlib/cycler |
| fonttools | 4.68.0 | Just van Rossum and fontTools contributors | MIT | https://github.com/fonttools/fonttools |
| fsspec | 2026.9.0 | fsspec developers | BSD-3-Clause | https://github.com/fsspec/filesystem_spec |
| isodate | 0.7.2 | Gerhard Weis, Hugo van Kemenade | BSD-3-Clause | https://github.com/gweis/isodate |
| jsonschema | 4.26.0 | Julian Berman | MIT | https://github.com/python-jsonschema/jsonschema |
| jsonschema-specifications | 2025.9.1 | Julian Berman | MIT | https://github.com/python-jsonschema/jsonschema-specifications |
| kiwisolver | 1.5.1 | Nucleic Development Team | BSD-3-Clause | https://github.com/nucleic/kiwi |
| natsort | 8.4.0 | Seth M. Morton | MIT | https://github.com/SethMMorton/natsort |
| packaging | 26.3 | Donald Stufft and contributors | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| Pillow | 13.0.0 | Jeffrey 'Alex' Clark and contributors; PIL: Secret Labs AB, Fredrik Lundh and contributors | MIT-CMU | https://github.com/python-pillow/Pillow |
| platformdirs | 4.14.5 | platformdirs contributors | MIT | https://github.com/tox-dev/platformdirs |
| pyparsing | 3.3.3 | Paul McGuire | MIT | https://github.com/pyparsing/pyparsing |
| python-dateutil | 4.0.0 | dateutil contributors | Apache-2.0 AND BSD-3-Clause | https://github.com/dateutil/dateutil |
| referencing | 0.37.0 | Julian Berman | MIT | https://github.com/python-jsonschema/referencing |
| rpds-py | 2026.9.1 | Julian Berman | MIT | https://github.com/crate-py/rpds |
| ruamel.yaml | 0.19.1 | Anthon van der Neut | MIT | https://sourceforge.net/projects/ruamel-yaml/ |
| tqdm | 4.70.1 | tqdm developers | MPL-2.0 AND MIT | https://github.com/tqdm/tqdm |
| typing_extensions | 4.16.0 | Python Software Foundation | PSF-2.0 | https://github.com/python/typing_extensions |
| tzdata | 2026.6 | Paul Ganssle, Stan Ulbrych (IANA time zone data, public domain) | Apache-2.0 | https://github.com/python/tzdata |

Test-only (not in the image): pytest (MIT); pyabf 2.3.8 (Scott W Harden, MIT) writes the ABF1 file of the
synthetic fixture. The fixture's EDF+, WFDB and LabChart-style MATLAB files are written by this repository's own
code, byte by byte, so the readers are checked against an independent writer.

Formats not read, and why: Spike2 `.smrx` needs CED's sonpy (binary-only, no source; GPL-3.0-or-later per its PyPI metadata; Linux wheel only for CPython 3.14); LabChart `.adicht` on
Linux needs adichtion (GPL-3.0-or-later). Both are reported per file with export instructions.

This app's own code (reader glue, channel typing, the HDF5 trial layout and overlap merging, the LabChart MATLAB
and text export parsers, the BIDS physio reader, the checks and figures) is Apache-2.0.
