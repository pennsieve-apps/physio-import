FROM --platform=linux/amd64 python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# Pinned Python dependencies, binary wheels only, no dependency resolution at build time (requirements.txt lists
# every package). Test folders the app never reads are removed. Nothing is downloaded at run time.
COPY requirements.txt /processor/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: --root-user-action=ignore -r /processor/requirements.txt \
    && find /usr/local/lib/python3.12 -depth -type d -name __pycache__ -exec rm -rf {} + \
    && find /usr/local/lib/python3.12/site-packages -depth -type d -name tests \
         \( -path "*/scipy/*" -o -path "*/numpy/*" -o -path "*/pandas/*" -o -path "*/matplotlib/*" -o -path "*/neo/*" \
            -o -path "*/pyedflib/*" -o -path "*/h5py/*" -o -path "*/wfdb/*" \) -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/matplotlib/mpl-data/sample_data \
    && python3 -c "import neo.rawio, wfdb, pyedflib, pynwb, nwbinspector, pymatreader, bioread, openpyxl, h5py, yaml; print('pynwb', pynwb.__version__, 'neo', neo.__version__)" \
    && mkdir -p /opt/physio-import/matplotlib \
    && MPLCONFIGDIR=/opt/physio-import/matplotlib python3 -c "import matplotlib.font_manager" \
    && ls /opt/physio-import/matplotlib/fontlist-*.json \
    && chmod -R a+rX /opt/physio-import

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

# The Matplotlib font cache is built above into a read-only path; at start the app copies it to $TMPDIR/mpl
# (MPLCONFIGDIR) because HOME and /tmp are replaced at run time on Lambda. Threads (OMP/OpenBLAS/MKL) are set by
# the app from the cgroup CPU quota before numpy loads. HDF5 file locking is off: inputs may be on read-only mounts.
ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PHYSIO_IMPORT_MPL_CACHE=/opt/physio-import/matplotlib \
    MPLBACKEND=Agg \
    HDF5_USE_FILE_LOCKING=FALSE

# uid 1000 on ECS; any uid works (nothing in the image is written at run time).
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app
USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
