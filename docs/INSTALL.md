# Installation

## Python analysis environment

Use the pinned Conda environment:

```bash
conda env create -f environment.yml
conda activate for2107-tozzi
python -m pip install -e .
for2107-repro preflight --config configs/smoke.yaml
```

## fMRIPrep with Docker

Install Docker Engine using the instructions for the host operating system, then
verify it independently:

```bash
docker version
docker run --rm hello-world
```

Register for a free license at
`https://surfer.nmr.mgh.harvard.edu/registration.html`, save it as
`license.txt` in the project root, and do not commit it. fMRIPrep 25.1.4 checks
for this license even when `--fs-no-reconall` is enabled. Pull the pinned image:

```bash
docker pull nipreps/fmriprep:25.1.4
```

Inspect the exact command without running it:

```bash
for2107-repro fmriprep-command --config configs/full.yaml
```

The same configuration model supports `apptainer` for HPC and `executable` for
a manually prepared installation. Containers are recommended because fMRIPrep
depends on pinned external FSL, ANTs, AFNI, FreeSurfer, Connectome Workbench,
and BIDS Validator installations that Conda alone does not encapsulate.
