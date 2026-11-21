# Security Policy

WaveSimFEM takes the security and integrity of scientific simulation workflows
seriously. This document outlines supported versions, responsible vulnerability
reporting, and security boundaries when running WaveSimFEM locally or on HPC
clusters.

## Supported Versions

Security fixes are prioritized for the latest release line:

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in WaveSimFEM, please report it
responsibly:

- **Primary Contact**: Email the maintainer directly at
  [ductho.le@outlook.com](mailto:ductho.le@outlook.com).
- **GitHub Security Advisory**: Open a private draft advisory via
  [GitHub Security Advisories](https://github.com/ductho-le/WaveSimFEM/security/advisories/new).

Please **do not** open public GitHub issues for sensitive security
vulnerabilities. We aim to acknowledge receipt within 48 hours and coordinate
remediation prior to public disclosure.

---

## Security Boundaries & Safe Usage

### 1. Safe Archive Deserialization (`.npz`)
All WaveSimFEM post-processing, verification, and utility readers open NumPy
archives strictly with `allow_pickle=False`, preventing arbitrary Python
bytecode execution when inspecting sensor archives. Still, treat archives from
untrusted sources with standard caution and verify checksums before sharing.

### 2. Strict Configuration Parsing (YAML / TOML / JSON)
YAML configurations are parsed exclusively with `_UniqueSafeLoader` (derived
from `yaml.SafeLoader`), blocking arbitrary YAML object tags, duplicate keys,
and unrecognized schema fields. TOML and JSON inputs use Python's standard
library parsers (`tomllib`, `json`).

### 3. External LaTeX Rendering (`output.usetex`)
External LaTeX rendering (`output.usetex: true`) invokes local `latex`,
`dvipng`, and Ghostscript binaries via Matplotlib and is **disabled by default**
(`output.usetex: false`). Do not enable `usetex: true` when post-processing
archives or configurations supplied by untrusted parties.

### 4. Output Paths, MPI & HPC Environments
- **Atomic writes**: Archives and reports are written atomically while honoring
  the process umask and existing file permissions. Use dedicated `output_dir`
  paths per run rather than shared world-writable directories.
- **Distributed execution**: Multi-rank CPU simulations communicate over
  standard MPI (`mpi4py` / MPICH / PETSc) communicators. On HPC clusters, run
  MPI jobs within isolated internal cluster networks.
- **GPU execution**: Optional CUDA acceleration uses CuPy (`cupy-cuda12x`) and
  runs custom elementwise kernels locally on the selected NVIDIA GPU device.
