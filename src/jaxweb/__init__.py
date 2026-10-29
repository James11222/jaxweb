"""
jaxweb: the NEXUS and NEXUS+ multiscale morphology filters for the cosmic web
(Cautun et al. 2013) on many GPUs with JAX.
"""

from __future__ import annotations

from .fft import irfft3, rfft3
from .hessian import hessian
from .io import load_density
from .mesh import initialize, make_mesh
from .signatures import NEXUS, NEXUSPLUS, maximum_signature, to_pycosmommf_layout
from .smoothing import density_floor, smooth
from .tagging import CLUSTER, FILAMENT, VOID, WALL, tag_structures

__version__ = "0.1.0"

__all__ = [
    "CLUSTER",
    "FILAMENT",
    "NEXUS",
    "NEXUSPLUS",
    "VOID",
    "WALL",
    "__version__",
    "density_floor",
    "hessian",
    "initialize",
    "irfft3",
    "load_density",
    "make_mesh",
    "maximum_signature",
    "rfft3",
    "smooth",
    "tag_structures",
    "to_pycosmommf_layout",
]
