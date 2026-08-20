"""Reaction-assembly interfaces shared by generation, routing, and audits.

Assembly adapters load transforms and role policy from the vendored registry.  Callers never embed
reaction SMARTS or reinterpret a registry hit as route certification.
"""

from forge.assembly.api import AssemblyAdapter, ForwardAssemblyCheck
from forge.assembly.ugi3 import Ugi3AssemblyAdapter, Ugi3AssemblyError

__all__ = [
    "AssemblyAdapter",
    "ForwardAssemblyCheck",
    "Ugi3AssemblyAdapter",
    "Ugi3AssemblyError",
]
