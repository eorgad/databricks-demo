"""
Drug Discovery Demo Utilities Package
"""

__version__ = "1.0.0"

from .data_generator import generate_molecular_dataset, MolecularDataGenerator
from .molecular_descriptors import (
    calculate_descriptors,
    calculate_morgan_fingerprint,
    check_lipinski_violations,
    calculate_qed,
    MolecularDescriptorCalculator
)

__all__ = [
    'generate_molecular_dataset',
    'MolecularDataGenerator',
    'calculate_descriptors',
    'calculate_morgan_fingerprint',
    'check_lipinski_violations',
    'calculate_qed',
    'MolecularDescriptorCalculator'
]
