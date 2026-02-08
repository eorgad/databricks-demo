"""
Molecular Descriptor Calculation
Uses RDKit to calculate molecular descriptors and fingerprints from SMILES strings.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_descriptors(smiles: str) -> Dict[str, float]:
    """
    Calculate molecular descriptors from SMILES string using RDKit
    
    Args:
        smiles: SMILES representation of molecule
        
    Returns:
        Dictionary of molecular descriptors
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
        
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            # Return NaN for invalid SMILES
            return {
                'molecular_weight': np.nan,
                'logp': np.nan,
                'tpsa': np.nan,
                'num_h_donors': np.nan,
                'num_h_acceptors': np.nan,
                'num_rotatable_bonds': np.nan,
                'num_aromatic_rings': np.nan,
                'num_heteroatoms': np.nan,
                'num_rings': np.nan,
                'num_saturated_rings': np.nan,
                'fraction_csp3': np.nan,
                'num_aliphatic_rings': np.nan
            }
        
        descriptors = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_h_donors': Lipinski.NumHDonors(mol),
            'num_h_acceptors': Lipinski.NumHAcceptors(mol),
            'num_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            'num_aromatic_rings': Lipinski.NumAromaticRings(mol),
            'num_heteroatoms': Lipinski.NumHeteroatoms(mol),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_saturated_rings': rdMolDescriptors.CalcNumSaturatedRings(mol),
            'fraction_csp3': rdMolDescriptors.CalcFractionCsp3(mol),
            'num_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol)
        }
        
        return descriptors
        
    except Exception as e:
        print(f"Error calculating descriptors for SMILES '{smiles}': {e}")
        return {
            'molecular_weight': np.nan,
            'logp': np.nan,
            'tpsa': np.nan,
            'num_h_donors': np.nan,
            'num_h_acceptors': np.nan,
            'num_rotatable_bonds': np.nan,
            'num_aromatic_rings': np.nan,
            'num_heteroatoms': np.nan,
            'num_rings': np.nan,
            'num_saturated_rings': np.nan,
            'fraction_csp3': np.nan,
            'num_aliphatic_rings': np.nan
        }


def calculate_morgan_fingerprint(smiles: str, 
                                 radius: int = 2, 
                                 n_bits: int = 2048) -> Optional[np.ndarray]:
    """
    Calculate Morgan (circular) fingerprint
    
    Args:
        smiles: SMILES string
        radius: Fingerprint radius (default: 2)
        n_bits: Number of bits in fingerprint (default: 2048)
        
    Returns:
        Numpy array of fingerprint bits, or None if invalid SMILES
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
        
    except Exception as e:
        print(f"Error calculating fingerprint for SMILES '{smiles}': {e}")
        return None


def calculate_maccs_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """
    Calculate MACCS keys fingerprint (166 bits)
    
    Args:
        smiles: SMILES string
        
    Returns:
        Numpy array of MACCS keys, or None if invalid SMILES
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)
        
    except Exception as e:
        print(f"Error calculating MACCS keys for SMILES '{smiles}': {e}")
        return None


def check_lipinski_violations(descriptors: Dict[str, float]) -> Dict[str, any]:
    """
    Check Lipinski's Rule of Five violations
    
    Rules:
    - Molecular weight <= 500 Da
    - LogP <= 5
    - H-bond donors <= 5
    - H-bond acceptors <= 10
    
    Args:
        descriptors: Dictionary of molecular descriptors
        
    Returns:
        Dictionary with violation flags and count
    """
    violations = {
        'mw_violation': descriptors.get('molecular_weight', 0) > 500,
        'logp_violation': descriptors.get('logp', 0) > 5,
        'hbd_violation': descriptors.get('num_h_donors', 0) > 5,
        'hba_violation': descriptors.get('num_h_acceptors', 0) > 10
    }
    
    violations['num_violations'] = sum(violations.values())
    violations['is_drug_like'] = violations['num_violations'] <= 1  # Allow 1 violation
    
    return violations


def calculate_qed(smiles: str) -> Optional[float]:
    """
    Calculate Quantitative Estimate of Drug-likeness (QED)
    
    Args:
        smiles: SMILES string
        
    Returns:
        QED score (0-1), or None if invalid SMILES
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import QED
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return QED.qed(mol)
        
    except Exception as e:
        print(f"Error calculating QED for SMILES '{smiles}': {e}")
        return None


def calculate_synthetic_accessibility(smiles: str) -> Optional[float]:
    """
    Calculate Synthetic Accessibility Score (SA Score)
    Range: 1 (easy to synthesize) to 10 (difficult to synthesize)
    
    Args:
        smiles: SMILES string
        
    Returns:
        SA Score, or None if invalid SMILES
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import RDConfig
        import os
        import sys
        
        # Add SA_Score directory to path
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        import sascorer
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return sascorer.calculateScore(mol)
        
    except Exception as e:
        # SA Score might not be available in all RDKit installations
        return None


def smiles_to_mol_image(smiles: str, 
                        img_size: Tuple[int, int] = (300, 300),
                        highlight_atoms: List[int] = None) -> Optional[bytes]:
    """
    Generate molecule image from SMILES
    
    Args:
        smiles: SMILES string
        img_size: Image size (width, height)
        highlight_atoms: List of atom indices to highlight
        
    Returns:
        PNG image as bytes, or None if invalid SMILES
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from io import BytesIO
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        img = Draw.MolToImage(mol, size=img_size, highlightAtoms=highlight_atoms)
        
        # Convert to bytes
        buf = BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
        
    except Exception as e:
        print(f"Error generating image for SMILES '{smiles}': {e}")
        return None


def calculate_similarity(smiles1: str, smiles2: str, method: str = 'tanimoto') -> Optional[float]:
    """
    Calculate molecular similarity between two SMILES strings
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        method: Similarity method ('tanimoto', 'dice')
        
    Returns:
        Similarity score (0-1), or None if invalid SMILES
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return None
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        if method == 'tanimoto':
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        elif method == 'dice':
            return DataStructs.DiceSimilarity(fp1, fp2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
            
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return None


def find_substructure_matches(smiles: str, substructure_smarts: str) -> Optional[List[Tuple]]:
    """
    Find substructure matches in molecule
    
    Args:
        smiles: SMILES string to search
        substructure_smarts: SMARTS pattern to find
        
    Returns:
        List of atom index tuples for matches, or None if invalid
    """
    try:
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(smiles)
        pattern = Chem.MolFromSmarts(substructure_smarts)
        
        if mol is None or pattern is None:
            return None
        
        return mol.GetSubstructMatches(pattern)
        
    except Exception as e:
        print(f"Error finding substructure: {e}")
        return None


class MolecularDescriptorCalculator:
    """
    Batch calculator for molecular descriptors
    Optimized for processing large datasets
    """
    
    def __init__(self, radius: int = 2, n_bits: int = 2048):
        """
        Initialize calculator
        
        Args:
            radius: Fingerprint radius
            n_bits: Number of fingerprint bits
        """
        self.radius = radius
        self.n_bits = n_bits
    
    def process_batch(self, smiles_list: List[str]) -> List[Dict]:
        """
        Process batch of SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of descriptor dictionaries
        """
        results = []
        
        for smiles in smiles_list:
            desc = calculate_descriptors(smiles)
            fp = calculate_morgan_fingerprint(smiles, self.radius, self.n_bits)
            
            result = {
                'smiles': smiles,
                **desc
            }
            
            # Add fingerprint if successfully calculated
            if fp is not None:
                for i, bit in enumerate(fp):
                    result[f'fp_{i}'] = int(bit)
            
            # Add drug-likeness checks
            lipinski = check_lipinski_violations(desc)
            result.update(lipinski)
            
            results.append(result)
        
        return results


# Pandas UDF for Spark integration
def create_descriptor_pandas_udf():
    """
    Create a Pandas UDF for calculating descriptors in Spark
    
    Returns:
        Pandas UDF function
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
    import pandas as pd
    
    schema = StructType([
        StructField("molecular_weight", DoubleType()),
        StructField("logp", DoubleType()),
        StructField("tpsa", DoubleType()),
        StructField("num_h_donors", IntegerType()),
        StructField("num_h_acceptors", IntegerType()),
        StructField("num_rotatable_bonds", IntegerType()),
        StructField("num_aromatic_rings", IntegerType()),
        StructField("num_heteroatoms", IntegerType()),
        StructField("num_rings", IntegerType()),
        StructField("num_saturated_rings", IntegerType()),
        StructField("fraction_csp3", DoubleType()),
        StructField("num_aliphatic_rings", IntegerType())
    ])
    
    @pandas_udf(schema)
    def calculate_descriptors_udf(smiles_series: pd.Series) -> pd.DataFrame:
        results = [calculate_descriptors(s) for s in smiles_series]
        return pd.DataFrame(results)
    
    return calculate_descriptors_udf


# Example usage
if __name__ == "__main__":
    # Test with some example molecules
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen
    ]
    
    print("Testing molecular descriptor calculations:\n")
    
    for smiles in test_smiles:
        print(f"SMILES: {smiles}")
        descriptors = calculate_descriptors(smiles)
        print(f"Descriptors: {descriptors}")
        
        violations = check_lipinski_violations(descriptors)
        print(f"Lipinski violations: {violations['num_violations']}")
        print(f"Drug-like: {violations['is_drug_like']}")
        print()
