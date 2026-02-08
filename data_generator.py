"""
Synthetic Molecular Data Generator
Generates realistic molecular structures with simulated properties for demo purposes.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple


class MolecularDataGenerator:
    """Generate synthetic molecular data for drug discovery demo"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the molecular data generator
        
        Args:
            random_seed: Seed for reproducibility
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Common molecular fragments for building SMILES
        self.fragments = {
            'aromatic': ['c1ccccc1', 'c1ccncc1', 'c1ccc2ccccc2c1', 'c1ccoc1', 'c1ccsc1'],
            'aliphatic': ['C', 'CC', 'CCC', 'CCCC', 'C(C)C', 'CC(C)C'],
            'functional': ['O', 'N', 'S', 'Cl', 'F', 'Br', '[N+](=O)[O-]', 'C(=O)O', 'C(=O)N'],
            'linkers': ['C', 'CC', 'N', 'O', 'S', 'C=C', 'C#C']
        }
        
        # Properties ranges based on Lipinski's Rule of Five and real drugs
        self.property_ranges = {
            'molecular_weight': (150, 600),
            'logp': (-2, 6),
            'tpsa': (20, 140),
            'num_h_donors': (0, 6),
            'num_h_acceptors': (0, 12),
            'num_rotatable_bonds': (0, 12),
            'num_aromatic_rings': (0, 4),
            'num_heteroatoms': (1, 10)
        }
    
    def generate_smiles(self, complexity: str = 'medium') -> str:
        """
        Generate a synthetic SMILES string
        
        Args:
            complexity: 'simple', 'medium', or 'complex'
            
        Returns:
            SMILES string
        """
        complexity_map = {
            'simple': (1, 2),
            'medium': (2, 4),
            'complex': (3, 6)
        }
        
        n_fragments = random.randint(*complexity_map.get(complexity, (2, 4)))
        
        # Build SMILES from fragments
        smiles_parts = []
        
        # Start with aromatic core
        smiles_parts.append(random.choice(self.fragments['aromatic']))
        
        # Add additional fragments
        for _ in range(n_fragments - 1):
            fragment_type = random.choice(['aliphatic', 'functional', 'aromatic'])
            fragment = random.choice(self.fragments[fragment_type])
            
            # Add linker sometimes
            if random.random() > 0.5:
                linker = random.choice(self.fragments['linkers'])
                smiles_parts.append(linker)
            
            smiles_parts.append(fragment)
        
        # Simple concatenation (not chemically validated, but sufficient for demo)
        smiles = ''.join(smiles_parts)
        
        return smiles
    
    def generate_molecular_properties(self, smiles: str) -> Dict[str, float]:
        """
        Generate realistic molecular properties based on SMILES
        Properties are correlated to make data realistic
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of molecular properties
        """
        # Base properties on SMILES length and composition
        smiles_length = len(smiles)
        num_carbons = smiles.count('C') + smiles.count('c')
        num_nitrogens = smiles.count('N') + smiles.count('n')
        num_oxygens = smiles.count('O') + smiles.count('o')
        
        # Molecular weight (correlated with length)
        mw = 150 + (smiles_length * 8) + np.random.normal(0, 30)
        mw = np.clip(mw, *self.property_ranges['molecular_weight'])
        
        # LogP (lipophilicity - correlated with carbons)
        logp = (num_carbons * 0.5) - (num_oxygens * 0.5) - (num_nitrogens * 0.3) + np.random.normal(0, 1)
        logp = np.clip(logp, *self.property_ranges['logp'])
        
        # TPSA (polar surface area - correlated with heteroatoms)
        tpsa = (num_oxygens * 20) + (num_nitrogens * 15) + np.random.normal(0, 10)
        tpsa = np.clip(tpsa, *self.property_ranges['tpsa'])
        
        # H-bond donors and acceptors
        h_donors = int(np.clip(num_nitrogens + (num_oxygens * 0.5) + np.random.normal(0, 0.5), 
                              *self.property_ranges['num_h_donors']))
        h_acceptors = int(np.clip(num_oxygens + num_nitrogens + np.random.normal(0, 1), 
                                 *self.property_ranges['num_h_acceptors']))
        
        # Other properties
        rotatable_bonds = int(np.clip(smiles_length * 0.15 + np.random.normal(0, 1), 
                                     *self.property_ranges['num_rotatable_bonds']))
        aromatic_rings = int(np.clip(smiles.count('c1') + np.random.normal(0, 0.5), 
                                    *self.property_ranges['num_aromatic_rings']))
        heteroatoms = int(np.clip(num_nitrogens + num_oxygens + smiles.count('S') + np.random.normal(0, 0.5),
                                 *self.property_ranges['num_heteroatoms']))
        
        return {
            'molecular_weight': round(mw, 2),
            'logp': round(logp, 2),
            'tpsa': round(tpsa, 2),
            'num_h_donors': h_donors,
            'num_h_acceptors': h_acceptors,
            'num_rotatable_bonds': rotatable_bonds,
            'num_aromatic_rings': aromatic_rings,
            'num_heteroatoms': heteroatoms
        }
    
    def simulate_toxicity(self, properties: Dict[str, float]) -> Tuple[bool, float]:
        """
        Simulate toxicity based on molecular properties
        Uses simplified rules based on drug-likeness
        
        Args:
            properties: Dictionary of molecular properties
            
        Returns:
            Tuple of (is_toxic, toxicity_probability)
        """
        toxicity_score = 0.0
        
        # Lipinski's Rule violations increase toxicity likelihood
        if properties['molecular_weight'] > 500:
            toxicity_score += 0.15
        if properties['logp'] > 5:
            toxicity_score += 0.2
        if properties['num_h_donors'] > 5:
            toxicity_score += 0.1
        if properties['num_h_acceptors'] > 10:
            toxicity_score += 0.1
        
        # Very low TPSA can indicate poor drug-likeness
        if properties['tpsa'] < 30:
            toxicity_score += 0.15
        
        # High number of aromatic rings
        if properties['num_aromatic_rings'] > 3:
            toxicity_score += 0.1
        
        # Add some randomness
        toxicity_score += np.random.normal(0, 0.1)
        toxicity_score = np.clip(toxicity_score, 0, 1)
        
        # Threshold for classification
        is_toxic = toxicity_score > 0.5
        
        return is_toxic, round(toxicity_score, 4)
    
    def simulate_solubility(self, properties: Dict[str, float]) -> float:
        """
        Simulate aqueous solubility (LogS) based on molecular properties
        LogS typically ranges from -10 (poorly soluble) to 0 (highly soluble)
        
        Args:
            properties: Dictionary of molecular properties
            
        Returns:
            LogS value
        """
        # Simplified solubility model based on known correlations
        logs = 0.5
        
        # Molecular weight decreases solubility
        logs -= (properties['molecular_weight'] - 300) * 0.005
        
        # LogP (lipophilicity) decreases solubility
        logs -= properties['logp'] * 0.5
        
        # Polar surface area increases solubility
        logs += (properties['tpsa'] - 70) * 0.01
        
        # H-bond donors/acceptors increase solubility
        logs += properties['num_h_donors'] * 0.1
        logs += properties['num_h_acceptors'] * 0.05
        
        # Add noise
        logs += np.random.normal(0, 0.5)
        
        # Clip to realistic range
        logs = np.clip(logs, -8, 1)
        
        return round(logs, 2)
    
    def simulate_bioactivity(self, properties: Dict[str, float]) -> Tuple[str, float]:
        """
        Simulate bioactivity level and IC50 value
        
        Args:
            properties: Dictionary of molecular properties
            
        Returns:
            Tuple of (activity_class, ic50_value)
        """
        # Simplified activity model
        activity_score = 0.5
        
        # Optimal MW around 300-500
        mw = properties['molecular_weight']
        if 300 <= mw <= 500:
            activity_score += 0.2
        
        # Optimal LogP around 2-4
        if 2 <= properties['logp'] <= 4:
            activity_score += 0.2
        
        # Add randomness
        activity_score += np.random.normal(0, 0.2)
        activity_score = np.clip(activity_score, 0, 1)
        
        # Classify activity
        if activity_score > 0.7:
            activity_class = 'active'
            ic50 = np.random.uniform(0.01, 1.0)  # nM
        elif activity_score > 0.4:
            activity_class = 'moderate'
            ic50 = np.random.uniform(1.0, 10.0)  # nM
        else:
            activity_class = 'inactive'
            ic50 = np.random.uniform(10.0, 1000.0)  # nM
        
        return activity_class, round(ic50, 3)
    
    def generate_dataset(self, 
                        n_molecules: int = 10000,
                        include_labels: bool = True) -> pd.DataFrame:
        """
        Generate complete molecular dataset
        
        Args:
            n_molecules: Number of molecules to generate
            include_labels: Whether to include target labels (toxicity, solubility)
            
        Returns:
            DataFrame with molecular data
        """
        print(f"Generating {n_molecules} synthetic molecules...")
        
        data = []
        
        for i in range(n_molecules):
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{n_molecules} molecules...")
            
            # Generate molecule
            complexity = random.choice(['simple', 'medium', 'medium', 'complex'])
            smiles = self.generate_smiles(complexity)
            properties = self.generate_molecular_properties(smiles)
            
            # Create record
            record = {
                'molecule_id': f'MOL_{i+1:06d}',
                'smiles': smiles,
                **properties
            }
            
            # Add labels if requested
            if include_labels:
                is_toxic, tox_prob = self.simulate_toxicity(properties)
                logs = self.simulate_solubility(properties)
                activity_class, ic50 = self.simulate_bioactivity(properties)
                
                record.update({
                    'is_toxic': is_toxic,
                    'toxicity_probability': tox_prob,
                    'solubility_logs': logs,
                    'activity_class': activity_class,
                    'ic50_nm': ic50
                })
            
            # Add metadata
            record['created_at'] = datetime.now() - timedelta(days=random.randint(0, 365))
            
            data.append(record)
        
        df = pd.DataFrame(data)
        print(f"✓ Generated {len(df)} molecules with {len(df.columns)} features")
        
        return df


def generate_molecular_dataset(n_molecules: int = 10000, 
                               random_seed: int = 42) -> pd.DataFrame:
    """
    Convenience function to generate molecular dataset
    
    Args:
        n_molecules: Number of molecules to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with molecular data
    """
    generator = MolecularDataGenerator(random_seed=random_seed)
    return generator.generate_dataset(n_molecules=n_molecules)


# Example usage
if __name__ == "__main__":
    # Generate small dataset for testing
    df = generate_molecular_dataset(n_molecules=1000)
    print("\nDataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nBasic statistics:")
    print(df.describe())
