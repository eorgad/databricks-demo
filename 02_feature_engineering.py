# Databricks notebook source
# MAGIC %md
# MAGIC # 🧬 Drug Discovery Demo - Part 2: Feature Engineering
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook demonstrates molecular feature engineering:
# MAGIC - Calculate molecular descriptors using RDKit
# MAGIC - Generate Morgan fingerprints
# MAGIC - Compute physicochemical properties
# MAGIC - Create Feature Store tables (Silver layer)
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Use RDKit for cheminformatics
# MAGIC - Distributed feature calculation with Pandas UDF
# MAGIC - Feature Store integration
# MAGIC - Feature quality validation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, struct
from pyspark.sql.types import *
import warnings
warnings.filterwarnings('ignore')

# Install RDKit if not available
try:
    from rdkit import Chem
    print("✓ RDKit already installed")
except ImportError:
    print("Installing RDKit...")
    %pip install rdkit
    from rdkit import Chem
    print("✓ RDKit installed successfully")

print("✓ Libraries imported")

# COMMAND ----------

# Configuration
CATALOG = "main"
SCHEMA = "drug_discovery"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.molecules_bronze_train"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.molecular_features"

# Feature parameters
FINGERPRINT_RADIUS = 2
FINGERPRINT_NBITS = 2048

print(f"Configuration:")
print(f"  Source: {SOURCE_TABLE}")
print(f"  Target: {TARGET_TABLE}")
print(f"  FP Radius: {FINGERPRINT_RADIUS}")
print(f"  FP Bits: {FINGERPRINT_NBITS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Data

# COMMAND ----------

# Load training data
df = spark.table(SOURCE_TABLE)
print(f"✓ Loaded {df.count():,} molecules")

# Show sample
display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Calculate Molecular Descriptors
# MAGIC
# MAGIC We'll calculate key molecular descriptors using RDKit:
# MAGIC - Molecular weight
# MAGIC - LogP (lipophilicity)
# MAGIC - TPSA (topological polar surface area)
# MAGIC - H-bond donors/acceptors
# MAGIC - Rotatable bonds
# MAGIC - Aromatic rings
# MAGIC - And more...

# COMMAND ----------

# DBTITLE 1,Define Descriptor Calculation Function
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

def calculate_molecular_descriptors(smiles: str) -> dict:
    """Calculate molecular descriptors from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            'molecular_weight': float(Descriptors.MolWt(mol)),
            'logp': float(Crippen.MolLogP(mol)),
            'tpsa': float(Descriptors.TPSA(mol)),
            'num_h_donors': int(Lipinski.NumHDonors(mol)),
            'num_h_acceptors': int(Lipinski.NumHAcceptors(mol)),
            'num_rotatable_bonds': int(Lipinski.NumRotatableBonds(mol)),
            'num_aromatic_rings': int(Lipinski.NumAromaticRings(mol)),
            'num_heteroatoms': int(Lipinski.NumHeteroatoms(mol)),
            'num_rings': int(rdMolDescriptors.CalcNumRings(mol)),
            'num_saturated_rings': int(rdMolDescriptors.CalcNumSaturatedRings(mol)),
            'fraction_csp3': float(rdMolDescriptors.CalcFractionCsp3(mol)),
            'num_aliphatic_rings': int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
            'num_stereo_centers': int(rdMolDescriptors.CalcNumAtomStereoCenters(mol)),
            'num_unspecified_stereo_centers': int(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol))
        }
    except:
        return None

# Test the function
test_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
test_result = calculate_molecular_descriptors(test_smiles)
print("Test calculation for Aspirin:")
print(test_result)

# COMMAND ----------

# DBTITLE 1,Create Pandas UDF for Distributed Processing
# Define schema for descriptor output
descriptor_schema = StructType([
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
    StructField("num_aliphatic_rings", IntegerType()),
    StructField("num_stereo_centers", IntegerType()),
    StructField("num_unspecified_stereo_centers", IntegerType())
])

@pandas_udf(descriptor_schema)
def calculate_descriptors_udf(smiles_series: pd.Series) -> pd.DataFrame:
    """Pandas UDF for batch descriptor calculation"""
    results = []
    for smiles in smiles_series:
        desc = calculate_molecular_descriptors(smiles)
        if desc:
            results.append(desc)
        else:
            # Return NaN for invalid molecules
            results.append({k: None for k in descriptor_schema.fieldNames()})
    return pd.DataFrame(results)

print("✓ Pandas UDF defined for distributed processing")

# COMMAND ----------

# DBTITLE 1,Calculate Descriptors for All Molecules
print("Calculating molecular descriptors...")
print("This will be distributed across Spark cluster...")

# Apply UDF to calculate descriptors
df_with_descriptors = df.withColumn(
    "descriptors",
    calculate_descriptors_udf(col("smiles"))
)

# Expand struct into individual columns
for field in descriptor_schema.fields:
    df_with_descriptors = df_with_descriptors.withColumn(
        f"calc_{field.name}",
        col(f"descriptors.{field.name}")
    )

df_with_descriptors = df_with_descriptors.drop("descriptors")

# Remove .cache() call (not supported on serverless)
count = df_with_descriptors.count()

print(f"✓ Calculated descriptors for {count:,} molecules")

# Show sample
display(df_with_descriptors.select("smiles", "calc_molecular_weight", "calc_logp", "calc_tpsa").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Generate Morgan Fingerprints
# MAGIC
# MAGIC Morgan fingerprints (also called circular fingerprints) encode molecular structure as binary vectors.
# MAGIC These are essential features for machine learning models.

# COMMAND ----------

# DBTITLE 1,Fingerprint Generation Function
from rdkit.Chem import AllChem

def generate_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> list:
    """Generate Morgan fingerprint as list of bits"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * n_bits
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return [int(b) for b in fp]
    except:
        return [0] * n_bits

# Test
test_fp = generate_morgan_fingerprint("CC(=O)Oc1ccccc1C(=O)O", FINGERPRINT_RADIUS, FINGERPRINT_NBITS)
print(f"Generated fingerprint with {len(test_fp)} bits")
print(f"Number of 'on' bits: {sum(test_fp)}")

# COMMAND ----------

# DBTITLE 1,Generate Fingerprints (Distributed)
# Define schema for fingerprint array
from pyspark.sql.types import ArrayType, IntegerType

@pandas_udf(ArrayType(IntegerType()))
def generate_fingerprint_udf(smiles_series: pd.Series) -> pd.Series:
    """Pandas UDF for fingerprint generation"""
    return smiles_series.apply(
        lambda s: generate_morgan_fingerprint(s, FINGERPRINT_RADIUS, FINGERPRINT_NBITS)
    )

print("Generating Morgan fingerprints...")

# Add fingerprint column
df_with_features = df_with_descriptors.withColumn(
    "morgan_fp",
    generate_fingerprint_udf(col("smiles"))
)

print("✓ Fingerprints generated")

# Show sample
display(df_with_features.select("smiles", "morgan_fp").limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature Quality Validation

# COMMAND ----------

# DBTITLE 1,Check for Invalid Descriptors
# Count null/invalid values
from pyspark.sql.functions import when, count, col

null_counts = df_with_features.select([
    count(when(col(c).isNull(), c)).alias(c) 
    for c in df_with_features.columns if c.startswith('calc_')
])

print("Null value counts in calculated features:")
display(null_counts)

# COMMAND ----------

# DBTITLE 1,Feature Statistics
# Calculate feature statistics
feature_cols = [c for c in df_with_features.columns if c.startswith('calc_')]

stats_df = df_with_features.select(feature_cols).describe()
display(stats_df)

# COMMAND ----------

# DBTITLE 1,Lipinski Rule of Five Check
from pyspark.sql.functions import when

# Calculate Lipinski violations
df_with_features = df_with_features.withColumn(
    "lipinski_mw_violation",
    when(col("calc_molecular_weight") > 500, 1).otherwise(0)
).withColumn(
    "lipinski_logp_violation",
    when(col("calc_logp") > 5, 1).otherwise(0)
).withColumn(
    "lipinski_hbd_violation",
    when(col("calc_num_h_donors") > 5, 1).otherwise(0)
).withColumn(
    "lipinski_hba_violation",
    when(col("calc_num_h_acceptors") > 10, 1).otherwise(0)
).withColumn(
    "lipinski_violations",
    col("lipinski_mw_violation") + 
    col("lipinski_logp_violation") + 
    col("lipinski_hbd_violation") + 
    col("lipinski_hba_violation")
).withColumn(
    "is_drug_like",
    when(col("lipinski_violations") <= 1, True).otherwise(False)
)

# Show Lipinski compliance summary
lipinski_summary = df_with_features.groupBy("lipinski_violations").count().orderBy("lipinski_violations")
print("Lipinski Rule of Five Compliance:")
display(lipinski_summary)

drug_like_pct = df_with_features.filter(col("is_drug_like") == True).count() / df_with_features.count() * 100
print(f"\nDrug-like molecules: {drug_like_pct:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Feature Visualization

# COMMAND ----------

# DBTITLE 1,Descriptor Distributions
# Sample data for visualization (take 10,000 rows)
sample_df = df_with_features.sample(fraction=0.2).toPandas()

# Plot key descriptor distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

descriptors_to_plot = [
    ('calc_molecular_weight', 'Calculated Molecular Weight'),
    ('calc_logp', 'Calculated LogP'),
    ('calc_tpsa', 'Calculated TPSA'),
    ('calc_num_h_donors', 'Calculated H-Donors'),
    ('calc_num_h_acceptors', 'Calculated H-Acceptors'),
    ('calc_num_rotatable_bonds', 'Calculated Rotatable Bonds')
]

for idx, (col_name, title) in enumerate(descriptors_to_plot):
    axes[idx].hist(sample_df[col_name].dropna(), bins=40, color='steelblue', alpha=0.7, edgecolor='black')
    axes[idx].set_title(title, fontweight='bold')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    
    mean_val = sample_df[col_name].mean()
    axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[idx].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Fingerprint Sparsity Analysis
# Analyze fingerprint sparsity
sample_fps = sample_df['morgan_fp'].apply(lambda x: np.array(x))
fp_matrix = np.vstack(sample_fps)

sparsity = 1 - (fp_matrix.sum() / fp_matrix.size)
avg_bits_on = fp_matrix.sum(axis=1).mean()

print(f"Fingerprint Statistics:")
print(f"  Sparsity: {sparsity*100:.2f}%")
print(f"  Average bits 'on' per molecule: {avg_bits_on:.1f} / {FINGERPRINT_NBITS}")

# Plot bit usage
bit_usage = fp_matrix.sum(axis=0)
plt.figure(figsize=(12, 4))
plt.plot(bit_usage, alpha=0.7)
plt.title('Morgan Fingerprint Bit Usage Across Molecules', fontweight='bold')
plt.xlabel('Bit Index')
plt.ylabel('Number of Molecules with Bit Set')
plt.grid(alpha=0.3)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save Feature Tables

# COMMAND ----------

# DBTITLE 1,Save to Delta Lake (Silver Layer)
print(f"Writing features to Delta Lake: {TARGET_TABLE}")

# Write feature table
(df_with_features
 .write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(TARGET_TABLE))

row_count = spark.table(TARGET_TABLE).count()
print(f"✓ Saved {row_count:,} rows to {TARGET_TABLE}")

# COMMAND ----------

# DBTITLE 1,Process Validation and Test Sets
# Also process validation and test sets
for split in ['val', 'test']:
    source = f"{CATALOG}.{SCHEMA}.molecules_bronze_{split}"
    target = f"{CATALOG}.{SCHEMA}.molecular_features_{split}"
    
    print(f"\nProcessing {split} set...")
    
    df_split = spark.table(source)
    
    # Calculate descriptors
    df_split = df_split.withColumn("descriptors", calculate_descriptors_udf(col("smiles")))
    for field in descriptor_schema.fields:
        df_split = df_split.withColumn(f"calc_{field.name}", col(f"descriptors.{field.name}"))
    df_split = df_split.drop("descriptors")
    
    # Generate fingerprints
    df_split = df_split.withColumn("morgan_fp", generate_fingerprint_udf(col("smiles")))
    
    # Add Lipinski checks
    df_split = (df_split
        .withColumn("lipinski_mw_violation", when(col("calc_molecular_weight") > 500, 1).otherwise(0))
        .withColumn("lipinski_logp_violation", when(col("calc_logp") > 5, 1).otherwise(0))
        .withColumn("lipinski_hbd_violation", when(col("calc_num_h_donors") > 5, 1).otherwise(0))
        .withColumn("lipinski_hba_violation", when(col("calc_num_h_acceptors") > 10, 1).otherwise(0))
        .withColumn("lipinski_violations", 
                   col("lipinski_mw_violation") + col("lipinski_logp_violation") + 
                   col("lipinski_hbd_violation") + col("lipinski_hba_violation"))
        .withColumn("is_drug_like", when(col("lipinski_violations") <= 1, True).otherwise(False))
    )
    
    # Save
    df_split.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(target)
    
    count = spark.table(target).count()
    print(f"✓ Saved {count:,} rows to {target}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary

# COMMAND ----------

print("="*70)
print("FEATURE ENGINEERING COMPLETE")
print("="*70)

print(f"\n📊 Features Created:")
print(f"  • Molecular descriptors: {len([c for c in df_with_features.columns if c.startswith('calc_')])}")
print(f"  • Fingerprint bits: {FINGERPRINT_NBITS}")
print(f"  • Total features: {len([c for c in df_with_features.columns if c.startswith('calc_')]) + FINGERPRINT_NBITS}")

print(f"\n💾 Feature Tables:")
print(f"  • Training: {TARGET_TABLE}")
print(f"  • Validation: {TARGET_TABLE}_val")
print(f"  • Test: {TARGET_TABLE}_test")

print(f"\n✅ Quality Metrics:")
print(f"  • Drug-like molecules: {drug_like_pct:.1f}%")
print(f"  • Valid descriptors: {100 - (null_counts.collect()[0].asDict().get('calc_molecular_weight', 0) / count * 100):.1f}%")
print(f"  • Fingerprint sparsity: {sparsity*100:.1f}%")

print(f"\n➡️  Next Steps:")
print(f"  1. Run Notebook 03: Toxicity Model Training")
print(f"  2. Train classification models")
print(f"  3. Evaluate model performance")
print(f"  4. Register models to MLflow")

print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **End of Notebook 02: Feature Engineering**
# MAGIC
# MAGIC Continue to [Notebook 03: Toxicity Model Training](...)
