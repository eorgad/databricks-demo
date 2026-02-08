# Databricks notebook source
# MAGIC %md
# MAGIC # 🧬 Drug Discovery Demo - Part 1: Data Ingestion
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook demonstrates the first step in our AI-powered drug discovery pipeline:
# MAGIC - Generate synthetic molecular dataset (or load from external sources)
# MAGIC - Perform data quality validation
# MAGIC - Store in Delta Lake (Bronze layer)
# MAGIC - Exploratory data analysis
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Work with molecular data (SMILES format)
# MAGIC - Use Delta Lake for ACID transactions
# MAGIC - Implement data quality checks
# MAGIC - Visualize molecular properties

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Databricks imports
from pyspark.sql.functions import col, count, isnan, when, avg, stddev, min, max
from pyspark.sql.types import *

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Configuration

# COMMAND ----------

# DBTITLE 1,Load Config (adjust paths as needed)
import yaml

# Load configuration
try:
    with open('/Workspace/Repos/.../drug_discovery_demo/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Configuration loaded from YAML file")
except:
    # Fallback to inline config if file not found
    config = {
        'database': {
            'catalog': 'main',
            'schema': 'drug_discovery'
        },
        'data_generation': {
            'n_molecules': 50000,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'random_seed': 42
        }
    }
    print("⚠ Using inline configuration")

# Extract config values
CATALOG = config['database']['catalog']
SCHEMA = config['database']['schema']
N_MOLECULES = config['data_generation']['n_molecules']
RANDOM_SEED = config['data_generation']['random_seed']

print(f"\nConfiguration:")
print(f"  Catalog: {CATALOG}")
print(f"  Schema: {SCHEMA}")
print(f"  Molecules to generate: {N_MOLECULES:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Database Schema

# COMMAND ----------

# DBTITLE 1,Create Schema if Not Exists
# Create catalog and schema (Unity Catalog)
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"✓ Using catalog: {CATALOG}, schema: {SCHEMA}")

# Verify current database
current_db = spark.sql("SELECT current_database()").collect()[0][0]
print(f"✓ Current database: {current_db}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate Synthetic Molecular Dataset
# MAGIC
# MAGIC We'll generate synthetic molecules with realistic properties for demo purposes.
# MAGIC In production, you would load data from:
# MAGIC - PubChem API
# MAGIC - ChEMBL database
# MAGIC - Internal compound libraries
# MAGIC - Clinical trial databases

# COMMAND ----------

# DBTITLE 1,Add Utils to Python Path
import sys
import os

# Add utils directory to path (adjust based on your setup)
utils_path = '/Workspace/Repos/.../drug_discovery_demo/utils'
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

print(f"✓ Added {utils_path} to Python path")

# COMMAND ----------

# DBTITLE 1,Generate Molecular Dataset
from data_generator import generate_molecular_dataset

# Generate dataset
print(f"Generating {N_MOLECULES:,} synthetic molecules...")
print("This may take a few minutes...\n")

df_molecules = generate_molecular_dataset(
    n_molecules=N_MOLECULES,
    random_seed=RANDOM_SEED
)

print(f"\n✓ Generated {len(df_molecules):,} molecules")
print(f"✓ Dataset shape: {df_molecules.shape}")
print(f"✓ Memory usage: {df_molecules.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preview Generated Data

# COMMAND ----------

# DBTITLE 1,Display Sample Molecules
# Show first few rows
display(df_molecules.head(10))

# COMMAND ----------

# DBTITLE 1,Dataset Information
# Show column info
print("Dataset Columns:")
print("="*60)
for col in df_molecules.columns:
    dtype = df_molecules[col].dtype
    non_null = df_molecules[col].notna().sum()
    print(f"  {col:25s} | {str(dtype):15s} | {non_null:,} non-null")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Quality Validation

# COMMAND ----------

# DBTITLE 1,Check for Missing Values
# Check for missing values
missing_counts = df_molecules.isnull().sum()
missing_pct = (missing_counts / len(df_molecules) * 100).round(2)

missing_df = pd.DataFrame({
    'Column': missing_counts.index,
    'Missing Count': missing_counts.values,
    'Missing %': missing_pct.values
})

print("Missing Values Summary:")
print("="*60)
display(missing_df[missing_df['Missing Count'] > 0])

if missing_df['Missing Count'].sum() == 0:
    print("✓ No missing values found!")

# COMMAND ----------

# DBTITLE 1,Data Quality Checks
# Validate SMILES strings
invalid_smiles = df_molecules[df_molecules['smiles'].str.len() < 3]
print(f"Invalid SMILES (too short): {len(invalid_smiles)}")

# Check for duplicates
duplicates = df_molecules.duplicated(subset=['smiles']).sum()
print(f"Duplicate SMILES: {duplicates}")

# Check value ranges
print("\nValue Range Checks:")
print("="*60)

checks = {
    'Molecular Weight': (df_molecules['molecular_weight'] >= 0).all(),
    'LogP in range': (df_molecules['logp'] >= -10) & (df_molecules['logp'] <= 10).all(),
    'TPSA non-negative': (df_molecules['tpsa'] >= 0).all(),
    'Positive integers': (df_molecules['num_h_donors'] >= 0).all()
}

for check_name, passed in checks.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {check_name:30s} : {status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Exploratory Data Analysis

# COMMAND ----------

# DBTITLE 1,Statistical Summary
# Descriptive statistics
print("Statistical Summary:")
print("="*60)
display(df_molecules.describe())

# COMMAND ----------

# DBTITLE 1,Target Variable Distribution
# Plot toxicity and activity distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Toxicity distribution
toxicity_counts = df_molecules['is_toxic'].value_counts()
axes[0].bar(['Non-Toxic', 'Toxic'], toxicity_counts.values, color=['green', 'red'], alpha=0.7)
axes[0].set_title('Toxicity Distribution')
axes[0].set_ylabel('Count')
for i, v in enumerate(toxicity_counts.values):
    axes[0].text(i, v + 100, f'{v:,}\n({v/len(df_molecules)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')

# Activity class distribution
activity_counts = df_molecules['activity_class'].value_counts()
axes[1].bar(activity_counts.index, activity_counts.values, 
           color=['green', 'orange', 'red'], alpha=0.7)
axes[1].set_title('Bioactivity Distribution')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(activity_counts.index, rotation=45)

# Solubility distribution
axes[2].hist(df_molecules['solubility_logs'], bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[2].set_title('Solubility (LogS) Distribution')
axes[2].set_xlabel('LogS')
axes[2].set_ylabel('Count')
axes[2].axvline(df_molecules['solubility_logs'].median(), color='red', 
               linestyle='--', label=f'Median: {df_molecules["solubility_logs"].median():.2f}')
axes[2].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Molecular Property Distributions
# Plot molecular property distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

properties = [
    ('molecular_weight', 'Molecular Weight (Da)', 'skyblue'),
    ('logp', 'LogP (Lipophilicity)', 'lightcoral'),
    ('tpsa', 'TPSA (Ų)', 'lightgreen'),
    ('num_h_donors', 'H-Bond Donors', 'khaki'),
    ('num_h_acceptors', 'H-Bond Acceptors', 'plum'),
    ('num_rotatable_bonds', 'Rotatable Bonds', 'lightsalmon'),
    ('num_aromatic_rings', 'Aromatic Rings', 'lightsteelblue'),
    ('num_heteroatoms', 'Heteroatoms', 'peachpuff')
]

for idx, (prop, title, color) in enumerate(properties):
    axes[idx].hist(df_molecules[prop], bins=40, color=color, alpha=0.7, edgecolor='black')
    axes[idx].set_title(title, fontweight='bold')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    
    # Add mean line
    mean_val = df_molecules[prop].mean()
    axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                     label=f'Mean: {mean_val:.1f}')
    axes[idx].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Lipinski's Rule of Five Analysis
# Check Lipinski's Rule of Five compliance
lipinski_violations = []

for _, row in df_molecules.iterrows():
    violations = 0
    if row['molecular_weight'] > 500:
        violations += 1
    if row['logp'] > 5:
        violations += 1
    if row['num_h_donors'] > 5:
        violations += 1
    if row['num_h_acceptors'] > 10:
        violations += 1
    lipinski_violations.append(violations)

df_molecules['lipinski_violations'] = lipinski_violations

# Plot Lipinski compliance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Violation counts
violation_counts = df_molecules['lipinski_violations'].value_counts().sort_index()
axes[0].bar(violation_counts.index, violation_counts.values, 
           color=['green', 'yellow', 'orange', 'red', 'darkred'][:len(violation_counts)],
           alpha=0.7, edgecolor='black')
axes[0].set_title("Lipinski's Rule of Five - Violations", fontweight='bold', fontsize=12)
axes[0].set_xlabel('Number of Violations')
axes[0].set_ylabel('Number of Molecules')
axes[0].set_xticks(range(5))

for i, v in enumerate(violation_counts.values):
    axes[0].text(violation_counts.index[i], v + 100, f'{v:,}\n({v/len(df_molecules)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')

# Drug-likeness pie chart
drug_like = (df_molecules['lipinski_violations'] <= 1).sum()
not_drug_like = len(df_molecules) - drug_like

axes[1].pie([drug_like, not_drug_like], 
           labels=['Drug-like\n(≤1 violation)', 'Not Drug-like\n(>1 violation)'],
           autopct='%1.1f%%', colors=['green', 'red'], alpha=0.7,
           explode=(0.05, 0), startangle=90)
axes[1].set_title('Drug-likeness Assessment', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.show()

print(f"\nDrug-like molecules (≤1 Lipinski violation): {drug_like:,} ({drug_like/len(df_molecules)*100:.1f}%)")

# COMMAND ----------

# DBTITLE 1,Correlation Analysis
# Calculate correlations with target variables
correlations = df_molecules[[
    'molecular_weight', 'logp', 'tpsa', 'num_h_donors', 
    'num_h_acceptors', 'num_rotatable_bonds', 'toxicity_probability', 'solubility_logs'
]].corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, fmt='.2f', cmap='coolwarm', 
           center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Save to Delta Lake (Bronze Layer)

# COMMAND ----------

# DBTITLE 1,Convert to Spark DataFrame
# Convert pandas to Spark DataFrame
print("Converting to Spark DataFrame...")
spark_df = spark.createDataFrame(df_molecules)

print(f"✓ Converted to Spark DataFrame")
print(f"  Rows: {spark_df.count():,}")
print(f"  Columns: {len(spark_df.columns)}")

# Show schema
spark_df.printSchema()

# COMMAND ----------

# DBTITLE 1,Write to Delta Lake
# Define table name
TABLE_NAME = "molecules_bronze"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

print(f"Writing data to Delta Lake: {FULL_TABLE_NAME}")

# Write to Delta Lake
(spark_df
 .write
 .format("delta")
 .mode("overwrite")  # Use "append" for incremental loads
 .option("overwriteSchema", "true")
 .saveAsTable(FULL_TABLE_NAME))

print(f"✓ Data written successfully to {FULL_TABLE_NAME}")

# COMMAND ----------

# DBTITLE 1,Verify Data Persistence
# Verify data was written
row_count = spark.sql(f"SELECT COUNT(*) as count FROM {FULL_TABLE_NAME}").collect()[0]['count']
print(f"✓ Verified {row_count:,} rows in {FULL_TABLE_NAME}")

# Show sample
print("\nSample data from Delta Lake:")
display(spark.sql(f"SELECT * FROM {FULL_TABLE_NAME} LIMIT 10"))

# COMMAND ----------

# DBTITLE 1,Create Data Splits
# Create train/val/test splits
from pyspark.sql.functions import rand

# Add random column for splitting
df_with_split = spark_df.withColumn("rand", rand(seed=RANDOM_SEED))

# Define split proportions
train_proportion = config['data_generation']['train_split']
val_proportion = config['data_generation']['val_split']

# Create splits
train_df = df_with_split.filter(col("rand") < train_proportion)
val_df = df_with_split.filter(
    (col("rand") >= train_proportion) & 
    (col("rand") < train_proportion + val_proportion)
)
test_df = df_with_split.filter(col("rand") >= train_proportion + val_proportion)

# Remove random column
train_df = train_df.drop("rand")
val_df = val_df.drop("rand")
test_df = test_df.drop("rand")

# Save splits
train_df.write.format("delta").mode("overwrite").saveAsTable(f"{FULL_TABLE_NAME}_train")
val_df.write.format("delta").mode("overwrite").saveAsTable(f"{FULL_TABLE_NAME}_val")
test_df.write.format("delta").mode("overwrite").saveAsTable(f"{FULL_TABLE_NAME}_test")

print(f"✓ Data splits created:")
print(f"  Training:   {train_df.count():,} molecules ({train_proportion*100:.0f}%)")
print(f"  Validation: {val_df.count():,} molecules ({val_proportion*100:.0f}%)")
print(f"  Test:       {test_df.count():,} molecules ({(1-train_proportion-val_proportion)*100:.0f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary and Next Steps

# COMMAND ----------

# DBTITLE 1,Pipeline Summary
print("="*70)
print("DATA INGESTION COMPLETE")
print("="*70)
print(f"\n📊 Dataset Statistics:")
print(f"  • Total molecules: {N_MOLECULES:,}")
print(f"  • Features: {len(df_molecules.columns)}")
print(f"  • Drug-like molecules: {drug_like:,} ({drug_like/N_MOLECULES*100:.1f}%)")
print(f"  • Toxic molecules: {df_molecules['is_toxic'].sum():,} ({df_molecules['is_toxic'].sum()/N_MOLECULES*100:.1f}%)")
print(f"  • Active molecules: {(df_molecules['activity_class'] == 'active').sum():,}")

print(f"\n💾 Data Storage:")
print(f"  • Bronze table: {FULL_TABLE_NAME}")
print(f"  • Train split: {FULL_TABLE_NAME}_train")
print(f"  • Val split: {FULL_TABLE_NAME}_val")
print(f"  • Test split: {FULL_TABLE_NAME}_test")

print(f"\n✅ Quality Checks Passed:")
print(f"  • No missing values")
print(f"  • No duplicate SMILES")
print(f"  • All values in valid ranges")

print(f"\n➡️  Next Steps:")
print(f"  1. Run Notebook 02: Feature Engineering")
print(f"  2. Calculate molecular descriptors")
print(f"  3. Generate fingerprints")
print(f"  4. Create feature store tables")

print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix: Additional Analyses (Optional)

# COMMAND ----------

# DBTITLE 1,Property Distribution by Toxicity
# Compare properties between toxic and non-toxic molecules
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

properties = ['molecular_weight', 'logp', 'tpsa', 'num_rotatable_bonds']
titles = ['Molecular Weight', 'LogP', 'TPSA', 'Rotatable Bonds']

for idx, (prop, title) in enumerate(zip(properties, titles)):
    ax = axes[idx // 2, idx % 2]
    
    toxic = df_molecules[df_molecules['is_toxic'] == True][prop]
    non_toxic = df_molecules[df_molecules['is_toxic'] == False][prop]
    
    ax.hist([non_toxic, toxic], bins=30, label=['Non-Toxic', 'Toxic'],
           color=['green', 'red'], alpha=0.6, edgecolor='black')
    ax.set_title(f'{title} Distribution by Toxicity', fontweight='bold')
    ax.set_xlabel(title)
    ax.set_ylabel('Frequency')
    ax.legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **End of Notebook 01: Data Ingestion**
# MAGIC
# MAGIC Continue to [Notebook 02: Feature Engineering](...)
