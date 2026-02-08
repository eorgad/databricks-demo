# 🚀 Quick Start Guide - Drug Discovery Demo

Get up and running with the Drug Discovery AI demo in 15 minutes!

## Prerequisites

✅ Databricks workspace (Runtime 13.3 LTS or higher)  
✅ Cluster with at least 8GB RAM per node  
✅ Basic knowledge of Python and Spark  

## Step 1: Upload to Databricks (2 minutes)

### Option A: Using Databricks Repos (Recommended)
```bash
1. In Databricks: Workspace → Repos → Add Repo
2. Enter your Git repository URL
3. Click Create Repo
```

### Option B: Manual Upload
```bash
1. Create folder in Workspace: /Workspace/Users/<your-email>/drug_discovery_demo
2. Upload all files maintaining the folder structure
```

## Step 2: Create Cluster (3 minutes)

1. **Navigate to**: Compute → Create Compute

2. **Cluster Configuration**:
   - **Name**: `drug-discovery-cluster`
   - **Cluster Mode**: Single Node (for demo) or Multi Node (for production)
   - **Databricks Runtime**: 13.3 LTS ML or higher
   - **Node Type**: Standard_DS3_v2 or larger
   - **Auto-termination**: 60 minutes

3. **Libraries to Install** (on cluster):
   - PyPI packages:
     ```
     rdkit==2023.9.1
     mlflow>=2.8.0
     hyperopt>=0.2.7
     shap>=0.42.0
     plotly>=5.17.0
     seaborn>=0.12.0
     pyyaml
     ```

4. **Start the cluster**

## Step 3: Configure Settings (2 minutes)

1. **Update config/config.yaml**:
   ```yaml
   database:
     catalog: "main"          # Change if using different catalog
     schema: "drug_discovery" # Your schema name
   
   mlflow:
     experiment_name: "/Users/<your-email>/drug_discovery_demo"  # YOUR EMAIL HERE
   ```

2. **Update notebook paths** in each notebook:
   - Search for: `/Workspace/Repos/.../drug_discovery_demo`
   - Replace with your actual path

## Step 4: Run the Pipeline (8 minutes)

Execute notebooks in order:

### 📓 Notebook 1: Data Ingestion (2-3 min)
```python
# What it does:
# - Generates 50,000 synthetic molecules
# - Performs data quality checks
# - Saves to Delta Lake
# - Creates train/val/test splits

# Expected output:
# ✓ 50,000 molecules generated
# ✓ Saved to molecules_bronze table
```

**Path**: `notebooks/01_data_ingestion.py`  
**Runtime**: ~2-3 minutes  
**Output Tables**: 
- `main.drug_discovery.molecules_bronze`
- `main.drug_discovery.molecules_bronze_train`
- `main.drug_discovery.molecules_bronze_val`
- `main.drug_discovery.molecules_bronze_test`

---

### 📓 Notebook 2: Feature Engineering (3-4 min)
```python
# What it does:
# - Calculates molecular descriptors (MW, LogP, TPSA, etc.)
# - Generates 2048-bit Morgan fingerprints
# - Validates drug-likeness (Lipinski's Rule of Five)
# - Creates feature tables

# Expected output:
# ✓ 14 descriptors + 2048 fingerprint features
# ✓ Saved to molecular_features table
```

**Path**: `notebooks/02_feature_engineering.py`  
**Runtime**: ~3-4 minutes  
**Output Tables**: 
- `main.drug_discovery.molecular_features`
- `main.drug_discovery.molecular_features_val`
- `main.drug_discovery.molecular_features_test`

---

### 📓 Notebook 3: Model Training (2-3 min)
```python
# What it does:
# - Trains toxicity classification models
# - Runs hyperparameter tuning (20 trials)
# - Evaluates with multiple metrics
# - Registers best model to MLflow

# Expected output:
# ✓ ROC-AUC: ~0.89
# ✓ Model registered: toxicity_predictor
```

**Path**: `notebooks/03_model_training_toxicity.py`  
**Runtime**: ~2-3 minutes  
**MLflow Artifacts**: Experiment tracking, model registry

---

## Step 5: Verify Success (1 minute)

### Check Data Tables
```sql
-- In Databricks SQL editor
USE CATALOG main;
USE SCHEMA drug_discovery;

-- Verify tables exist
SHOW TABLES;

-- Check molecule count
SELECT COUNT(*) FROM molecules_bronze;  -- Should be ~50,000

-- Check feature count
SELECT COUNT(*) FROM molecular_features;  -- Should be ~35,000 (train split)
```

### Check MLflow Experiments
```python
import mlflow

# View experiments
mlflow.search_experiments()

# View runs
runs = mlflow.search_runs(experiment_names=["/Users/<your-email>/drug_discovery_demo"])
print(runs[['run_id', 'metrics.test_roc_auc', 'tags.mlflow.runName']])
```

### Check Model Registry
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
models = client.search_registered_models()

for model in models:
    if model.name == "toxicity_predictor":
        print(f"✓ Model found: {model.name}")
        print(f"  Latest version: {model.latest_versions[0].version}")
```

## Expected Results

After running all notebooks, you should have:

✅ **50,000 synthetic molecules** in Delta Lake  
✅ **2,062 features per molecule** (14 descriptors + 2048 fingerprints)  
✅ **Trained ML model** with ~89% ROC-AUC  
✅ **Model registered** in MLflow Registry  
✅ **Experiment tracking** with metrics and artifacts  

## What to Demo

### 1. Show the Business Value (2 min)
```
"Traditional drug discovery screens ~100 compounds per day manually.
Our AI pipeline screens 10,000+ per day automatically.
Cost: $5,000 per lab test → $0.10 per AI prediction.
Time: 6-12 months → 2 weeks to identify lead compounds."
```

### 2. Walk Through the Pipeline (5 min)
1. **Data Ingestion**: Show Delta Lake tables, data quality checks
2. **Feature Engineering**: Explain molecular fingerprints, show descriptors
3. **Model Training**: Show MLflow experiments, model comparison
4. **Results**: Show ROC curve, feature importance, predictions

### 3. Highlight Key Technologies (2 min)
- **Delta Lake**: ACID transactions, time travel
- **Spark**: Distributed processing of millions of molecules
- **MLflow**: Experiment tracking, model registry
- **RDKit**: Industry-standard cheminformatics

### 4. Show Production Readiness (2 min)
- **Scalability**: Handles millions of molecules
- **Reproducibility**: Versioned data, tracked experiments
- **Deployment**: Model registry → REST API
- **Monitoring**: Feature drift detection, model performance

## Troubleshooting

### Issue: RDKit import error
```bash
# Solution: Install on cluster
%pip install rdkit==2023.9.1
dbutils.library.restartPython()
```

### Issue: Table not found
```python
# Solution: Check catalog/schema
spark.sql("USE CATALOG main")
spark.sql("USE SCHEMA drug_discovery")
```

### Issue: Out of memory
```python
# Solution: Increase cluster size or reduce data
config['data_generation']['n_molecules'] = 10000  # Reduce from 50,000
```

### Issue: MLflow experiment not found
```python
# Solution: Create experiment manually
mlflow.create_experiment("/Users/<your-email>/drug_discovery_demo")
```

## Next Steps

Once the demo is running:

1. **Customize the data**: 
   - Connect to real molecular databases (PubChem, ChEMBL)
   - Load your internal compound library

2. **Improve models**:
   - Try deep learning (Graph Neural Networks)
   - Add more target properties (solubility, bioactivity)
   - Ensemble multiple models

3. **Deploy to production**:
   - Set up batch inference pipeline
   - Create REST API endpoint
   - Build monitoring dashboard

4. **Scale up**:
   - Process millions of molecules
   - Multi-task learning (predict multiple properties)
   - Active learning loop

## Support

- **Documentation**: See [README.md](README.md) for full details
- **Issues**: Check notebook comments for troubleshooting
- **Questions**: Reach out to your Databricks team

---

**Time to Value**: 15 minutes from zero to working ML pipeline! 🎉
