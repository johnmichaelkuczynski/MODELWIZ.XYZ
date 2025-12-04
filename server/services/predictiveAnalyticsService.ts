import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";

interface PredictiveAnalyticsParameters {
  problemType: "binary_classification" | "multiclass_classification" | "regression" | "ranking" | "auto";
  targetVariable: string;
  dataSource: string;
  businessContext: string;
  featureColumns?: string[];
  idColumn?: string;
  dateColumn?: string;
  testSize: number;
  validationStrategy: "holdout" | "kfold" | "stratified_kfold" | "time_series";
  optimizationMetric: string;
  classWeight: "balanced" | "none" | "auto";
  featureEngineeringLevel: "minimal" | "standard" | "aggressive";
  interpretabilityLevel: "basic" | "full";
  ensembleMethods: boolean;
  businessThreshold: number;
  positiveClassLabel?: string;
  handleImbalance: boolean;
  imbalanceStrategy: "smote" | "class_weight" | "undersample";
  hyperparameterTuning: "none" | "random" | "grid" | "optuna";
  tuningTrials: number;
}

interface PredictiveAnalyticsResult {
  success: boolean;
  parameters: PredictiveAnalyticsParameters;
  pythonCode: string;
  providerUsed: string;
  error?: string;
}

const PARSING_PROMPT = `You are an expert at extracting predictive analytics parameters from natural language descriptions.

Given a business problem description, extract the following parameters for an end-to-end ML pipeline:

REQUIRED:
- problemType: "binary_classification" | "multiclass_classification" | "regression" | "ranking" | "auto"
- targetVariable: The outcome variable to predict
- dataSource: Description of data source (file path, "synthetic", or inline data description)
- businessContext: Brief summary of the business problem

OPTIONAL (use intelligent defaults):
- featureColumns: Array of feature names (default: null, meaning all non-target columns)
- idColumn: Column name for unique identifiers to exclude (default: null)
- dateColumn: Date column for time-based splits (default: null)
- testSize: Holdout test proportion (default: 0.2)
- validationStrategy: "holdout" | "kfold" | "stratified_kfold" | "time_series" (default: based on problem)
- optimizationMetric: Metric to optimize (default: based on problem type)
- classWeight: "balanced" | "none" | "auto" (default: "auto")
- featureEngineeringLevel: "minimal" | "standard" | "aggressive" (default: "standard")
- interpretabilityLevel: "basic" | "full" (default: "full")
- ensembleMethods: Whether to include stacking/voting (default: true)
- businessThreshold: Decision threshold for classification (default: 0.5)
- positiveClassLabel: Label for positive class (default: null)
- handleImbalance: Whether to handle class imbalance (default: auto-detect from description)
- imbalanceStrategy: "smote" | "class_weight" | "undersample" (default: "smote")
- hyperparameterTuning: "none" | "random" | "grid" | "optuna" (default: "optuna")
- tuningTrials: Number of Optuna trials (default: 50)

PARSING RULES:
- "predict whether", "will they", "yes or no", "churn", "fraud", "convert" → binary_classification
- "which category", "classify into", "segment into groups", "sentiment" → multiclass_classification
- "predict how much", "forecast value", "estimate amount", "price", "revenue" → regression
- "score", "rank", "prioritize", "lead scoring" → ranking (treat as regression)
- "imbalanced", "rare event", "few positives" → handleImbalance: true
- "explainable", "interpretable", "understand why" → interpretabilityLevel: "full"
- "production", "deploy", "real-time" → Include deployment focus
- "aggressive feature engineering" → featureEngineeringLevel: "aggressive"
- "minimal preprocessing" → featureEngineeringLevel: "minimal"

DEFAULT METRICS BY PROBLEM TYPE:
- binary_classification: "roc_auc"
- multiclass_classification: "f1_weighted"
- regression/ranking: "rmse"

Respond with ONLY valid JSON, no markdown or explanation:
{
  "problemType": "...",
  "targetVariable": "...",
  "dataSource": "...",
  "businessContext": "...",
  "featureColumns": null,
  "idColumn": null,
  "dateColumn": null,
  "testSize": 0.2,
  "validationStrategy": "stratified_kfold",
  "optimizationMetric": "roc_auc",
  "classWeight": "auto",
  "featureEngineeringLevel": "standard",
  "interpretabilityLevel": "full",
  "ensembleMethods": true,
  "businessThreshold": 0.5,
  "positiveClassLabel": null,
  "handleImbalance": false,
  "imbalanceStrategy": "smote",
  "hyperparameterTuning": "optuna",
  "tuningTrials": 50
}`;

async function parseWithGrok(description: string): Promise<PredictiveAnalyticsParameters | null> {
  const apiKey = process.env.GROK_API_KEY;
  if (!apiKey) return null;

  try {
    const openai = new OpenAI({
      apiKey,
      baseURL: "https://api.x.ai/v1",
    });

    const response = await openai.chat.completions.create({
      model: "grok-3-latest",
      messages: [
        { role: "system", content: PARSING_PROMPT },
        { role: "user", content: `Parse this predictive analytics request:\n\n${description}` },
      ],
      temperature: 0.1,
      max_tokens: 2000,
    });

    const content = response.choices[0]?.message?.content || "";
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
  } catch (error) {
    console.error("Grok parsing failed:", error);
  }
  return null;
}

async function parseWithOpenAI(description: string): Promise<PredictiveAnalyticsParameters | null> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return null;

  try {
    const openai = new OpenAI({ apiKey });
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: PARSING_PROMPT },
        { role: "user", content: `Parse this predictive analytics request:\n\n${description}` },
      ],
      temperature: 0.1,
      max_tokens: 2000,
    });

    const content = response.choices[0]?.message?.content || "";
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
  } catch (error) {
    console.error("OpenAI parsing failed:", error);
  }
  return null;
}

async function parseWithAnthropic(description: string): Promise<PredictiveAnalyticsParameters | null> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return null;

  try {
    const anthropic = new Anthropic({ apiKey });
    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 2000,
      messages: [
        { role: "user", content: `${PARSING_PROMPT}\n\nParse this predictive analytics request:\n\n${description}` },
      ],
    });

    const content = response.content[0].type === "text" ? response.content[0].text : "";
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
  } catch (error) {
    console.error("Anthropic parsing failed:", error);
  }
  return null;
}

export async function parsePredictiveDescription(description: string): Promise<PredictiveAnalyticsResult> {
  let params: PredictiveAnalyticsParameters | null = null;
  let providerUsed = "";

  // Try Grok first (ZHI 5 - default for data science)
  params = await parseWithGrok(description);
  if (params) {
    providerUsed = "ZHI 5 (Grok)";
  }

  // Fallback to OpenAI
  if (!params) {
    params = await parseWithOpenAI(description);
    if (params) {
      providerUsed = "ZHI 1 (OpenAI)";
    }
  }

  // Fallback to Anthropic
  if (!params) {
    params = await parseWithAnthropic(description);
    if (params) {
      providerUsed = "ZHI 2 (Anthropic)";
    }
  }

  if (!params) {
    return {
      success: false,
      parameters: {} as PredictiveAnalyticsParameters,
      pythonCode: "",
      providerUsed: "",
      error: "Failed to parse predictive analytics parameters from all providers",
    };
  }

  // Apply defaults
  const finalParams: PredictiveAnalyticsParameters = {
    problemType: params.problemType || "auto",
    targetVariable: params.targetVariable || "target",
    dataSource: params.dataSource || "synthetic",
    businessContext: params.businessContext || description,
    featureColumns: params.featureColumns || undefined,
    idColumn: params.idColumn || undefined,
    dateColumn: params.dateColumn || undefined,
    testSize: params.testSize || 0.2,
    validationStrategy: params.validationStrategy || "stratified_kfold",
    optimizationMetric: params.optimizationMetric || getDefaultMetric(params.problemType),
    classWeight: params.classWeight || "auto",
    featureEngineeringLevel: params.featureEngineeringLevel || "standard",
    interpretabilityLevel: params.interpretabilityLevel || "full",
    ensembleMethods: params.ensembleMethods !== false,
    businessThreshold: params.businessThreshold || 0.5,
    positiveClassLabel: params.positiveClassLabel || undefined,
    handleImbalance: params.handleImbalance || false,
    imbalanceStrategy: params.imbalanceStrategy || "smote",
    hyperparameterTuning: params.hyperparameterTuning || "optuna",
    tuningTrials: params.tuningTrials || 50,
  };

  const pythonCode = generatePredictiveCode(finalParams);

  return {
    success: true,
    parameters: finalParams,
    pythonCode,
    providerUsed,
  };
}

function getDefaultMetric(problemType: string): string {
  switch (problemType) {
    case "binary_classification":
      return "roc_auc";
    case "multiclass_classification":
      return "f1_weighted";
    case "regression":
    case "ranking":
      return "rmse";
    default:
      return "roc_auc";
  }
}

function generatePredictiveCode(params: PredictiveAnalyticsParameters): string {
  const timestamp = new Date().toISOString();
  const isClassification = params.problemType.includes("classification");
  const isBinary = params.problemType === "binary_classification";

  return `"""
Predictive Analytics Pipeline: ${params.businessContext}
Problem Type: ${params.problemType}
Generated by ModelWiz.xyz / Cognitive Analysis Platform

Target Variable: ${params.targetVariable}
Business Objective: ${params.businessContext}
Generated on: ${timestamp}

This pipeline includes:
- Automated Exploratory Data Analysis
- Feature Engineering & Selection (${params.featureEngineeringLevel})
- Multiple Model Training & Comparison
- Ensemble Methods (Stacking & Voting)
- Model Interpretation (SHAP)
- Production Deployment Artifacts
- Monitoring & Drift Detection

Required packages:
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost shap optuna imbalanced-learn

Optional packages:
pip install plotly kaleido feature-engine category_encoders
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import pickle
import json
import os
warnings.filterwarnings('ignore')

# Core ML
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, 
                                      KFold, GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                    LabelEncoder, OneHotEncoder, OrdinalEncoder)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression, 
                                        mutual_info_classif, mutual_info_regression,
                                        RFE, RFECV, SelectFromModel)

# Models
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor,
                               AdaBoostClassifier, AdaBoostRegressor,
                               ExtraTreesClassifier, ExtraTreesRegressor,
                               StackingClassifier, StackingRegressor,
                               VotingClassifier, VotingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, log_loss,
                              confusion_matrix, classification_report, roc_curve,
                              precision_recall_curve, mean_squared_error, 
                              mean_absolute_error, r2_score, mean_absolute_percentage_error)

# XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not installed. pip install xgboost")

# LightGBM
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Note: LightGBM not installed. pip install lightgbm")

# CatBoost
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Note: CatBoost not installed. pip install catboost")

# SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Note: SHAP not installed. pip install shap")

# Imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Note: imbalanced-learn not installed. pip install imbalanced-learn")

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Note: Optuna not installed. pip install optuna")

print("=" * 80)
print("PREDICTIVE ANALYTICS PIPELINE")
print("${params.businessContext}")
print("=" * 80)


# ==============================================================================
# PHASE 1: CONFIGURATION
# ==============================================================================

class PipelineConfig:
    """Central configuration for the predictive analytics pipeline"""
    
    # Problem definition
    PROBLEM_TYPE = '${params.problemType}'
    TARGET_COLUMN = '${params.targetVariable}'
    ID_COLUMN = ${params.idColumn ? `'${params.idColumn}'` : 'None'}
    DATE_COLUMN = ${params.dateColumn ? `'${params.dateColumn}'` : 'None'}
    
    # Data splitting
    TEST_SIZE = ${params.testSize}
    VALIDATION_FOLDS = 5
    RANDOM_STATE = 42
    
    # Feature engineering
    FEATURE_ENGINEERING_LEVEL = '${params.featureEngineeringLevel}'
    
    # Model training
    OPTIMIZATION_METRIC = '${params.optimizationMetric}'
    USE_ENSEMBLE = ${params.ensembleMethods ? 'True' : 'False'}
    HYPERPARAMETER_TUNING = '${params.hyperparameterTuning}'
    TUNING_TRIALS = ${params.tuningTrials}
    
    # Class imbalance
    HANDLE_IMBALANCE = ${params.handleImbalance ? 'True' : 'False'}
    IMBALANCE_STRATEGY = '${params.imbalanceStrategy}'
    
    # Interpretability
    INTERPRETABILITY_LEVEL = '${params.interpretabilityLevel}'
    
    # Business settings
    BUSINESS_THRESHOLD = ${params.businessThreshold}
    POSITIVE_CLASS_LABEL = ${params.positiveClassLabel ? `'${params.positiveClassLabel}'` : 'None'}
    
    # Output paths
    OUTPUT_DIR = './model_artifacts'
    
    @classmethod
    def print_config(cls):
        print("\\n=== PIPELINE CONFIGURATION ===")
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                print(f"  {attr}: {getattr(cls, attr)}")

PipelineConfig.print_config()

# Create output directory
os.makedirs(PipelineConfig.OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# PHASE 2: DATA LOADING
# ==============================================================================
print("\\n" + "-" * 80)
print("PHASE 2: DATA LOADING & INITIAL ASSESSMENT")
print("-" * 80)

${generateDataLoadingCode(params)}

print(f"\\n✓ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")


# ==============================================================================
# PHASE 3: DATA QUALITY ASSESSMENT
# ==============================================================================

class DataAssessment:
    """Automated data quality assessment"""
    
    def __init__(self, df, target_col, id_col=None, date_col=None):
        self.df = df.copy()
        self.target_col = target_col
        self.id_col = id_col
        self.date_col = date_col
        self.assessment = {}
        
    def run_assessment(self):
        print("\\n=== DATA QUALITY ASSESSMENT ===\\n")
        
        self.assessment['n_rows'] = len(self.df)
        self.assessment['n_cols'] = len(self.df.columns)
        print(f"Dataset dimensions: {self.assessment['n_rows']:,} rows × {self.assessment['n_cols']} columns")
        
        self._assess_target()
        self._assess_feature_types()
        self._assess_missing()
        self._assess_duplicates()
        self._assess_cardinality()
        self._assess_outliers()
        
        if PipelineConfig.PROBLEM_TYPE in ['binary_classification', 'multiclass_classification']:
            self._assess_class_balance()
        
        return self.assessment
    
    def _assess_target(self):
        print(f"\\n--- Target Variable: {self.target_col} ---")
        target = self.df[self.target_col]
        
        self.assessment['target_dtype'] = str(target.dtype)
        self.assessment['target_missing'] = target.isnull().sum()
        self.assessment['target_unique'] = target.nunique()
        
        print(f"  Data type: {self.assessment['target_dtype']}")
        print(f"  Unique values: {self.assessment['target_unique']}")
        print(f"  Missing: {self.assessment['target_missing']}")
        
        if PipelineConfig.PROBLEM_TYPE == 'regression':
            print(f"  Mean: {target.mean():.4f}")
            print(f"  Std: {target.std():.4f}")
            print(f"  Range: [{target.min():.4f}, {target.max():.4f}]")
        else:
            print(f"  Distribution:")
            for val, count in target.value_counts().items():
                pct = count / len(target) * 100
                print(f"    {val}: {count:,} ({pct:.1f}%)")
    
    def _assess_feature_types(self):
        print(f"\\n--- Feature Types ---")
        
        feature_cols = [c for c in self.df.columns 
                       if c not in [self.target_col, self.id_col, self.date_col]]
        
        numeric_cols = self.df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        boolean_cols = self.df[feature_cols].select_dtypes(include=['bool']).columns.tolist()
        datetime_cols = self.df[feature_cols].select_dtypes(include=['datetime64']).columns.tolist()
        
        self.assessment['numeric_features'] = numeric_cols
        self.assessment['categorical_features'] = categorical_cols
        self.assessment['boolean_features'] = boolean_cols
        self.assessment['datetime_features'] = datetime_cols
        
        print(f"  Numeric: {len(numeric_cols)} features")
        print(f"  Categorical: {len(categorical_cols)} features")
        print(f"  Boolean: {len(boolean_cols)} features")
        print(f"  Datetime: {len(datetime_cols)} features")
        
    def _assess_missing(self):
        print(f"\\n--- Missing Values ---")
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        cols_with_missing = missing[missing > 0]
        
        self.assessment['cols_with_missing'] = len(cols_with_missing)
        self.assessment['total_missing_cells'] = missing.sum()
        
        if len(cols_with_missing) > 0:
            print(f"  Columns with missing values: {len(cols_with_missing)}")
            print(f"  Total missing cells: {missing.sum():,}")
        else:
            print("  ✓ No missing values detected")
    
    def _assess_duplicates(self):
        print(f"\\n--- Duplicate Rows ---")
        
        n_duplicates = self.df.duplicated().sum()
        self.assessment['duplicate_rows'] = n_duplicates
        
        if n_duplicates > 0:
            print(f"  ⚠️  Duplicate rows: {n_duplicates:,}")
        else:
            print("  ✓ No duplicate rows detected")
    
    def _assess_cardinality(self):
        print(f"\\n--- Cardinality Analysis ---")
        
        high_cardinality = []
        for col in self.assessment.get('categorical_features', []):
            n_unique = self.df[col].nunique()
            if n_unique > 50:
                high_cardinality.append((col, n_unique))
        
        self.assessment['high_cardinality_features'] = high_cardinality
        
        if high_cardinality:
            print(f"  High cardinality categorical (>50 unique):")
            for col, n in high_cardinality[:5]:
                print(f"    {col}: {n} unique values")
    
    def _assess_outliers(self):
        print(f"\\n--- Outlier Detection (IQR Method) ---")
        
        outlier_cols = []
        for col in self.assessment.get('numeric_features', []):
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            if n_outliers > 0:
                outlier_cols.append((col, n_outliers, n_outliers/len(self.df)*100))
        
        self.assessment['outlier_features'] = outlier_cols
        
        if outlier_cols:
            print(f"  Features with outliers:")
            for col, n, pct in sorted(outlier_cols, key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {col}: {n:,} outliers ({pct:.1f}%)")
        else:
            print("  ✓ No significant outliers detected")
    
    def _assess_class_balance(self):
        print(f"\\n--- Class Balance ---")
        
        target = self.df[self.target_col]
        value_counts = target.value_counts()
        
        majority_class = value_counts.index[0]
        minority_class = value_counts.index[-1]
        imbalance_ratio = value_counts.iloc[-1] / value_counts.iloc[0]
        
        self.assessment['imbalance_ratio'] = imbalance_ratio
        
        if imbalance_ratio < 0.2:
            print(f"  ⚠️  SEVERE class imbalance detected!")
            print(f"  Imbalance ratio: {imbalance_ratio:.3f}")
            print(f"  → Recommend: SMOTE, class weights, or undersampling")
        elif imbalance_ratio < 0.5:
            print(f"  ⚠️  Moderate class imbalance detected")
        else:
            print(f"  ✓ Classes are relatively balanced (ratio: {imbalance_ratio:.3f})")

# Run assessment
assessor = DataAssessment(
    df, 
    target_col=PipelineConfig.TARGET_COLUMN,
    id_col=PipelineConfig.ID_COLUMN,
    date_col=PipelineConfig.DATE_COLUMN
)
data_assessment = assessor.run_assessment()


# ==============================================================================
# PHASE 4: AUTOMATED EDA
# ==============================================================================
print("\\n" + "-" * 80)
print("PHASE 4: AUTOMATED EXPLORATORY DATA ANALYSIS")
print("-" * 80)

class AutomatedEDA:
    """Automated exploratory data analysis with visualizations"""
    
    def __init__(self, df, target_col, numeric_features, categorical_features, problem_type):
        self.df = df.copy()
        self.target_col = target_col
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.problem_type = problem_type
        self.insights = []
        
    def run_eda(self, save_plots=True):
        print("\\n=== AUTOMATED EDA ===\\n")
        
        self._analyze_numeric_features()
        self._analyze_categorical_features()
        self._analyze_correlations()
        
        if save_plots:
            self._generate_eda_visualizations()
        
        self._print_insights()
        
        return self.insights
    
    def _analyze_numeric_features(self):
        print("--- Numeric Feature Analysis ---")
        
        if not self.numeric_features:
            print("  No numeric features to analyze")
            return
        
        stats = self.df[self.numeric_features].describe().T
        stats['skewness'] = self.df[self.numeric_features].skew()
        
        skewed = stats[abs(stats['skewness']) > 1].index.tolist()
        if skewed:
            self.insights.append(f"Highly skewed features: {skewed[:5]}")
            print(f"  Highly skewed features: {len(skewed)}")
        
        zero_var = stats[stats['std'] == 0].index.tolist()
        if zero_var:
            self.insights.append(f"⚠️ Zero variance features: {zero_var}")
            print(f"  ⚠️ Zero variance features: {zero_var}")
        
        print(f"  Analyzed {len(self.numeric_features)} numeric features")
        
    def _analyze_categorical_features(self):
        print("\\n--- Categorical Feature Analysis ---")
        
        if not self.categorical_features:
            print("  No categorical features to analyze")
            return
        
        print(f"  Analyzed {len(self.categorical_features)} categorical features")
    
    def _analyze_correlations(self):
        print("\\n--- Correlation Analysis ---")
        
        if len(self.numeric_features) < 2:
            print("  Not enough numeric features for correlation analysis")
            return
        
        corr_matrix = self.df[self.numeric_features].corr()
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        
        if high_corr_pairs:
            print(f"  Highly correlated feature pairs (|r| > 0.8): {len(high_corr_pairs)}")
            self.insights.append(f"Highly correlated pairs found: {len(high_corr_pairs)}")
        else:
            print("  No highly correlated feature pairs detected")
    
    def _generate_eda_visualizations(self):
        print("\\n--- Generating EDA Visualizations ---")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Target distribution
        ax1 = fig.add_subplot(2, 3, 1)
        if self.problem_type == 'regression':
            self.df[self.target_col].hist(bins=50, ax=ax1, color='steelblue', edgecolor='k')
            ax1.set_xlabel(self.target_col)
            ax1.set_ylabel('Frequency')
            ax1.set_title('Target Distribution')
        else:
            self.df[self.target_col].value_counts().plot(kind='bar', ax=ax1, color='steelblue', edgecolor='k')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Count')
            ax1.set_title('Target Class Distribution')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Missing values
        ax2 = fig.add_subplot(2, 3, 2)
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False).head(10)
        if len(missing) > 0:
            missing.plot(kind='barh', ax=ax2, color='coral')
            ax2.set_xlabel('Missing Count')
            ax2.set_title('Top Features with Missing Values')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            ax2.set_title('Missing Values')
        
        # 3. Correlation heatmap
        ax3 = fig.add_subplot(2, 3, 3)
        if len(self.numeric_features) > 1:
            top_numeric = self.df[self.numeric_features].var().sort_values(ascending=False).head(8).index.tolist()
            corr_subset = self.df[top_numeric].corr()
            sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                       ax=ax3, annot_kws={'size': 8}, square=True)
            ax3.set_title('Correlation Heatmap (Top 8)')
        
        # 4-6. Top numeric feature distributions
        top_numeric = self.numeric_features[:3] if self.numeric_features else []
        for idx, col in enumerate(top_numeric):
            ax = fig.add_subplot(2, 3, 4 + idx)
            if self.problem_type in ['binary_classification', 'multiclass_classification']:
                for label in self.df[self.target_col].unique():
                    subset = self.df[self.df[self.target_col] == label][col]
                    subset.hist(bins=30, alpha=0.5, label=str(label), ax=ax)
                ax.legend()
            else:
                self.df[col].hist(bins=50, ax=ax, color='steelblue', edgecolor='k', alpha=0.7)
            ax.set_xlabel(col)
            ax.set_title(f'Distribution: {col}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PipelineConfig.OUTPUT_DIR, 'eda_analysis.png'), dpi=150, bbox_inches='tight')
        plt.show()
        print("  ✓ EDA visualizations saved")
    
    def _print_insights(self):
        print("\\n=== KEY EDA INSIGHTS ===")
        for i, insight in enumerate(self.insights, 1):
            print(f"  {i}. {insight}")

# Run automated EDA
eda = AutomatedEDA(
    df=df,
    target_col=PipelineConfig.TARGET_COLUMN,
    numeric_features=data_assessment['numeric_features'],
    categorical_features=data_assessment['categorical_features'],
    problem_type=PipelineConfig.PROBLEM_TYPE
)
eda_insights = eda.run_eda()


# ==============================================================================
# PHASE 5: FEATURE ENGINEERING
# ==============================================================================
print("\\n" + "-" * 80)
print("PHASE 5: FEATURE ENGINEERING")
print("-" * 80)

${generateFeatureEngineeringCode(params)}


# ==============================================================================
# PHASE 6: FEATURE SELECTION
# ==============================================================================
print("\\n" + "-" * 80)
print("PHASE 6: FEATURE SELECTION")
print("-" * 80)

${generateFeatureSelectionCode(params)}


# ==============================================================================
# PHASE 7: DATA SPLITTING
# ==============================================================================
print("\\n" + "-" * 80)
print("PHASE 7: DATA SPLITTING")
print("-" * 80)

${generateDataSplittingCode(params)}


# ==============================================================================
# PHASE 8: MODEL TRAINING & COMPARISON
# ==============================================================================
print("\\n" + "-" * 80)
print("PHASE 8: MODEL TRAINING & COMPARISON")
print("-" * 80)

${generateModelTrainingCode(params)}


# ==============================================================================
# PHASE 9: ENSEMBLE METHODS
# ==============================================================================
${params.ensembleMethods ? generateEnsembleCode(params) : '# Ensemble methods disabled\nprint("\\nEnsemble methods: DISABLED")'}


# ==============================================================================
# PHASE 10: MODEL INTERPRETATION
# ==============================================================================
print("\\n" + "-" * 80)
print("PHASE 10: MODEL INTERPRETATION")
print("-" * 80)

${generateInterpretationCode(params)}


# ==============================================================================
# PHASE 11: FINAL EVALUATION & VISUALIZATIONS
# ==============================================================================
print("\\n" + "-" * 80)
print("PHASE 11: FINAL EVALUATION & VISUALIZATIONS")
print("-" * 80)

${generateFinalEvaluationCode(params)}


# ==============================================================================
# PHASE 12: PRODUCTION DEPLOYMENT ARTIFACTS
# ==============================================================================
print("\\n" + "-" * 80)
print("PHASE 12: PRODUCTION DEPLOYMENT ARTIFACTS")
print("-" * 80)

${generateDeploymentCode(params)}


# ==============================================================================
# PHASE 13: MONITORING & DRIFT DETECTION
# ==============================================================================
print("\\n" + "-" * 80)
print("PHASE 13: MONITORING & DRIFT DETECTION")
print("-" * 80)

${generateMonitoringCode(params)}


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\\n" + "=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)

print(f"""
✓ Best Model: {trainer.best_model_name}
✓ Artifacts saved to: {PipelineConfig.OUTPUT_DIR}/
✓ Use predict() function for new predictions
✓ Monitor model performance with drift detection utilities

Files Generated:
  - {PipelineConfig.OUTPUT_DIR}/best_model.pkl
  - {PipelineConfig.OUTPUT_DIR}/scaler.pkl
  - {PipelineConfig.OUTPUT_DIR}/feature_list.json
  - {PipelineConfig.OUTPUT_DIR}/model_metrics.json
  - {PipelineConfig.OUTPUT_DIR}/eda_analysis.png
  - {PipelineConfig.OUTPUT_DIR}/final_evaluation.png
${params.interpretabilityLevel === 'full' ? `  - ${PipelineConfig.OUTPUT_DIR}/shap_analysis.png` : ''}
""")
`;
}

function generateDataLoadingCode(params: PredictiveAnalyticsParameters): string {
  if (params.dataSource === "synthetic" || params.dataSource.includes("synthetic")) {
    const isClassification = params.problemType.includes("classification");
    const isBinary = params.problemType === "binary_classification";
    
    return `
# Generate synthetic data for demonstration
np.random.seed(PipelineConfig.RANDOM_STATE)
n_samples = 5000
n_features = 15

# Generate features
feature_data = {}
for i in range(n_features):
    if i % 3 == 0:
        feature_data[f'numeric_{i}'] = np.random.randn(n_samples) * 10 + 50
    elif i % 3 == 1:
        feature_data[f'numeric_{i}'] = np.random.exponential(5, n_samples)
    else:
        feature_data[f'numeric_{i}'] = np.random.uniform(0, 100, n_samples)

# Add categorical features
feature_data['category_A'] = np.random.choice(['Low', 'Medium', 'High'], n_samples)
feature_data['category_B'] = np.random.choice(['Type1', 'Type2', 'Type3', 'Type4'], n_samples)
feature_data['region'] = np.random.choice(['North', 'South', 'East', 'West'], n_samples)

# Generate target based on features
${isClassification ? `
# Classification target
base_prob = 0.3 + 0.3 * (feature_data['numeric_0'] > 50) + 0.2 * (feature_data['numeric_3'] > 5)
base_prob = np.clip(base_prob, 0.05, 0.95)
${isBinary ? 
`target = (np.random.random(n_samples) < base_prob).astype(int)` : 
`target = np.random.choice(['Class_A', 'Class_B', 'Class_C'], n_samples, p=[0.5, 0.3, 0.2])`}
` : `
# Regression target  
target = (
    2.5 * feature_data['numeric_0'] + 
    1.8 * feature_data['numeric_3'] - 
    0.5 * feature_data['numeric_6'] + 
    np.random.randn(n_samples) * 10 + 100
)`}

feature_data['${params.targetVariable}'] = target

# Create DataFrame
df = pd.DataFrame(feature_data)

# Add some missing values (5% randomly)
for col in df.columns[:5]:
    mask = np.random.random(len(df)) < 0.05
    df.loc[mask, col] = np.nan
`;
  }
  
  return `
# Load data from file
# Modify the path below to match your data location
data_path = '${params.dataSource}'

if data_path.endswith('.csv'):
    df = pd.read_csv(data_path)
elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
    df = pd.read_excel(data_path)
elif data_path.endswith('.parquet'):
    df = pd.read_parquet(data_path)
else:
    raise ValueError(f"Unsupported file format: {data_path}")
`;
}

function generateFeatureEngineeringCode(params: PredictiveAnalyticsParameters): string {
  return `
class FeatureEngineer:
    """Automated feature engineering pipeline"""
    
    def __init__(self, df, target_col, numeric_features, categorical_features, 
                 id_col=None, date_col=None, level='standard'):
        self.df = df.copy()
        self.target_col = target_col
        self.numeric_features = [f for f in numeric_features if f in df.columns]
        self.categorical_features = [f for f in categorical_features if f in df.columns]
        self.id_col = id_col
        self.date_col = date_col
        self.level = level
        self.feature_log = []
        self.new_features = []
        
    def engineer_features(self):
        print(f"\\n=== FEATURE ENGINEERING (Level: {self.level}) ===\\n")
        
        # Remove ID column from features
        if self.id_col and self.id_col in self.df.columns:
            self.df = self.df.drop(columns=[self.id_col])
            print(f"  Removed ID column: {self.id_col}")
        
        # Handle missing values
        self._handle_missing()
        
        # Encode categorical features
        self._encode_categoricals()
        
        if self.level in ['standard', 'aggressive']:
            self._create_interactions()
            self._create_polynomial_features()
        
        if self.level == 'aggressive':
            self._create_statistical_features()
            self._create_ratio_features()
        
        if self.date_col:
            self._create_date_features()
        
        self._print_summary()
        
        return self.df, self.feature_log
    
    def _handle_missing(self):
        print("--- Handling Missing Values ---")
        
        for col in self.numeric_features:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                self.feature_log.append(f"Imputed {col} with median")
        
        for col in self.categorical_features:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else 'UNKNOWN'
                self.df[col] = self.df[col].fillna(mode_val)
        
        print(f"  ✓ Missing values handled")
    
    def _encode_categoricals(self):
        print("\\n--- Encoding Categorical Features ---")
        
        for col in self.categorical_features:
            if col not in self.df.columns or col == self.target_col:
                continue
            
            n_unique = self.df[col].nunique()
            
            if n_unique == 2:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                
            elif n_unique <= 10:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df = self.df.drop(columns=[col])
                self.new_features.extend(dummies.columns.tolist())
                
            else:
                freq_map = self.df[col].value_counts(normalize=True).to_dict()
                self.df[f'{col}_freq'] = self.df[col].map(freq_map)
                self.df = self.df.drop(columns=[col])
                self.new_features.append(f'{col}_freq')
        
        print(f"  ✓ Encoded categorical features")
    
    def _create_interactions(self):
        print("\\n--- Creating Interaction Features ---")
        
        numeric_cols = [c for c in self.numeric_features if c in self.df.columns][:5]
        
        if len(numeric_cols) < 2:
            return
        
        interaction_count = 0
        for i in range(len(numeric_cols)):
            for j in range(i+1, min(len(numeric_cols), i+3)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                new_col = f'{col1}_x_{col2}'
                self.df[new_col] = self.df[col1] * self.df[col2]
                self.new_features.append(new_col)
                interaction_count += 1
        
        print(f"  ✓ Created {interaction_count} interaction features")
    
    def _create_polynomial_features(self):
        print("\\n--- Creating Polynomial Features ---")
        
        numeric_cols = [c for c in self.numeric_features if c in self.df.columns][:5]
        poly_count = 0
        
        for col in numeric_cols:
            self.df[f'{col}_squared'] = self.df[col] ** 2
            self.new_features.append(f'{col}_squared')
            poly_count += 1
            
            if (self.df[col] > 0).all():
                self.df[f'{col}_log'] = np.log1p(self.df[col])
                self.new_features.append(f'{col}_log')
                poly_count += 1
        
        print(f"  ✓ Created {poly_count} polynomial features")
    
    def _create_statistical_features(self):
        print("\\n--- Creating Statistical Features ---")
        
        numeric_cols = [c for c in self.numeric_features if c in self.df.columns]
        
        if len(numeric_cols) < 3:
            return
        
        numeric_df = self.df[numeric_cols]
        
        self.df['numeric_mean'] = numeric_df.mean(axis=1)
        self.df['numeric_std'] = numeric_df.std(axis=1)
        self.df['numeric_range'] = numeric_df.max(axis=1) - numeric_df.min(axis=1)
        
        self.new_features.extend(['numeric_mean', 'numeric_std', 'numeric_range'])
        print(f"  ✓ Created 3 statistical features")
    
    def _create_ratio_features(self):
        print("\\n--- Creating Ratio Features ---")
        
        numeric_cols = [c for c in self.numeric_features if c in self.df.columns][:4]
        ratio_count = 0
        
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                
                if (self.df[col2].abs() > 0.01).all():
                    new_col = f'{col1}_div_{col2}'
                    self.df[new_col] = self.df[col1] / (self.df[col2] + 1e-8)
                    self.new_features.append(new_col)
                    ratio_count += 1
        
        print(f"  ✓ Created {ratio_count} ratio features")
    
    def _create_date_features(self):
        print("\\n--- Creating Date Features ---")
        
        if self.date_col not in self.df.columns:
            return
        
        date_series = pd.to_datetime(self.df[self.date_col], errors='coerce')
        
        self.df['year'] = date_series.dt.year
        self.df['month'] = date_series.dt.month
        self.df['dayofweek'] = date_series.dt.dayofweek
        self.df['is_weekend'] = (date_series.dt.dayofweek >= 5).astype(int)
        
        self.new_features.extend(['year', 'month', 'dayofweek', 'is_weekend'])
        self.df = self.df.drop(columns=[self.date_col])
        
        print(f"  ✓ Created 4 date features")
    
    def _print_summary(self):
        print("\\n=== FEATURE ENGINEERING SUMMARY ===")
        print(f"  Original features: {len(self.numeric_features) + len(self.categorical_features)}")
        print(f"  New features created: {len(self.new_features)}")
        print(f"  Total features: {len([c for c in self.df.columns if c != self.target_col])}")

# Run feature engineering
feature_engineer = FeatureEngineer(
    df=df,
    target_col=PipelineConfig.TARGET_COLUMN,
    numeric_features=data_assessment['numeric_features'],
    categorical_features=data_assessment['categorical_features'],
    id_col=PipelineConfig.ID_COLUMN,
    date_col=PipelineConfig.DATE_COLUMN,
    level=PipelineConfig.FEATURE_ENGINEERING_LEVEL
)

df_engineered, feature_log = feature_engineer.engineer_features()
`;
}

function generateFeatureSelectionCode(params: PredictiveAnalyticsParameters): string {
  const isClassification = params.problemType.includes("classification");
  
  return `
class FeatureSelector:
    """Automated feature selection using multiple methods"""
    
    def __init__(self, df, target_col, problem_type):
        self.df = df.copy()
        self.target_col = target_col
        self.problem_type = problem_type
        self.feature_cols = [c for c in df.columns if c != target_col]
        
    def select_features(self, n_features=None):
        print(f"\\n=== FEATURE SELECTION ===\\n")
        
        X = self.df[self.feature_cols]
        y = self.df[self.target_col]
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        if n_features is None:
            n_features = max(10, int(len(self.feature_cols) * 0.6))
        
        print(f"  Original features: {len(self.feature_cols)}")
        print(f"  Target selection: {n_features} features\\n")
        
        # Method 1: Variance-based
        variances = X.var()
        var_features = variances.sort_values(ascending=False).head(n_features).index.tolist()
        
        # Method 2: Correlation with target
        ${isClassification ? `
        y_numeric = LabelEncoder().fit_transform(y) if y.dtype == 'object' else y
        correlations = X.apply(lambda col: np.abs(np.corrcoef(col.fillna(0), y_numeric)[0, 1])).fillna(0)
        ` : `
        correlations = X.apply(lambda col: np.abs(col.corr(y))).fillna(0)
        `}
        corr_features = correlations.sort_values(ascending=False).head(n_features).index.tolist()
        
        # Method 3: Mutual Information
        ${isClassification ? 
        `mi_func = mutual_info_classif` : 
        `mi_func = mutual_info_regression`}
        
        try:
            mi_scores = mi_func(X.fillna(0), y, random_state=42)
            mi_series = pd.Series(mi_scores, index=X.columns)
            mi_features = mi_series.sort_values(ascending=False).head(n_features).index.tolist()
        except:
            mi_features = var_features
        
        # Combine: features appearing in multiple methods
        feature_counts = {}
        for features in [var_features, corr_features, mi_features]:
            for feat in features:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
        
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        final_features = [f[0] for f in sorted_features[:n_features]]
        
        print(f"  Selected {len(final_features)} features")
        print(f"  Top features: {final_features[:5]}")
        
        return final_features

# Run feature selection
selector = FeatureSelector(
    df=df_engineered,
    target_col=PipelineConfig.TARGET_COLUMN,
    problem_type=PipelineConfig.PROBLEM_TYPE
)
selected_features = selector.select_features()
`;
}

function generateDataSplittingCode(params: PredictiveAnalyticsParameters): string {
  const isClassification = params.problemType.includes("classification");
  
  return `
# Prepare final datasets
X = df_engineered[selected_features].fillna(0)
y = df_engineered[PipelineConfig.TARGET_COLUMN]

# Encode target if needed
${isClassification ? `
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y), index=y.index)
    class_names = label_encoder.classes_.tolist()
    print(f"\\nTarget encoded: {dict(zip(class_names, range(len(class_names))))}")
else:
    class_names = sorted(y.unique().tolist())
    label_encoder = None
` : `
class_names = None
label_encoder = None
`}

# Train-test split
${isClassification ? `
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=PipelineConfig.TEST_SIZE, 
    random_state=PipelineConfig.RANDOM_STATE,
    stratify=y
)
` : `
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=PipelineConfig.TEST_SIZE, 
    random_state=PipelineConfig.RANDOM_STATE
)
`}

print(f"\\n=== DATA SPLIT ===")
print(f"  Training set: {len(X_train):,} samples ({(1-PipelineConfig.TEST_SIZE)*100:.0f}%)")
print(f"  Test set: {len(X_test):,} samples ({PipelineConfig.TEST_SIZE*100:.0f}%)")
print(f"  Features: {X_train.shape[1]}")

${params.handleImbalance && isClassification ? `
# Handle class imbalance
if IMBLEARN_AVAILABLE and PipelineConfig.IMBALANCE_STRATEGY == 'smote':
    print(f"\\n--- Applying SMOTE for class imbalance ---")
    print(f"  Before SMOTE: {dict(pd.Series(y_train).value_counts())}")
    
    smote = SMOTE(random_state=PipelineConfig.RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"  After SMOTE: {dict(pd.Series(y_train_balanced).value_counts())}")
    
    X_train, y_train = X_train_balanced, y_train_balanced
` : ''}

# Feature scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(f"\\n✓ Data prepared for modeling")
`;
}

function generateModelTrainingCode(params: PredictiveAnalyticsParameters): string {
  const isClassification = params.problemType.includes("classification");
  const isBinary = params.problemType === "binary_classification";
  
  return `
class ModelTrainer:
    """Train and compare multiple models"""
    
    def __init__(self, problem_type, optimization_metric, random_state=42):
        self.problem_type = problem_type
        self.optimization_metric = optimization_metric
        self.random_state = random_state
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None
        
    def get_models(self):
        """Get candidate models based on problem type"""
        
        ${isClassification ? `
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1, class_weight='balanced'),
            'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMClassifier(n_estimators=100, random_state=self.random_state, verbose=-1, class_weight='balanced')
        
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostClassifier(iterations=100, random_state=self.random_state, verbose=0, auto_class_weights='Balanced')
        ` : `
        models = {
            'Ridge': Ridge(random_state=self.random_state),
            'Lasso': Lasso(random_state=self.random_state),
            'ElasticNet': ElasticNet(random_state=self.random_state),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=self.random_state)
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
        
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostRegressor(iterations=100, random_state=self.random_state, verbose=0)
        `}
        
        return models
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, cv_folds=5):
        """Train all models and evaluate performance"""
        
        print(f"\\n=== TRAINING {len(self.get_models())} MODELS ===\\n")
        
        models = self.get_models()
        
        for name, model in models.items():
            print(f"Training {name}...", end=" ")
            
            try:
                model.fit(X_train, y_train)
                
                ${isClassification ? `
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                scoring = 'roc_auc' if self.problem_type == 'binary_classification' else 'f1_weighted'
                ` : `
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                scoring = 'neg_root_mean_squared_error'
                `}
                
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                y_pred = model.predict(X_test)
                
                ${isBinary ? `
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                metrics = {
                    'Model': name,
                    'CV Score': cv_scores.mean(),
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_pred, zero_division=0),
                    'F1': f1_score(y_test, y_pred, zero_division=0),
                    'ROC-AUC': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
                }
                print(f"F1: {metrics['F1']:.4f}, ROC-AUC: {metrics['ROC-AUC']:.4f}")
                ` : isClassification ? `
                metrics = {
                    'Model': name,
                    'CV Score': cv_scores.mean(),
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                }
                print(f"F1: {metrics['F1']:.4f}")
                ` : `
                metrics = {
                    'Model': name,
                    'CV Score': -cv_scores.mean(),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'R2': r2_score(y_test, y_pred),
                }
                print(f"RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}")
                `}
                
                self.models[name] = model
                self.results.append(metrics)
                    
            except Exception as e:
                print(f"Failed: {e}")
        
        self.results_df = pd.DataFrame(self.results)
        
        ${isBinary ? `
        sort_col = 'ROC-AUC'
        ascending = False
        ` : isClassification ? `
        sort_col = 'F1'
        ascending = False
        ` : `
        sort_col = 'RMSE'
        ascending = True
        `}
        
        self.results_df = self.results_df.sort_values(sort_col, ascending=ascending)
        
        self.best_model_name = self.results_df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]
        
        return self.results_df
    
    def print_results(self):
        print("\\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        
        display_df = self.results_df.copy()
        for col in display_df.columns:
            if display_df[col].dtype == 'float64':
                display_df[col] = display_df[col].round(4)
        
        print(display_df.to_string(index=False))
        
        print(f"\\n✓ BEST MODEL: {self.best_model_name}")

# Train models
trainer = ModelTrainer(
    problem_type=PipelineConfig.PROBLEM_TYPE,
    optimization_metric=PipelineConfig.OPTIMIZATION_METRIC,
    random_state=PipelineConfig.RANDOM_STATE
)

results_df = trainer.train_and_evaluate(
    X_train_scaled, X_test_scaled, y_train, y_test,
    cv_folds=PipelineConfig.VALIDATION_FOLDS
)

trainer.print_results()
`;
}

function generateEnsembleCode(params: PredictiveAnalyticsParameters): string {
  const isClassification = params.problemType.includes("classification");
  const isBinary = params.problemType === "binary_classification";
  
  return `
print("\\n" + "-" * 80)
print("PHASE 9: ENSEMBLE METHODS")
print("-" * 80)

print("\\n=== BUILDING ENSEMBLE MODELS ===\\n")

# Get top 3 models for stacking
top_models = results_df.head(3)['Model'].tolist()
print(f"  Base models for stacking: {top_models}")

${isClassification ? `
estimators = [(name, trainer.models[name]) for name in top_models]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(random_state=PipelineConfig.RANDOM_STATE),
    cv=5,
    n_jobs=-1
)

voting_model = VotingClassifier(
    estimators=estimators,
    voting='soft',
    n_jobs=-1
)
` : `
estimators = [(name, trainer.models[name]) for name in top_models]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(random_state=PipelineConfig.RANDOM_STATE),
    cv=5,
    n_jobs=-1
)

voting_model = VotingRegressor(
    estimators=estimators,
    n_jobs=-1
)
`}

ensemble_results = []

for name, model in [('Stacking', stacking_model), ('Voting', voting_model)]:
    print(f"\\nTraining {name} Ensemble...")
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    ${isBinary ? `
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'Model': f'{name} Ensemble',
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }
    print(f"  {name}: F1={metrics['F1']:.4f}, ROC-AUC={metrics['ROC-AUC']:.4f}")
    ` : isClassification ? `
    metrics = {
        'Model': f'{name} Ensemble',
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    print(f"  {name}: F1={metrics['F1']:.4f}")
    ` : `
    metrics = {
        'Model': f'{name} Ensemble',
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    print(f"  {name}: RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
    `}
    
    ensemble_results.append(metrics)
    trainer.models[f'{name} Ensemble'] = model

ensemble_df = pd.DataFrame(ensemble_results)
print("\\n=== ENSEMBLE RESULTS ===")
print(ensemble_df.round(4).to_string(index=False))

# Update best model if ensemble is better
${isBinary ? `
best_ensemble = ensemble_df.loc[ensemble_df['ROC-AUC'].idxmax()]
if best_ensemble['ROC-AUC'] > results_df.iloc[0]['ROC-AUC']:
    trainer.best_model_name = best_ensemble['Model']
    trainer.best_model = trainer.models[best_ensemble['Model']]
    print(f"\\n✓ Ensemble outperforms single models!")
` : isClassification ? `
best_ensemble = ensemble_df.loc[ensemble_df['F1'].idxmax()]
if best_ensemble['F1'] > results_df.iloc[0]['F1']:
    trainer.best_model_name = best_ensemble['Model']
    trainer.best_model = trainer.models[best_ensemble['Model']]
` : `
best_ensemble = ensemble_df.loc[ensemble_df['RMSE'].idxmin()]
if best_ensemble['RMSE'] < results_df.iloc[0]['RMSE']:
    trainer.best_model_name = best_ensemble['Model']
    trainer.best_model = trainer.models[best_ensemble['Model']]
`}

print(f"\\n✓ FINAL BEST MODEL: {trainer.best_model_name}")
best_model = trainer.best_model
`;
}

function generateInterpretationCode(params: PredictiveAnalyticsParameters): string {
  if (params.interpretabilityLevel !== "full") {
    return `
print("\\n=== FEATURE IMPORTANCE (Basic) ===\\n")

if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(importance_df.head(15).to_string(index=False))
elif hasattr(best_model, 'coef_'):
    coef = best_model.coef_.flatten() if len(best_model.coef_.shape) > 1 else best_model.coef_
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': np.abs(coef)
    }).sort_values('Coefficient', ascending=False)
    
    print(importance_df.head(15).to_string(index=False))
`;
  }
  
  return `
if SHAP_AVAILABLE:
    print("\\n=== SHAP ANALYSIS ===\\n")
    
    print(f"  Generating SHAP explanations for {trainer.best_model_name}...")
    
    try:
        shap_sample_size = min(500, len(X_test_scaled))
        X_shap = X_test_scaled.iloc[:shap_sample_size]
        
        model_type = type(best_model).__name__
        
        if 'Forest' in model_type or 'XGB' in model_type or 'LGBM' in model_type or 'CatBoost' in model_type or 'Gradient' in model_type:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_shap)
        else:
            background = shap.sample(X_train_scaled, 100)
            explainer = shap.KernelExplainer(
                best_model.predict_proba if hasattr(best_model, 'predict_proba') else best_model.predict,
                background
            )
            shap_values = explainer.shap_values(X_shap)
        
        # For binary classification, get values for positive class
        if PipelineConfig.PROBLEM_TYPE == 'binary_classification' and isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate feature importance from SHAP
        if isinstance(shap_values, list):
            shap_importance = np.abs(shap_values[0]).mean(axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)
        
        shap_importance_df = pd.DataFrame({
            'Feature': X_shap.columns,
            'SHAP Importance': shap_importance
        }).sort_values('SHAP Importance', ascending=False)
        
        print("\\n=== TOP FEATURES BY SHAP IMPORTANCE ===")
        print(shap_importance_df.head(15).to_string(index=False))
        
        # Generate SHAP visualizations
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        plt.sca(axes[0])
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=15)
        axes[0].set_title('SHAP Feature Importance', fontsize=12)
        
        plt.sca(axes[1])
        shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
        axes[1].set_title('SHAP Value Distribution', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PipelineConfig.OUTPUT_DIR, 'shap_analysis.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\\n  ✓ SHAP visualizations saved")
        
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")
        print("  Falling back to model feature importance...")
        
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\\n=== TOP FEATURES BY MODEL IMPORTANCE ===")
            print(importance_df.head(15).to_string(index=False))

else:
    print("\\n=== FEATURE IMPORTANCE (Model-Based) ===\\n")
    
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(importance_df.head(15).to_string(index=False))
`;
}

function generateFinalEvaluationCode(params: PredictiveAnalyticsParameters): string {
  const isClassification = params.problemType.includes("classification");
  const isBinary = params.problemType === "binary_classification";
  
  return `
# Final predictions
y_pred_final = best_model.predict(X_test_scaled)

${isClassification ? `
if hasattr(best_model, 'predict_proba'):
    y_proba_final = best_model.predict_proba(X_test_scaled)
else:
    y_proba_final = None
` : ''}

# Generate comprehensive visualizations
fig = plt.figure(figsize=(20, 12))

${isBinary ? `
# Plot 1: Confusion Matrix
ax1 = fig.add_subplot(2, 3, 1)
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix')

# Plot 2: ROC Curve
ax2 = fig.add_subplot(2, 3, 2)
if y_proba_final is not None:
    fpr, tpr, _ = roc_curve(y_test, y_proba_final[:, 1])
    auc = roc_auc_score(y_test, y_proba_final[:, 1])
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=1)
    ax2.fill_between(fpr, tpr, alpha=0.2)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()

# Plot 3: Precision-Recall Curve
ax3 = fig.add_subplot(2, 3, 3)
if y_proba_final is not None:
    precision, recall, _ = precision_recall_curve(y_test, y_proba_final[:, 1])
    pr_auc = average_precision_score(y_test, y_proba_final[:, 1])
    ax3.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AUC = {pr_auc:.4f})')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve')
    ax3.legend()

# Plot 4: Probability Distribution
ax4 = fig.add_subplot(2, 3, 4)
if y_proba_final is not None:
    for label in [0, 1]:
        mask = y_test == label
        ax4.hist(y_proba_final[mask, 1], bins=30, alpha=0.5, label=f'Class {label}')
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Count')
    ax4.set_title('Probability Distribution by Class')
    ax4.legend()
    ax4.axvline(x=PipelineConfig.BUSINESS_THRESHOLD, color='r', linestyle='--', label='Threshold')

# Plot 5: Feature Importance
ax5 = fig.add_subplot(2, 3, 5)
if hasattr(best_model, 'feature_importances_'):
    imp_df = pd.DataFrame({'Feature': selected_features, 'Importance': best_model.feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=True).tail(10)
    ax5.barh(imp_df['Feature'], imp_df['Importance'], color='steelblue')
    ax5.set_xlabel('Importance')
    ax5.set_title('Top 10 Feature Importances')

# Plot 6: Classification Report
ax6 = fig.add_subplot(2, 3, 6)
report = classification_report(y_test, y_pred_final, output_dict=True)
report_df = pd.DataFrame(report).transpose()
ax6.axis('off')
ax6.table(cellText=report_df.round(3).values,
          colLabels=report_df.columns,
          rowLabels=report_df.index,
          cellLoc='center',
          loc='center')
ax6.set_title('Classification Report')
` : isClassification ? `
# Multi-class: Confusion Matrix and Metrics
ax1 = fig.add_subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix')

# Feature Importance
ax2 = fig.add_subplot(2, 2, 2)
if hasattr(best_model, 'feature_importances_'):
    imp_df = pd.DataFrame({'Feature': selected_features, 'Importance': best_model.feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=True).tail(10)
    ax2.barh(imp_df['Feature'], imp_df['Importance'], color='steelblue')
    ax2.set_xlabel('Importance')
    ax2.set_title('Top 10 Feature Importances')

# Classification Report
ax3 = fig.add_subplot(2, 2, 3)
report = classification_report(y_test, y_pred_final, output_dict=True)
report_df = pd.DataFrame(report).transpose()
ax3.axis('off')
ax3.table(cellText=report_df.round(3).values,
          colLabels=report_df.columns,
          rowLabels=report_df.index,
          cellLoc='center',
          loc='center')
ax3.set_title('Classification Report')
` : `
# Regression: Actual vs Predicted
ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(y_test, y_pred_final, alpha=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
ax1.set_title('Actual vs Predicted')

# Residual Distribution
ax2 = fig.add_subplot(2, 3, 2)
residuals = y_test - y_pred_final
ax2.hist(residuals, bins=50, color='steelblue', edgecolor='k')
ax2.axvline(x=0, color='r', linestyle='--')
ax2.set_xlabel('Residual')
ax2.set_ylabel('Frequency')
ax2.set_title('Residual Distribution')

# Residuals vs Predicted
ax3 = fig.add_subplot(2, 3, 3)
ax3.scatter(y_pred_final, residuals, alpha=0.5)
ax3.axhline(y=0, color='r', linestyle='--')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Residual')
ax3.set_title('Residuals vs Predicted')

# Feature Importance
ax4 = fig.add_subplot(2, 3, 4)
if hasattr(best_model, 'feature_importances_'):
    imp_df = pd.DataFrame({'Feature': selected_features, 'Importance': best_model.feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=True).tail(10)
    ax4.barh(imp_df['Feature'], imp_df['Importance'], color='steelblue')
    ax4.set_xlabel('Importance')
    ax4.set_title('Top 10 Feature Importances')

# Metrics Summary
ax5 = fig.add_subplot(2, 3, 5)
metrics_data = {
    'Metric': ['RMSE', 'MAE', 'R²', 'MAPE (%)'],
    'Value': [
        np.sqrt(mean_squared_error(y_test, y_pred_final)),
        mean_absolute_error(y_test, y_pred_final),
        r2_score(y_test, y_pred_final),
        mean_absolute_percentage_error(y_test, y_pred_final) * 100
    ]
}
metrics_df = pd.DataFrame(metrics_data)
ax5.axis('off')
ax5.table(cellText=metrics_df.round(4).values,
          colLabels=metrics_df.columns,
          cellLoc='center',
          loc='center')
ax5.set_title('Regression Metrics')
`}

plt.tight_layout()
plt.savefig(os.path.join(PipelineConfig.OUTPUT_DIR, 'final_evaluation.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\\n  ✓ Final evaluation visualizations saved")
`;
}

function generateDeploymentCode(params: PredictiveAnalyticsParameters): string {
  return `
print("\\n=== SAVING DEPLOYMENT ARTIFACTS ===\\n")

# Save best model
model_path = os.path.join(PipelineConfig.OUTPUT_DIR, 'best_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"  ✓ Model saved: {model_path}")

# Save scaler
scaler_path = os.path.join(PipelineConfig.OUTPUT_DIR, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ Scaler saved: {scaler_path}")

# Save label encoder if classification
if label_encoder is not None:
    encoder_path = os.path.join(PipelineConfig.OUTPUT_DIR, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"  ✓ Label encoder saved: {encoder_path}")

# Save feature list
feature_path = os.path.join(PipelineConfig.OUTPUT_DIR, 'feature_list.json')
with open(feature_path, 'w') as f:
    json.dump(selected_features, f, indent=2)
print(f"  ✓ Feature list saved: {feature_path}")

# Save model metrics
metrics_path = os.path.join(PipelineConfig.OUTPUT_DIR, 'model_metrics.json')
metrics_to_save = {
    'best_model': trainer.best_model_name,
    'problem_type': PipelineConfig.PROBLEM_TYPE,
    'n_features': len(selected_features),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'results': results_df.to_dict('records')
}
with open(metrics_path, 'w') as f:
    json.dump(metrics_to_save, f, indent=2)
print(f"  ✓ Metrics saved: {metrics_path}")

# Create prediction function
def predict(new_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions on new data.
    
    Args:
        new_data: DataFrame with the same features used in training
        
    Returns:
        Predictions array
    """
    # Ensure all required features are present
    for feat in selected_features:
        if feat not in new_data.columns:
            new_data[feat] = 0
    
    # Select and order features
    X_new = new_data[selected_features].fillna(0)
    
    # Scale features
    X_new_scaled = pd.DataFrame(
        scaler.transform(X_new), 
        columns=X_new.columns, 
        index=X_new.index
    )
    
    # Make predictions
    predictions = best_model.predict(X_new_scaled)
    
    # Decode if classification
    if label_encoder is not None:
        predictions = label_encoder.inverse_transform(predictions)
    
    return predictions

def predict_proba(new_data: pd.DataFrame) -> np.ndarray:
    """
    Get prediction probabilities (for classification only).
    
    Args:
        new_data: DataFrame with the same features used in training
        
    Returns:
        Probability array
    """
    if not hasattr(best_model, 'predict_proba'):
        raise ValueError("Model does not support probability predictions")
    
    for feat in selected_features:
        if feat not in new_data.columns:
            new_data[feat] = 0
    
    X_new = new_data[selected_features].fillna(0)
    X_new_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns, index=X_new.index)
    
    return best_model.predict_proba(X_new_scaled)

print("\\n  ✓ Prediction functions created: predict(), predict_proba()")
`;
}

function generateMonitoringCode(params: PredictiveAnalyticsParameters): string {
  return `
print("\\n=== MONITORING & DRIFT DETECTION UTILITIES ===\\n")

class ModelMonitor:
    """Utilities for monitoring model performance and detecting data drift"""
    
    def __init__(self, training_data, feature_cols):
        self.training_stats = {
            'mean': training_data[feature_cols].mean(),
            'std': training_data[feature_cols].std(),
            'min': training_data[feature_cols].min(),
            'max': training_data[feature_cols].max()
        }
        self.feature_cols = feature_cols
        
    def check_data_drift(self, new_data, threshold=2.0):
        """
        Check for data drift using z-score comparison.
        
        Args:
            new_data: New data to check for drift
            threshold: Z-score threshold for drift detection
            
        Returns:
            Dictionary with drift analysis results
        """
        drift_results = {}
        
        for col in self.feature_cols:
            if col not in new_data.columns:
                continue
                
            new_mean = new_data[col].mean()
            train_mean = self.training_stats['mean'][col]
            train_std = self.training_stats['std'][col]
            
            if train_std > 0:
                z_score = abs(new_mean - train_mean) / train_std
                drift_results[col] = {
                    'z_score': z_score,
                    'drift_detected': z_score > threshold,
                    'train_mean': train_mean,
                    'new_mean': new_mean
                }
        
        drifted_features = [k for k, v in drift_results.items() if v['drift_detected']]
        
        return {
            'n_features_checked': len(drift_results),
            'n_features_drifted': len(drifted_features),
            'drifted_features': drifted_features,
            'details': drift_results
        }
    
    def check_missing_rate_change(self, new_data, threshold=0.1):
        """Check if missing rate has increased significantly"""
        missing_rate = new_data[self.feature_cols].isnull().mean()
        high_missing = missing_rate[missing_rate > threshold].index.tolist()
        
        return {
            'features_with_high_missing': high_missing,
            'missing_rates': missing_rate.to_dict()
        }
    
    def log_prediction(self, prediction, features, timestamp=None):
        """Log a prediction for future analysis"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'prediction': prediction if isinstance(prediction, (int, float, str)) else prediction.tolist(),
            'feature_summary': {
                'mean': features[self.feature_cols].mean().to_dict() if hasattr(features, 'mean') else None
            }
        }
        
        return log_entry

# Create monitor instance
monitor = ModelMonitor(X_train, selected_features)

print("  ✓ ModelMonitor class created")
print("  ✓ Available methods:")
print("    - monitor.check_data_drift(new_data)")
print("    - monitor.check_missing_rate_change(new_data)")
print("    - monitor.log_prediction(prediction, features)")

# Example usage
print("\\n--- Example: Checking drift on test data ---")
drift_check = monitor.check_data_drift(X_test)
print(f"  Features checked: {drift_check['n_features_checked']}")
print(f"  Features with drift: {drift_check['n_features_drifted']}")
if drift_check['drifted_features']:
    print(f"  Drifted features: {drift_check['drifted_features'][:5]}")
`;
}
