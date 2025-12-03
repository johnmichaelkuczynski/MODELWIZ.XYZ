import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

interface RegressionParameters {
  regressionType: string;
  targetVariable: string;
  featureVariables: string[];
  dataSource: string;
  testSize: number;
  randomState: number;
  polynomialDegree: number;
  regularizationAlpha: number;
  l1Ratio: number;
  crossValidationFolds: number;
  scaleFeatures: boolean;
  outputFormat: string;
  customInstructions: string;
}

interface ParsedRegression {
  parameters: RegressionParameters;
  pythonCode: string;
  providerUsed: string;
}

const REGRESSION_SYSTEM_PROMPT = `You are an expert Python data scientist. Your task is to analyze a natural language description of a regression problem and generate complete, production-ready Python code.

You must:
1. Extract all parameters from the user's description
2. Auto-detect the appropriate regression type if not specified
3. Generate complete Python code with all required sections

EXTRACTION RULES:
- If user says "predict X from Y and Z" → target_variable = X, feature_variables = [Y, Z]
- If user says "regression" without specifying type → default to Multiple Linear Regression
- If user mentions "classification" or binary outcomes → use Logistic Regression
- If user mentions "overfitting" or "regularization" → suggest Ridge or Lasso
- If user mentions "feature selection" → use Lasso
- If user mentions "curved" or "nonlinear" → use Polynomial Regression
- Default test_size: 0.2, random_state: 42, polynomial_degree: 2, cv_folds: 5, scale_features: true

SUPPORTED REGRESSION TYPES:
1. simple_linear - Single predictor, continuous target
2. multiple_linear - Multiple predictors, continuous target
3. polynomial - Non-linear relationships using polynomial features
4. ridge - L2 regularization for multicollinearity
5. lasso - L1 regularization for feature selection
6. elastic_net - Combined L1/L2 regularization
7. logistic - Binary or multiclass classification

You must respond with a JSON object containing:
{
  "parameters": {
    "regressionType": "string (one of the types above)",
    "targetVariable": "string",
    "featureVariables": ["array", "of", "features"],
    "dataSource": "description of data source or 'synthetic'",
    "testSize": number (0-1),
    "randomState": number,
    "polynomialDegree": number (for polynomial only),
    "regularizationAlpha": number (for ridge/lasso/elastic_net),
    "l1Ratio": number (for elastic_net only, 0-1),
    "crossValidationFolds": number,
    "scaleFeatures": boolean,
    "outputFormat": "py" or "ipynb",
    "customInstructions": "any special requirements"
  },
  "pythonCode": "complete Python code as a string"
}

REQUIRED CODE SECTIONS (in order):
1. Header with docstring (model type, target, features, timestamp)
2. All imports (numpy, pandas, sklearn, matplotlib, seaborn, etc.)
3. Data loading/generation
4. Exploratory data analysis (shape, head, describe, missing values, correlations)
5. Data preprocessing (feature/target split, scaling, train/test split)
6. Model training with cross-validation
7. Model evaluation (R², RMSE, MAE for regression; accuracy, precision, recall for classification)
8. Coefficient interpretation with feature importance
9. Visualizations (actual vs predicted, residuals, feature coefficients)
10. Prediction function for new data
11. Model persistence (joblib save/load)

Ensure the code is:
- Syntactically valid Python 3.8+
- Immediately executable
- Well-commented
- Following scikit-learn best practices`;

async function callGrok(prompt: string): Promise<string> {
  const grokApiKey = process.env.GROK_API_KEY;
  if (!grokApiKey) throw new Error('GROK_API_KEY not configured');
  
  const response = await fetch('https://api.x.ai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${grokApiKey}`
    },
    body: JSON.stringify({
      model: 'grok-3-latest',
      messages: [
        { role: 'system', content: REGRESSION_SYSTEM_PROMPT },
        { role: 'user', content: prompt }
      ],
      temperature: 0.3,
      max_tokens: 16000
    })
  });
  
  if (!response.ok) {
    throw new Error(`Grok API error: ${response.status}`);
  }
  
  const data = await response.json();
  return data.choices[0].message.content;
}

async function callOpenAI(prompt: string): Promise<string> {
  const response = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [
      { role: 'system', content: REGRESSION_SYSTEM_PROMPT },
      { role: 'user', content: prompt }
    ],
    temperature: 0.3,
    max_tokens: 16000
  });
  return response.choices[0].message?.content || '';
}

async function callAnthropic(prompt: string): Promise<string> {
  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 16000,
    system: REGRESSION_SYSTEM_PROMPT,
    messages: [{ role: 'user', content: prompt }]
  });
  const textBlock = response.content.find((block: any) => block.type === 'text');
  return textBlock ? (textBlock as any).text : '';
}

async function callDeepSeek(prompt: string): Promise<string> {
  const deepseekApiKey = process.env.DEEPSEEK_API_KEY;
  if (!deepseekApiKey) throw new Error('DEEPSEEK_API_KEY not configured');
  
  const response = await fetch('https://api.deepseek.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${deepseekApiKey}`
    },
    body: JSON.stringify({
      model: 'deepseek-chat',
      messages: [
        { role: 'system', content: REGRESSION_SYSTEM_PROMPT },
        { role: 'user', content: prompt }
      ],
      temperature: 0.3,
      max_tokens: 16000
    })
  });
  
  if (!response.ok) {
    throw new Error(`DeepSeek API error: ${response.status}`);
  }
  
  const data = await response.json();
  return data.choices[0].message.content;
}

async function callPerplexity(prompt: string): Promise<string> {
  const perplexityApiKey = process.env.PERPLEXITY_API_KEY;
  if (!perplexityApiKey) throw new Error('PERPLEXITY_API_KEY not configured');
  
  const response = await fetch('https://api.perplexity.ai/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${perplexityApiKey}`
    },
    body: JSON.stringify({
      model: 'sonar-pro',
      messages: [
        { role: 'system', content: REGRESSION_SYSTEM_PROMPT },
        { role: 'user', content: prompt }
      ],
      temperature: 0.3,
      max_tokens: 16000
    })
  });
  
  if (!response.ok) {
    throw new Error(`Perplexity API error: ${response.status}`);
  }
  
  const data = await response.json();
  return data.choices[0].message.content;
}

export async function parseRegressionDescription(
  description: string,
  customInstructions: string = '',
  provider: string = 'grok'
): Promise<ParsedRegression> {
  const fullPrompt = customInstructions 
    ? `${description}\n\nAdditional Instructions:\n${customInstructions}`
    : description;
  
  let response: string;
  let providerUsed: string;
  
  try {
    switch (provider.toLowerCase()) {
      case 'openai':
      case 'zhi1':
        response = await callOpenAI(fullPrompt);
        providerUsed = 'ZHI 1 (GPT-4o)';
        break;
      case 'anthropic':
      case 'zhi2':
        response = await callAnthropic(fullPrompt);
        providerUsed = 'ZHI 2 (Claude)';
        break;
      case 'deepseek':
      case 'zhi3':
        response = await callDeepSeek(fullPrompt);
        providerUsed = 'ZHI 3 (DeepSeek)';
        break;
      case 'perplexity':
      case 'zhi4':
        response = await callPerplexity(fullPrompt);
        providerUsed = 'ZHI 4 (Perplexity)';
        break;
      case 'grok':
      case 'zhi5':
      default:
        response = await callGrok(fullPrompt);
        providerUsed = 'ZHI 5 (Grok)';
        break;
    }
    
    // Parse JSON from response
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Could not parse JSON response from LLM');
    }
    
    const parsed = JSON.parse(jsonMatch[0]);
    
    return {
      parameters: parsed.parameters,
      pythonCode: parsed.pythonCode,
      providerUsed
    };
  } catch (error: any) {
    console.error('Error parsing regression description:', error);
    throw new Error(`Failed to generate regression model: ${error.message}`);
  }
}

export function generateRegressionPythonCode(params: RegressionParameters): string {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '').slice(0, 15);
  const features = params.featureVariables.join("', '");
  
  let modelImport = '';
  let modelClass = '';
  let scoringMetric = 'r2';
  let isClassification = false;
  
  switch (params.regressionType) {
    case 'simple_linear':
    case 'multiple_linear':
      modelImport = 'from sklearn.linear_model import LinearRegression';
      modelClass = 'LinearRegression()';
      break;
    case 'polynomial':
      modelImport = 'from sklearn.linear_model import LinearRegression\nfrom sklearn.preprocessing import PolynomialFeatures';
      modelClass = 'LinearRegression()';
      break;
    case 'ridge':
      modelImport = 'from sklearn.linear_model import Ridge';
      modelClass = `Ridge(alpha=${params.regularizationAlpha})`;
      break;
    case 'lasso':
      modelImport = 'from sklearn.linear_model import Lasso';
      modelClass = `Lasso(alpha=${params.regularizationAlpha})`;
      break;
    case 'elastic_net':
      modelImport = 'from sklearn.linear_model import ElasticNet';
      modelClass = `ElasticNet(alpha=${params.regularizationAlpha}, l1_ratio=${params.l1Ratio})`;
      break;
    case 'logistic':
      modelImport = 'from sklearn.linear_model import LogisticRegression';
      modelClass = 'LogisticRegression(max_iter=1000)';
      scoringMetric = 'accuracy';
      isClassification = true;
      break;
    default:
      modelImport = 'from sklearn.linear_model import LinearRegression';
      modelClass = 'LinearRegression()';
  }
  
  const code = `"""
Regression Model: ${params.regressionType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
Generated by Cognitive Analysis Platform - Data Science Panel
Target Variable: ${params.targetVariable}
Features: ['${features}']
Generated on: ${new Date().toISOString()}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler${params.regressionType === 'polynomial' ? ', PolynomialFeatures' : ''}
${modelImport}
from sklearn.metrics import ${isClassification 
  ? 'accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve' 
  : 'mean_squared_error, r2_score, mean_absolute_error'}
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = ${params.randomState}
TEST_SIZE = ${params.testSize}
CV_FOLDS = ${params.crossValidationFolds}
SCALE_FEATURES = ${params.scaleFeatures ? 'True' : 'False'}

# Feature and target column names
FEATURE_COLUMNS = ['${features}']
TARGET_COLUMN = '${params.targetVariable}'

print("=" * 70)
print("REGRESSION MODEL ANALYSIS")
print("Model Type: ${params.regressionType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}")
print("=" * 70)

# ============================================================================
# DATA LOADING / GENERATION
# ============================================================================
print("\\n--- DATA LOADING ---")

# ${params.dataSource === 'synthetic' || params.dataSource.includes('generate') || params.dataSource.includes('synthetic') 
  ? 'Generating synthetic data based on specifications' 
  : 'Load your data here - replace with actual file path'}

${params.dataSource === 'synthetic' || params.dataSource.includes('generate') || params.dataSource.includes('synthetic') ? `
# Synthetic data generation
np.random.seed(RANDOM_STATE)
n_samples = 500

# Generate features
${params.featureVariables.map((f, i) => `${f} = np.random.uniform(0, 100, n_samples)`).join('\n')}

# Generate target variable with correlations to features
${params.targetVariable} = ${params.featureVariables.map((f, i) => `${f} * ${(i + 1) * 0.5}`).join(' + ')} + np.random.normal(0, 10, n_samples)

df = pd.DataFrame({
${params.featureVariables.map(f => `    '${f}': ${f},`).join('\n')}
    '${params.targetVariable}': ${params.targetVariable}
})
` : `
# Load data from file (replace with your actual file path)
# df = pd.read_csv('your_data.csv')

# For demonstration, creating sample data
np.random.seed(RANDOM_STATE)
n_samples = 500
data = {col: np.random.randn(n_samples) * 10 + 50 for col in FEATURE_COLUMNS}
data[TARGET_COLUMN] = sum(data[col] * (i+1) * 0.3 for i, col in enumerate(FEATURE_COLUMNS)) + np.random.randn(n_samples) * 5
df = pd.DataFrame(data)
`}

print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\\n--- EXPLORATORY DATA ANALYSIS ---")

print("\\nFirst 5 Rows:")
print(df.head())

print("\\nDescriptive Statistics:")
print(df.describe().round(2))

print("\\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values found")
else:
    print(missing[missing > 0])

print("\\nCorrelation with Target:")
correlations = df.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)
print(correlations.round(4))

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("\\n--- DATA PREPROCESSING ---")

# Define features and target
X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

# Handle missing values
X = X.fillna(X.median())

${params.regressionType === 'polynomial' ? `
# Polynomial feature transformation
poly = PolynomialFeatures(degree=${params.polynomialDegree}, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(FEATURE_COLUMNS)
X = pd.DataFrame(X_poly, columns=feature_names)
print(f"Polynomial features created: {len(feature_names)} features from {len(FEATURE_COLUMNS)} original")
` : ''}

# Feature scaling
scaler = StandardScaler()
if SCALE_FEATURES:
    X_scaled = scaler.fit_transform(X)
    print("Features scaled using StandardScaler")
else:
    X_scaled = X.values
    print("Feature scaling disabled")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"\\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================================================
# MODEL TRAINING
# ============================================================================
print("\\n--- MODEL TRAINING ---")

model = ${modelClass}
model.fit(X_train, y_train)
print("Model trained successfully")

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=CV_FOLDS, scoring='${scoringMetric}')
print(f"\\n{CV_FOLDS}-Fold Cross-Validation:")
print(f"  Scores: {cv_scores.round(4)}")
print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# MODEL EVALUATION
# ============================================================================
print("\\n--- MODEL EVALUATION ---")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

${isClassification ? `
print("\\n=== CLASSIFICATION METRICS ===")
print("\\nTraining Set:")
print(f"  Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"  Precision: {precision_score(y_train, y_pred_train, average='weighted'):.4f}")
print(f"  Recall: {recall_score(y_train, y_pred_train, average='weighted'):.4f}")
print(f"  F1 Score: {f1_score(y_train, y_pred_train, average='weighted'):.4f}")

print("\\nTest Set:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"  F1 Score: {f1_score(y_test, y_pred_test, average='weighted'):.4f}")

print("\\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test))

print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))
` : `
print("\\n=== REGRESSION METRICS ===")

# Training metrics
train_r2 = r2_score(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)

print("\\nTraining Set:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE: {train_mae:.4f}")

# Test metrics
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\\nTest Set:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE: {test_mae:.4f}")

# Overfitting check
if train_r2 - test_r2 > 0.1:
    print("\\n⚠️  WARNING: Possible overfitting detected")
    print(f"   (Train R² - Test R² = {train_r2 - test_r2:.4f})")
else:
    print("\\n✓ No significant overfitting detected")
`}

# ============================================================================
# COEFFICIENT INTERPRETATION
# ============================================================================
print("\\n--- COEFFICIENT INTERPRETATION ---")

${params.regressionType === 'polynomial' ? 'coef_feature_names = feature_names' : 'coef_feature_names = FEATURE_COLUMNS'}

coef_df = pd.DataFrame({
    'Feature': coef_feature_names,
    'Coefficient': model.coef_ if len(model.coef_.shape) == 1 else model.coef_[0]
})
coef_df['Abs_Coefficient'] = abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

print("\\n=== FEATURE IMPORTANCE (by coefficient magnitude) ===")
print(coef_df.to_string(index=False))

print(f"\\nIntercept: {model.intercept_${isClassification ? '[0]' : ''}:.4f}")

${params.regressionType === 'lasso' ? `
# Lasso feature selection analysis
zero_coefs = (coef_df['Coefficient'] == 0).sum()
print(f"\\nLasso eliminated {zero_coefs} features (coefficient = 0)")
print("Remaining features:")
print(coef_df[coef_df['Coefficient'] != 0]['Feature'].tolist())
` : ''}

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\\n--- GENERATING VISUALIZATIONS ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

${isClassification ? `
# Plot 1: Confusion Matrix Heatmap
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_title('Confusion Matrix')

# Plot 2: ROC Curve (if binary)
if len(np.unique(y)) == 2:
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[0, 1].plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'r--')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
else:
    axes[0, 1].text(0.5, 0.5, 'ROC for multiclass\\nnot shown', ha='center', va='center')
    axes[0, 1].set_title('ROC Curve (Binary Only)')
` : `
# Plot 1: Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', s=50)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Values')
axes[0, 0].set_ylabel('Predicted Values')
axes[0, 0].set_title(f'Actual vs Predicted (R² = {test_r2:.4f})')
axes[0, 0].legend()

# Plot 2: Residuals Distribution
residuals = y_test - y_pred_test
axes[0, 1].hist(residuals, bins=30, edgecolor='k', alpha=0.7, color='steelblue')
axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 1].axvline(x=residuals.mean(), color='orange', linestyle='-', lw=2, label=f'Mean: {residuals.mean():.2f}')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Residuals Distribution')
axes[0, 1].legend()
`}

# Plot 3: Residuals vs Predicted (or Feature Importance for classification)
${isClassification ? `
top_features = coef_df.head(10)
colors = ['forestgreen' if c > 0 else 'crimson' for c in top_features['Coefficient']]
axes[1, 0].barh(top_features['Feature'], top_features['Coefficient'], color=colors, edgecolor='k')
axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1, 0].set_xlabel('Coefficient Value')
axes[1, 0].set_title('Top 10 Feature Coefficients')
` : `
axes[1, 0].scatter(y_pred_test, residuals, alpha=0.6, edgecolors='k', s=50)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residuals vs Predicted (Homoscedasticity Check)')
`}

# Plot 4: Feature Coefficients
top_coefs = coef_df.head(10)
colors = ['forestgreen' if c > 0 else 'crimson' for c in top_coefs['Coefficient']]
bars = axes[1, 1].barh(top_coefs['Feature'], top_coefs['Coefficient'], color=colors, edgecolor='k')
axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1, 1].set_xlabel('Coefficient Value')
axes[1, 1].set_title('Feature Coefficients (Top 10)')

plt.tight_layout()
plt.savefig('regression_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved as 'regression_analysis.png'")

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_new(feature_values):
    """
    Make predictions on new data.
    
    Parameters:
    -----------
    feature_values : list or array
        Values for features in order: ${params.featureVariables.map(f => `'${f}'`).join(', ')}
    
    Returns:
    --------
    prediction : float or int
        Model prediction
    """
    feature_array = np.array(feature_values).reshape(1, -1)
    ${params.regressionType === 'polynomial' ? `
    feature_poly = poly.transform(feature_array)
    if SCALE_FEATURES:
        feature_scaled = scaler.transform(feature_poly)
    else:
        feature_scaled = feature_poly
    ` : `
    if SCALE_FEATURES:
        feature_scaled = scaler.transform(feature_array)
    else:
        feature_scaled = feature_array
    `}
    prediction = model.predict(feature_scaled)
    return prediction[0]

# Example usage
print("\\n--- PREDICTION FUNCTION READY ---")
print("Use predict_new([${params.featureVariables.map(() => 'value').join(', ')}]) to make predictions")
print(f"Feature order: {FEATURE_COLUMNS}")

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\\n--- SAVING MODEL ---")

joblib.dump(model, 'regression_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
${params.regressionType === 'polynomial' ? "joblib.dump(poly, 'polynomial_transformer.pkl')" : ''}

print("Model saved as 'regression_model.pkl'")
print("Scaler saved as 'feature_scaler.pkl'")
${params.regressionType === 'polynomial' ? "print(\"Polynomial transformer saved as 'polynomial_transformer.pkl'\")" : ''}

print("\\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

# ============================================================================
# HOW TO LOAD AND USE SAVED MODEL
# ============================================================================
"""
To use this model later:

import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load('regression_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
${params.regressionType === 'polynomial' ? "poly = joblib.load('polynomial_transformer.pkl')" : ''}

# Prepare new data
new_data = np.array([[${params.featureVariables.map(() => '50.0').join(', ')}]])
${params.regressionType === 'polynomial' ? 'new_data = poly.transform(new_data)' : ''}
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)[0]
print(f"Predicted value: {prediction:.4f}")
"""
`;

  return code;
}
