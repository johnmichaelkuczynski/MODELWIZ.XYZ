import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const anthropicApiKey = process.env.ANTHROPIC_API_KEY;
const deepseekApiKey = process.env.DEEPSEEK_API_KEY;
const perplexityApiKey = process.env.PERPLEXITY_API_KEY;
const grokApiKey = process.env.GROK_API_KEY;

interface ForecastingParameters {
  modelType: string;
  dataSource: string;
  dateColumn: string;
  valueColumn: string;
  forecastHorizon: number;
  frequency: string;
  seasonalPeriod: number;
  confidenceLevel: number;
  trainTestSplit: number;
  exogenousVariables: string[];
  holidayCountry: string | null;
  decompositionType: string;
  autoModel: boolean;
  compareModels: boolean;
  generateSyntheticData: boolean;
  syntheticDataDescription: string | null;
}

const FORECASTING_PARSING_PROMPT = `You are a time series forecasting expert. Analyze the following natural language description and extract forecasting parameters.

SUPPORTED MODEL TYPES:
- "arima" - Non-seasonal ARIMA
- "sarima" - Seasonal ARIMA (most common for seasonal data)
- "sarimax" - SARIMA with exogenous variables
- "var" - Vector AutoRegression (multivariate)
- "ses" - Simple Exponential Smoothing
- "holt" - Holt's Linear Method
- "holtwinters" - Holt-Winters Triple Exponential Smoothing
- "ets" - Automatic Exponential Smoothing
- "prophet" - Facebook Prophet
- "auto" - Automatic model selection with comparison

FREQUENCY CODES:
- "H" - Hourly (seasonal_period: 24)
- "D" - Daily (seasonal_period: 7)
- "W" - Weekly (seasonal_period: 52)
- "MS" or "M" - Monthly (seasonal_period: 12)
- "Q" - Quarterly (seasonal_period: 4)
- "Y" or "A" - Yearly (seasonal_period: 1)

PARSING RULES:
- "forecast", "predict future", "next N periods" → Forecasting task
- "daily", "weekly", "monthly", "quarterly", "yearly", "hourly" → Set frequency
- "seasonal", "seasonality", "repeating pattern" → Include seasonal component
- "holidays", "special events" → Use Prophet with holiday effects
- "multiple variables", "multivariate" → Use VAR
- "external factors", "regressors", "predictors" → Use SARIMAX
- "auto", "best model", "compare" → Auto mode with model comparison
- "generate data", "synthetic", "simulate" → Generate synthetic data

Extract and respond with ONLY a valid JSON object (no markdown, no code blocks):

{
  "modelType": "sarima",
  "dataSource": "file_path or 'synthetic'",
  "dateColumn": "date column name",
  "valueColumn": "target column name",
  "forecastHorizon": 12,
  "frequency": "MS",
  "seasonalPeriod": 12,
  "confidenceLevel": 0.95,
  "trainTestSplit": 0.8,
  "exogenousVariables": [],
  "holidayCountry": null,
  "decompositionType": "additive",
  "autoModel": false,
  "compareModels": false,
  "generateSyntheticData": true,
  "syntheticDataDescription": "description of synthetic data to generate"
}

Description to analyze:
`;

export async function parseForecastingDescription(
  description: string,
  customInstructions: string,
  llmProvider: string = 'grok'
): Promise<{ parameters: ForecastingParameters; providerUsed: string }> {
  const fullPrompt = FORECASTING_PARSING_PROMPT + description + 
    (customInstructions ? `\n\nAdditional instructions: ${customInstructions}` : '');

  let response: string = '';
  let providerUsed = llmProvider;

  try {
    if (llmProvider === 'grok' && grokApiKey) {
      const grokClient = new OpenAI({
        apiKey: grokApiKey,
        baseURL: 'https://api.x.ai/v1'
      });
      const result = await grokClient.chat.completions.create({
        model: 'grok-3-latest',
        messages: [{ role: 'user', content: fullPrompt }],
        temperature: 0.3,
        max_tokens: 2000
      });
      response = result.choices[0]?.message?.content || '';
      providerUsed = 'Zhi 5 (Grok)';
    } else if (llmProvider === 'openai' && process.env.OPENAI_API_KEY) {
      const result = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [{ role: 'user', content: fullPrompt }],
        temperature: 0.3,
        max_tokens: 2000
      });
      response = result.choices[0]?.message?.content || '';
      providerUsed = 'Zhi 1 (GPT-4o)';
    } else if (llmProvider === 'anthropic' && anthropicApiKey) {
      const anthropicResponse = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': anthropicApiKey,
          'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 2000,
          messages: [{ role: 'user', content: fullPrompt }]
        })
      });
      const data = await anthropicResponse.json();
      response = data.content?.[0]?.text || '';
      providerUsed = 'Zhi 2 (Claude)';
    } else if (llmProvider === 'deepseek' && deepseekApiKey) {
      const deepseekClient = new OpenAI({
        apiKey: deepseekApiKey,
        baseURL: 'https://api.deepseek.com'
      });
      const result = await deepseekClient.chat.completions.create({
        model: 'deepseek-chat',
        messages: [{ role: 'user', content: fullPrompt }],
        temperature: 0.3,
        max_tokens: 2000
      });
      response = result.choices[0]?.message?.content || '';
      providerUsed = 'Zhi 3 (DeepSeek)';
    } else {
      if (grokApiKey) {
        const grokClient = new OpenAI({
          apiKey: grokApiKey,
          baseURL: 'https://api.x.ai/v1'
        });
        const result = await grokClient.chat.completions.create({
          model: 'grok-3-latest',
          messages: [{ role: 'user', content: fullPrompt }],
          temperature: 0.3,
          max_tokens: 2000
        });
        response = result.choices[0]?.message?.content || '';
        providerUsed = 'Zhi 5 (Grok)';
      } else {
        throw new Error('No LLM provider available');
      }
    }

    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Could not parse LLM response as JSON');
    }

    const parameters = JSON.parse(jsonMatch[0]) as ForecastingParameters;

    if (!parameters.modelType) parameters.modelType = 'auto';
    if (!parameters.forecastHorizon) parameters.forecastHorizon = 12;
    if (!parameters.frequency) parameters.frequency = 'MS';
    if (!parameters.seasonalPeriod) {
      parameters.seasonalPeriod = getSeasonalPeriod(parameters.frequency);
    }
    if (!parameters.confidenceLevel) parameters.confidenceLevel = 0.95;
    if (!parameters.trainTestSplit) parameters.trainTestSplit = 0.8;
    if (!parameters.decompositionType) parameters.decompositionType = 'additive';
    if (!parameters.dateColumn) parameters.dateColumn = 'date';
    if (!parameters.valueColumn) parameters.valueColumn = 'value';

    return { parameters, providerUsed };
  } catch (error: any) {
    console.error('Forecasting parsing error:', error);
    throw new Error(`Failed to parse forecasting description: ${error.message}`);
  }
}

function getSeasonalPeriod(frequency: string): number {
  const seasonalMap: { [key: string]: number } = {
    'H': 24,
    'D': 7,
    'W': 52,
    'M': 12,
    'MS': 12,
    'Q': 4,
    'Y': 1,
    'A': 1
  };
  return seasonalMap[frequency.toUpperCase()] || 12;
}

export function generateForecastingPythonCode(params: ForecastingParameters): string {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const modelTypeDisplay = params.modelType.toUpperCase();
  
  let code = `"""
Statistical Forecasting Model: ${modelTypeDisplay}${params.compareModels ? ' (with Model Comparison)' : ''}
Generated by Zhi Data Science Platform
Target Variable: ${params.valueColumn}
Forecast Horizon: ${params.forecastHorizon} periods
Frequency: ${params.frequency}
Generated on: ${new Date().toISOString()}

Required packages:
pip install numpy pandas matplotlib seaborn statsmodels scipy scikit-learn
Optional: pip install prophet pmdarima holidays
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical modeling
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats

# Auto ARIMA (optional)
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    print("Note: pmdarima not installed. Install with: pip install pmdarima")

# Prophet (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Note: Prophet not installed. Install with: pip install prophet")

# Holidays (optional)
try:
    import holidays as hld
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    print("Note: holidays not installed. Install with: pip install holidays")

print("=" * 70)
print("TIME SERIES FORECASTING ANALYSIS")
print("=" * 70)

`;

  if (params.generateSyntheticData || params.dataSource === 'synthetic') {
    code += generateSyntheticDataCode(params);
  } else {
    code += generateDataLoadingCode(params);
  }

  code += generateEDACode(params);
  code += generateStationarityTestsCode(params);
  code += generateDecompositionCode(params);
  code += generateACFPACFCode(params);
  code += generateTrainTestSplitCode(params);

  if (params.autoModel || params.compareModels || params.modelType === 'auto') {
    code += generateModelComparisonCode(params);
  } else {
    code += generateSpecificModelCode(params);
  }

  code += generateForecastCode(params);
  code += generateVisualizationsCode(params);
  code += generateForecastFunctionCode(params);
  code += generateModelPersistenceCode(params);

  return code;
}

function generateSyntheticDataCode(params: ForecastingParameters): string {
  return `
# --- DATA GENERATION ---
print("\\n" + "-" * 70)
print("GENERATING SYNTHETIC TIME SERIES DATA")
print("-" * 70)

np.random.seed(42)

# Generate date range
${params.frequency === 'MS' || params.frequency === 'M' ? `
# Monthly data
dates = pd.date_range(start='2018-01-01', periods=84, freq='MS')  # 7 years
n_periods = len(dates)
` : params.frequency === 'D' ? `
# Daily data
dates = pd.date_range(start='2022-01-01', periods=730, freq='D')  # 2 years
n_periods = len(dates)
` : params.frequency === 'W' ? `
# Weekly data
dates = pd.date_range(start='2020-01-01', periods=208, freq='W')  # 4 years
n_periods = len(dates)
` : `
# Default monthly data
dates = pd.date_range(start='2018-01-01', periods=84, freq='MS')
n_periods = len(dates)
`}

print(f"Generating {n_periods} observations")

# Base value
base_value = 1000

# Trend component (${params.decompositionType === 'multiplicative' ? 'multiplicative' : 'additive'} growth)
${params.decompositionType === 'multiplicative' ? `
# Multiplicative trend: ~5% annual growth
monthly_growth = (1.05) ** (1/${params.seasonalPeriod}) - 1
trend = base_value * (1 + monthly_growth) ** np.arange(n_periods)
` : `
# Additive trend: linear growth
trend = base_value + 10 * np.arange(n_periods)
`}

# Seasonal component
seasonal_period = ${params.seasonalPeriod}
${params.decompositionType === 'multiplicative' ? `
# Multiplicative seasonality
seasonal_factors = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_periods) / seasonal_period)
values = trend * seasonal_factors
` : `
# Additive seasonality
seasonal_amplitude = base_value * 0.2
seasonal = seasonal_amplitude * np.sin(2 * np.pi * np.arange(n_periods) / seasonal_period)
values = trend + seasonal
`}

# Add noise
noise_level = ${params.decompositionType === 'multiplicative' ? '0.05' : 'base_value * 0.05'}
${params.decompositionType === 'multiplicative' ? `
noise = np.random.normal(0, noise_level, n_periods)
values = values * (1 + noise)
` : `
noise = np.random.normal(0, noise_level, n_periods)
values = values + noise
`}

# Ensure non-negative
values = np.maximum(values, 0)

# Create DataFrame
df = pd.DataFrame({
    '${params.dateColumn}': dates,
    '${params.valueColumn}': values.round(2)
})
df = df.set_index('${params.dateColumn}')
df.index.freq = '${params.frequency}'

series = df['${params.valueColumn}']

print(f"\\nData generated successfully!")
print(f"Date range: {series.index.min()} to {series.index.max()}")
print(f"Number of observations: {len(series)}")

`;
}

function generateDataLoadingCode(params: ForecastingParameters): string {
  return `
# --- DATA LOADING ---
print("\\n" + "-" * 70)
print("LOADING DATA")
print("-" * 70)

# Load your data (modify path as needed)
# df = pd.read_csv('your_data.csv', parse_dates=['${params.dateColumn}'])

# Example: Creating sample data - replace with your actual data loading
print("NOTE: Replace this section with your actual data loading code")
print("Example: df = pd.read_csv('your_data.csv', parse_dates=['${params.dateColumn}'])")

# Placeholder synthetic data - REPLACE WITH YOUR DATA
np.random.seed(42)
dates = pd.date_range(start='2018-01-01', periods=84, freq='${params.frequency}')
values = 1000 + 5 * np.arange(len(dates)) + 200 * np.sin(2 * np.pi * np.arange(len(dates)) / ${params.seasonalPeriod})
values += np.random.normal(0, 50, len(dates))

df = pd.DataFrame({
    '${params.dateColumn}': dates,
    '${params.valueColumn}': values.round(2)
})
df = df.set_index('${params.dateColumn}')

# Ensure datetime index
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

# Set frequency
if df.index.freq is None:
    df = df.asfreq('${params.frequency}')

series = df['${params.valueColumn}']

print(f"\\nDataset loaded: {len(series)} observations")
print(f"Date range: {series.index.min()} to {series.index.max()}")
print(f"Frequency: {series.index.freq}")

`;
}

function generateEDACode(params: ForecastingParameters): string {
  return `
# --- EXPLORATORY DATA ANALYSIS ---
print("\\n" + "-" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("-" * 70)

print("\\n=== TIME SERIES SUMMARY ===")
print(f"Number of observations: {len(series)}")
print(f"Start date: {series.index.min()}")
print(f"End date: {series.index.max()}")
print(f"Frequency: {series.index.freq}")

print("\\n=== DESCRIPTIVE STATISTICS ===")
print(series.describe().round(2))

# Check for missing values
missing = series.isnull().sum()
if missing > 0:
    print(f"\\nMissing values detected: {missing} ({missing/len(series)*100:.1f}%)")
    print("Missing values will be interpolated.")
    series = series.interpolate(method='time')
else:
    print("\\nNo missing values detected")

# Check for duplicates
duplicates = series.index.duplicated().sum()
if duplicates > 0:
    print(f"Duplicate timestamps detected: {duplicates}")
    series = series[~series.index.duplicated(keep='first')]

# Basic characteristics
print("\\n=== TIME SERIES CHARACTERISTICS ===")
print(f"Mean: {series.mean():.2f}")
print(f"Std Dev: {series.std():.2f}")
print(f"Coefficient of Variation: {(series.std()/series.mean())*100:.1f}%")
print(f"Min: {series.min():.2f} (at {series.idxmin()})")
print(f"Max: {series.max():.2f} (at {series.idxmax()})")

# Trend analysis
n_splits = min(4, len(series) // 20)
if n_splits >= 2:
    splits = np.array_split(series.values, n_splits)
    means = [s.mean() for s in splits]
    variances = [s.var() for s in splits]
    
    print(f"\\nMean across {n_splits} periods: {[f'{m:.2f}' for m in means]}")
    print(f"Variance across {n_splits} periods: {[f'{v:.2f}' for v in variances]}")
    
    mean_change = (max(means) - min(means)) / np.mean(means) * 100
    var_change = (max(variances) - min(variances)) / np.mean(variances) * 100 if np.mean(variances) > 0 else 0
    
    if mean_change > 50:
        print("  -> Significant mean change detected (possible trend)")
    if var_change > 100:
        print("  -> Significant variance change detected (consider multiplicative model)")

# Seasonality detection
seasonal_period = ${params.seasonalPeriod}
print(f"\\n=== SEASONALITY ANALYSIS (period={seasonal_period}) ===")
if len(series) >= 2 * seasonal_period:
    autocorr = series.autocorr(lag=seasonal_period)
    print(f"Autocorrelation at seasonal lag ({seasonal_period}): {autocorr:.4f}")
    if abs(autocorr) > 0.5:
        print("  -> Strong seasonal pattern detected")
    elif abs(autocorr) > 0.3:
        print("  -> Moderate seasonal pattern detected")
    else:
        print("  -> Weak or no seasonal pattern detected")
else:
    print(f"Insufficient data for seasonality analysis (need >= {2*seasonal_period} observations)")

`;
}

function generateStationarityTestsCode(params: ForecastingParameters): string {
  return `
# --- STATIONARITY TESTS ---
print("\\n" + "-" * 70)
print("STATIONARITY ANALYSIS")
print("-" * 70)

def adf_test(ts, name='Series'):
    """Augmented Dickey-Fuller test for stationarity"""
    result = adfuller(ts.dropna(), autolag='AIC')
    print(f"\\n=== ADF Test: {name} ===")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"P-value: {result[1]:.4f}")
    print(f"Lags Used: {result[2]}")
    print(f"Observations: {result[3]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    
    if result[1] < 0.05:
        print("RESULT: Series is STATIONARY (reject null hypothesis)")
        return True
    else:
        print("RESULT: Series is NON-STATIONARY (fail to reject null hypothesis)")
        return False

def kpss_test(ts, name='Series'):
    """KPSS test for stationarity"""
    result = kpss(ts.dropna(), regression='c', nlags='auto')
    print(f"\\n=== KPSS Test: {name} ===")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"P-value: {result[1]:.4f}")
    print(f"Lags Used: {result[2]}")
    
    if result[1] > 0.05:
        print("RESULT: Series is STATIONARY (fail to reject null hypothesis)")
        return True
    else:
        print("RESULT: Series is NON-STATIONARY (reject null hypothesis)")
        return False

# Run tests on original series
is_stationary_adf = adf_test(series, 'Original Series')
is_stationary_kpss = kpss_test(series, 'Original Series')

# Determine differencing needed
differencing_order = 0
test_series = series.copy()

if not is_stationary_adf or not is_stationary_kpss:
    print("\\n--- Testing First Difference ---")
    diff_1 = series.diff().dropna()
    is_diff1_stationary = adf_test(diff_1, 'First Difference')
    
    if is_diff1_stationary:
        differencing_order = 1
        test_series = diff_1
    else:
        print("\\n--- Testing Second Difference ---")
        diff_2 = series.diff().diff().dropna()
        is_diff2_stationary = adf_test(diff_2, 'Second Difference')
        if is_diff2_stationary:
            differencing_order = 2
            test_series = diff_2
        else:
            print("Series may require transformation (log, Box-Cox)")
            differencing_order = 1

print(f"\\n=== STATIONARITY CONCLUSION ===")
print(f"Recommended differencing order (d): {differencing_order}")

`;
}

function generateDecompositionCode(params: ForecastingParameters): string {
  return `
# --- TIME SERIES DECOMPOSITION ---
print("\\n" + "-" * 70)
print("TIME SERIES DECOMPOSITION")
print("-" * 70)

seasonal_period = ${params.seasonalPeriod}
decomposition_available = False

# Determine decomposition type
rolling_mean = series.rolling(window=seasonal_period).mean()
rolling_std = series.rolling(window=seasonal_period).std()

if len(rolling_mean.dropna()) > 10:
    corr_mean_std = rolling_mean.corr(rolling_std)
    print(f"\\nCorrelation between rolling mean and std: {corr_mean_std:.4f}")
    
    if corr_mean_std > 0.5:
        decomp_type = 'multiplicative'
        print("  -> Using MULTIPLICATIVE decomposition (variance increases with level)")
    else:
        decomp_type = 'additive'
        print("  -> Using ADDITIVE decomposition (constant variance)")
else:
    decomp_type = '${params.decompositionType}'
    print(f"  -> Using {decomp_type.upper()} decomposition (default)")

# Perform decomposition
if len(series) >= 2 * seasonal_period:
    try:
        # STL Decomposition (more robust)
        stl = STL(series, period=seasonal_period, robust=True)
        stl_result = stl.fit()
        
        trend_strength = 1 - (stl_result.resid.var() / (stl_result.trend + stl_result.resid).var())
        seasonal_strength = 1 - (stl_result.resid.var() / (stl_result.seasonal + stl_result.resid).var())
        
        print(f"\\n=== STL DECOMPOSITION RESULTS ===")
        print(f"Trend strength: {trend_strength:.4f} (0=weak, 1=strong)")
        print(f"Seasonal strength: {seasonal_strength:.4f} (0=weak, 1=strong)")
        
        if trend_strength > 0.6:
            print("  -> Strong trend component detected")
        if seasonal_strength > 0.6:
            print("  -> Strong seasonal component detected")
        
        # Classical decomposition for reference
        classical_result = seasonal_decompose(series, model=decomp_type, period=seasonal_period)
        
        decomposition_available = True
    except Exception as e:
        print(f"Decomposition failed: {e}")
        decomposition_available = False
else:
    print(f"Insufficient data for seasonal decomposition (need >= {2*seasonal_period} observations)")
    decomposition_available = False

`;
}

function generateACFPACFCode(params: ForecastingParameters): string {
  return `
# --- ACF/PACF ANALYSIS ---
print("\\n" + "-" * 70)
print("ACF/PACF ANALYSIS")
print("-" * 70)

# Compute ACF and PACF
n_lags = min(40, len(test_series) // 4)

acf_values = acf(test_series.dropna(), nlags=n_lags, fft=True)
pacf_values = pacf(test_series.dropna(), nlags=n_lags, method='ywm')

# Confidence interval (95%)
ci = 1.96 / np.sqrt(len(test_series))

# Identify significant lags
significant_acf = np.where(np.abs(acf_values[1:]) > ci)[0] + 1
significant_pacf = np.where(np.abs(pacf_values[1:]) > ci)[0] + 1

print(f"\\n95% Confidence Interval: +/-{ci:.4f}")
print(f"\\nSignificant ACF lags: {significant_acf[:10].tolist()}{'...' if len(significant_acf) > 10 else ''}")
print(f"Significant PACF lags: {significant_pacf[:10].tolist()}{'...' if len(significant_pacf) > 10 else ''}")

# Suggest ARIMA orders
print("\\n=== ARIMA ORDER SUGGESTIONS ===")

# AR order (p): Based on PACF cutoff
if len(significant_pacf) == 0:
    suggested_p = 0
elif significant_pacf[0] == 1 and (len(significant_pacf) == 1 or significant_pacf[1] > 2):
    suggested_p = 1
else:
    suggested_p = min(3, max(significant_pacf[:3]) if len(significant_pacf) > 0 else 1)

# MA order (q): Based on ACF cutoff
if len(significant_acf) == 0:
    suggested_q = 0
elif significant_acf[0] == 1 and (len(significant_acf) == 1 or significant_acf[1] > 2):
    suggested_q = 1
else:
    suggested_q = min(3, max(significant_acf[:3]) if len(significant_acf) > 0 else 1)

print(f"Suggested AR order (p): {suggested_p}")
print(f"Suggested differencing (d): {differencing_order}")
print(f"Suggested MA order (q): {suggested_q}")

# Seasonal orders
seasonal_period = ${params.seasonalPeriod}
if seasonal_period > 1 and len(series) >= 2 * seasonal_period:
    seasonal_acf = acf_values[seasonal_period] if len(acf_values) > seasonal_period else 0
    seasonal_pacf = pacf_values[seasonal_period] if len(pacf_values) > seasonal_period else 0
    
    suggested_P = 1 if abs(seasonal_pacf) > ci else 0
    suggested_D = 1 if abs(seasonal_acf) > 0.5 else 0
    suggested_Q = 1 if abs(seasonal_acf) > ci else 0
    
    print(f"\\nSuggested Seasonal AR (P): {suggested_P}")
    print(f"Suggested Seasonal Differencing (D): {suggested_D}")
    print(f"Suggested Seasonal MA (Q): {suggested_Q}")
    print(f"Seasonal Period (s): {seasonal_period}")

`;
}

function generateTrainTestSplitCode(params: ForecastingParameters): string {
  return `
# --- TRAIN-TEST SPLIT ---
print("\\n" + "-" * 70)
print("TRAIN-TEST SPLIT FOR BACKTESTING")
print("-" * 70)

# Split data for model validation
train_size = int(len(series) * ${params.trainTestSplit})
train = series[:train_size]
test = series[train_size:]

print(f"\\nTraining set: {len(train)} observations ({len(train)/len(series)*100:.1f}%)")
print(f"  Date range: {train.index.min()} to {train.index.max()}")
print(f"\\nTest set: {len(test)} observations ({len(test)/len(series)*100:.1f}%)")
print(f"  Date range: {test.index.min()} to {test.index.max()}")

# Forecast horizon
forecast_horizon = ${params.forecastHorizon}
print(f"\\nForecast horizon: {forecast_horizon} periods")

`;
}

function generateModelComparisonCode(params: ForecastingParameters): string {
  return `
# --- MODEL COMPARISON ---
print("\\n" + "-" * 70)
print("MODEL COMPARISON")
print("-" * 70)

def evaluate_forecast(actual, predicted, model_name):
    """Calculate forecast accuracy metrics"""
    actual_vals = actual.values if hasattr(actual, 'values') else np.array(actual)
    pred_vals = predicted.values if hasattr(predicted, 'values') else np.array(predicted)
    
    # Handle any NaN
    mask = ~(np.isnan(actual_vals) | np.isnan(pred_vals))
    actual_vals = actual_vals[mask]
    pred_vals = pred_vals[mask]
    
    if len(actual_vals) == 0:
        return None
    
    mae = mean_absolute_error(actual_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(actual_vals, pred_vals))
    mape = mean_absolute_percentage_error(actual_vals, pred_vals) * 100
    
    # Mean Absolute Scaled Error (MASE)
    naive_mae = np.mean(np.abs(np.diff(actual_vals)))
    mase = mae / naive_mae if naive_mae > 0 else np.inf
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MASE': mase
    }

# Store results
comparison_results = []
fitted_models = {}
forecasts = {}

seasonal_period = ${params.seasonalPeriod}

# 1. SARIMA
print("\\n1. Fitting SARIMA...")
try:
    if PMDARIMA_AVAILABLE:
        auto_model = auto_arima(
            train,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            d=None,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            D=None,
            m=seasonal_period,
            seasonal=True if seasonal_period > 1 else False,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            random_state=42
        )
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        print(f"   Best ARIMA order: {order}")
        print(f"   Best Seasonal order: {seasonal_order}")
    else:
        order = (suggested_p, differencing_order, suggested_q)
        seasonal_order = (suggested_P, suggested_D, suggested_Q, seasonal_period) if seasonal_period > 1 else (0, 0, 0, 0)
        print(f"   Using suggested orders: {order} x {seasonal_order}")
    
    sarima = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = sarima.fit(disp=False)
    sarima_pred = sarima_fit.get_forecast(steps=len(test))
    sarima_forecast = sarima_pred.predicted_mean
    sarima_forecast.index = test.index
    
    result = evaluate_forecast(test, sarima_forecast, 'SARIMA')
    if result:
        comparison_results.append(result)
        fitted_models['SARIMA'] = sarima_fit
        forecasts['SARIMA'] = sarima_forecast
        print(f"   SARIMA RMSE: {result['RMSE']:.4f}")
except Exception as e:
    print(f"   SARIMA failed: {e}")

# 2. Holt-Winters
print("\\n2. Fitting Holt-Winters...")
try:
    hw = ExponentialSmoothing(
        train,
        trend='add',
        seasonal=decomp_type,
        seasonal_periods=seasonal_period,
        damped_trend=True
    )
    hw_fit = hw.fit(optimized=True)
    hw_forecast = hw_fit.forecast(len(test))
    hw_forecast.index = test.index
    
    result = evaluate_forecast(test, hw_forecast, 'Holt-Winters')
    if result:
        comparison_results.append(result)
        fitted_models['Holt-Winters'] = hw_fit
        forecasts['Holt-Winters'] = hw_forecast
        print(f"   Holt-Winters RMSE: {result['RMSE']:.4f}")
except Exception as e:
    print(f"   Holt-Winters failed: {e}")

# 3. Simple Exponential Smoothing (baseline)
print("\\n3. Fitting Simple Exponential Smoothing...")
try:
    ses = SimpleExpSmoothing(train)
    ses_fit = ses.fit(optimized=True)
    ses_forecast = ses_fit.forecast(len(test))
    ses_forecast.index = test.index
    
    result = evaluate_forecast(test, ses_forecast, 'Simple Exp Smoothing')
    if result:
        comparison_results.append(result)
        fitted_models['Simple Exp Smoothing'] = ses_fit
        forecasts['Simple Exp Smoothing'] = ses_forecast
        print(f"   SES RMSE: {result['RMSE']:.4f}")
except Exception as e:
    print(f"   SES failed: {e}")

# 4. Prophet (if available)
if PROPHET_AVAILABLE:
    print("\\n4. Fitting Prophet...")
    try:
        prophet_train = train.reset_index()
        prophet_train.columns = ['ds', 'y']
        
        prophet = Prophet(
            yearly_seasonality='auto',
            weekly_seasonality='auto' if '${params.frequency}' in ['D', 'H'] else False,
            daily_seasonality=False,
            seasonality_mode=decomp_type,
            interval_width=${params.confidenceLevel}
        )
        prophet.fit(prophet_train)
        
        future = prophet.make_future_dataframe(periods=len(test), freq='${params.frequency}')
        prophet_pred = prophet.predict(future)
        prophet_forecast = prophet_pred.iloc[-len(test):]['yhat']
        prophet_forecast.index = test.index
        
        result = evaluate_forecast(test, prophet_forecast, 'Prophet')
        if result:
            comparison_results.append(result)
            fitted_models['Prophet'] = prophet
            forecasts['Prophet'] = prophet_forecast
            print(f"   Prophet RMSE: {result['RMSE']:.4f}")
    except Exception as e:
        print(f"   Prophet failed: {e}")
else:
    print("\\n4. Prophet not available (skipped)")

# 5. Naive Seasonal (baseline)
print("\\n5. Computing Naive Seasonal Forecast (baseline)...")
if seasonal_period > 0 and len(series) > seasonal_period:
    naive_forecast = series.shift(seasonal_period).loc[test.index]
    if not naive_forecast.isnull().all():
        result = evaluate_forecast(test, naive_forecast.dropna(), 'Naive Seasonal')
        if result:
            comparison_results.append(result)
            forecasts['Naive Seasonal'] = naive_forecast
            print(f"   Naive Seasonal RMSE: {result['RMSE']:.4f}")
else:
    naive_forecast = pd.Series([train.iloc[-1]] * len(test), index=test.index)
    result = evaluate_forecast(test, naive_forecast, 'Naive (Last Value)')
    if result:
        comparison_results.append(result)
        forecasts['Naive'] = naive_forecast
        print(f"   Naive RMSE: {result['RMSE']:.4f}")

# Summary
comparison_df = pd.DataFrame(comparison_results)
comparison_df = comparison_df.sort_values('RMSE')

print("\\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)
print(comparison_df.round(4).to_string(index=False))

# Select best model
best_model_name = comparison_df.iloc[0]['Model']
print(f"\\nBEST MODEL: {best_model_name}")
print(f"  RMSE: {comparison_df.iloc[0]['RMSE']:.4f}")
print(f"  MAPE: {comparison_df.iloc[0]['MAPE']:.2f}%")
print(f"  MASE: {comparison_df.iloc[0]['MASE']:.4f}")

# Use best model for final forecast
best_fit = fitted_models.get(best_model_name)
model_name = best_model_name

`;
}

function generateSpecificModelCode(params: ForecastingParameters): string {
  const modelType = params.modelType.toLowerCase();
  
  if (modelType === 'prophet') {
    return `
# --- PROPHET MODEL ---
print("\\n" + "-" * 70)
print("PROPHET MODEL")
print("-" * 70)

if not PROPHET_AVAILABLE:
    raise ImportError("Prophet is not installed. Install with: pip install prophet")

# Prepare data for Prophet
prophet_train = train.reset_index()
prophet_train.columns = ['ds', 'y']

# Initialize Prophet
print("\\nFitting Prophet model...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality='${params.frequency}' in ['D', 'H'],
    daily_seasonality='${params.frequency}' == 'H',
    seasonality_mode='${params.decompositionType}',
    changepoint_prior_scale=0.05,
    interval_width=${params.confidenceLevel}
)

${params.holidayCountry ? `
# Add country holidays
if HOLIDAYS_AVAILABLE:
    model.add_country_holidays(country_name='${params.holidayCountry}')
    print(f"Added holidays for: ${params.holidayCountry}")
` : ''}

model.fit(prophet_train)

# Make predictions on test set
future_test = model.make_future_dataframe(periods=len(test), freq='${params.frequency}')
predictions = model.predict(future_test)
test_forecast = predictions.iloc[-len(test):]['yhat']
test_forecast.index = test.index

# Evaluate
result = evaluate_forecast(test, test_forecast, 'Prophet')
print(f"\\n=== PROPHET RESULTS ===")
print(f"Test RMSE: {result['RMSE']:.4f}")
print(f"Test MAPE: {result['MAPE']:.2f}%")

best_model_name = 'Prophet'
best_fit = model
forecasts = {'Prophet': test_forecast}
comparison_df = pd.DataFrame([result])

`;
  } else if (modelType === 'holtwinters' || modelType === 'ets') {
    return `
# --- HOLT-WINTERS EXPONENTIAL SMOOTHING ---
print("\\n" + "-" * 70)
print("HOLT-WINTERS EXPONENTIAL SMOOTHING")
print("-" * 70)

seasonal_period = ${params.seasonalPeriod}

# Fit Holt-Winters model
print("\\nFitting Holt-Winters model...")
model = ExponentialSmoothing(
    train,
    trend='add',
    seasonal='${params.decompositionType}',
    seasonal_periods=seasonal_period,
    damped_trend=True
)
model_fit = model.fit(optimized=True)

print("\\n=== MODEL PARAMETERS ===")
if hasattr(model_fit, 'params'):
    for param, value in model_fit.params.items():
        if value is not None and not np.isnan(value):
            print(f"  {param}: {value:.4f}")

# Make predictions on test set
test_forecast = model_fit.forecast(len(test))
test_forecast.index = test.index

# Evaluate
result = evaluate_forecast(test, test_forecast, 'Holt-Winters')
print(f"\\n=== HOLT-WINTERS RESULTS ===")
print(f"Test RMSE: {result['RMSE']:.4f}")
print(f"Test MAPE: {result['MAPE']:.2f}%")
print(f"AIC: {model_fit.aic:.2f}")
print(f"BIC: {model_fit.bic:.2f}")

best_model_name = 'Holt-Winters'
best_fit = model_fit
forecasts = {'Holt-Winters': test_forecast}
comparison_df = pd.DataFrame([result])

`;
  } else {
    return `
# --- SARIMA MODEL ---
print("\\n" + "-" * 70)
print("SARIMA MODEL FITTING")
print("-" * 70)

seasonal_period = ${params.seasonalPeriod}

# Determine ARIMA orders
if PMDARIMA_AVAILABLE:
    print("\\nRunning auto_arima for optimal order selection...")
    auto_model = auto_arima(
        train,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        d=None,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        D=None,
        m=seasonal_period,
        seasonal=True if seasonal_period > 1 else False,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        random_state=42
    )
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order
    print(f"\\nBest ARIMA order: {order}")
    if seasonal_period > 1:
        print(f"Best Seasonal order: {seasonal_order}")
else:
    order = (suggested_p, differencing_order, suggested_q)
    seasonal_order = (suggested_P, suggested_D, suggested_Q, seasonal_period) if seasonal_period > 1 else (0, 0, 0, 0)
    print(f"\\nUsing suggested orders: {order}")
    if seasonal_period > 1:
        print(f"Seasonal order: {seasonal_order}")

# Fit SARIMAX model
print("\\nFitting SARIMAX model...")
model = SARIMAX(
    train,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_fit = model.fit(disp=False)

print("\\n=== MODEL SUMMARY ===")
print(model_fit.summary().tables[0])
print(model_fit.summary().tables[1])

# Model diagnostics
print("\\n=== MODEL DIAGNOSTICS ===")
print(f"AIC: {model_fit.aic:.2f}")
print(f"BIC: {model_fit.bic:.2f}")
print(f"Log-Likelihood: {model_fit.llf:.2f}")

# Ljung-Box test
residuals = model_fit.resid
lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
print("\\nLjung-Box Test (residual autocorrelation):")
print(lb_test.round(4))

if all(lb_test['lb_pvalue'] > 0.05):
    print("Residuals show no significant autocorrelation (good model fit)")
else:
    print("Residuals show autocorrelation (consider different orders)")

# Make predictions on test set
test_pred = model_fit.get_forecast(steps=len(test))
test_forecast = test_pred.predicted_mean
test_forecast.index = test.index

# Evaluate
result = evaluate_forecast(test, test_forecast, 'SARIMA')
print(f"\\n=== SARIMA RESULTS ===")
print(f"Test RMSE: {result['RMSE']:.4f}")
print(f"Test MAPE: {result['MAPE']:.2f}%")

best_model_name = 'SARIMA'
best_fit = model_fit
forecasts = {'SARIMA': test_forecast}
comparison_df = pd.DataFrame([result])

`;
  }
}

function generateForecastCode(params: ForecastingParameters): string {
  return `
# --- GENERATE PRODUCTION FORECAST ---
print("\\n" + "-" * 70)
print("GENERATING PRODUCTION FORECAST")
print("-" * 70)

# Refit model on full data
print(f"\\nRefitting {best_model_name} on full dataset...")

confidence_level = ${params.confidenceLevel}
forecast_horizon = ${params.forecastHorizon}
seasonal_period = ${params.seasonalPeriod}

if best_model_name in ['SARIMA', 'ARIMA']:
    full_model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
    full_fit = full_model.fit(disp=False)
    
    forecast_result = full_fit.get_forecast(steps=forecast_horizon)
    forecast_values = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=1-confidence_level)
    forecast_lower = conf_int.iloc[:, 0]
    forecast_upper = conf_int.iloc[:, 1]

elif best_model_name in ['Holt-Winters', 'Simple Exp Smoothing']:
    if best_model_name == 'Holt-Winters':
        full_model = ExponentialSmoothing(series, trend='add', seasonal=decomp_type,
                                           seasonal_periods=seasonal_period, damped_trend=True)
    else:
        full_model = SimpleExpSmoothing(series)
    
    full_fit = full_model.fit(optimized=True)
    forecast_values = full_fit.forecast(forecast_horizon)
    
    # Simulate confidence intervals
    residual_std = full_fit.resid.std()
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    horizon_factor = np.sqrt(np.arange(1, forecast_horizon + 1))
    forecast_lower = forecast_values - z_score * residual_std * horizon_factor
    forecast_upper = forecast_values + z_score * residual_std * horizon_factor

elif best_model_name == 'Prophet' and PROPHET_AVAILABLE:
    full_prophet = Prophet(yearly_seasonality='auto', seasonality_mode=decomp_type,
                           interval_width=confidence_level)
    full_df = series.reset_index()
    full_df.columns = ['ds', 'y']
    full_prophet.fit(full_df)
    
    future = full_prophet.make_future_dataframe(periods=forecast_horizon, freq='${params.frequency}')
    prediction = full_prophet.predict(future)
    
    forecast_values = prediction.iloc[-forecast_horizon:]['yhat']
    forecast_lower = prediction.iloc[-forecast_horizon:]['yhat_lower']
    forecast_upper = prediction.iloc[-forecast_horizon:]['yhat_upper']
    forecast_values.index = future.iloc[-forecast_horizon:]['ds']
    forecast_lower.index = forecast_values.index
    forecast_upper.index = forecast_values.index
    
    full_fit = full_prophet

else:
    # Default: Simple forecast
    full_fit = SimpleExpSmoothing(series).fit(optimized=True)
    forecast_values = full_fit.forecast(forecast_horizon)
    residual_std = full_fit.resid.std()
    z_score = 1.96
    horizon_factor = np.sqrt(np.arange(1, forecast_horizon + 1))
    forecast_lower = forecast_values - z_score * residual_std * horizon_factor
    forecast_upper = forecast_values + z_score * residual_std * horizon_factor

# Create forecast DataFrame
forecast_df = pd.DataFrame({
    'Date': forecast_values.index,
    'Forecast': forecast_values.values,
    'Lower_CI': forecast_lower.values,
    'Upper_CI': forecast_upper.values
})
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
forecast_df = forecast_df.set_index('Date')

print(f"\\n=== FORECAST RESULTS ({forecast_horizon} periods) ===")
print(f"Confidence Level: {confidence_level*100:.0f}%")
print(f"\\nForecast Summary:")
print(forecast_df.round(2))

print(f"\\n=== FORECAST STATISTICS ===")
print(f"Mean forecast value: {forecast_df['Forecast'].mean():.2f}")
print(f"Min forecast value: {forecast_df['Forecast'].min():.2f}")
print(f"Max forecast value: {forecast_df['Forecast'].max():.2f}")
print(f"Average CI width: {(forecast_df['Upper_CI'] - forecast_df['Lower_CI']).mean():.2f}")

`;
}

function generateVisualizationsCode(params: ForecastingParameters): string {
  return `
# --- VISUALIZATIONS ---
print("\\n" + "-" * 70)
print("GENERATING VISUALIZATIONS")
print("-" * 70)

fig = plt.figure(figsize=(16, 14))

# Plot 1: Full Time Series with Forecast
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(series.index, series.values, 'b-', linewidth=1.5, label='Historical Data')
ax1.plot(forecast_df.index, forecast_df['Forecast'], 'r-', linewidth=2, label='Forecast')
ax1.fill_between(forecast_df.index, forecast_df['Lower_CI'], forecast_df['Upper_CI'],
                  color='red', alpha=0.2, label=f'{${params.confidenceLevel}*100:.0f}% CI')
ax1.axvline(x=series.index[-1], color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('${params.valueColumn}', fontsize=11)
ax1.set_title(f'Time Series Forecast - {best_model_name}', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Zoomed Forecast
ax2 = fig.add_subplot(3, 2, 2)
zoom_periods = min(len(series), forecast_horizon * 3)
recent_data = series.iloc[-zoom_periods:]
ax2.plot(recent_data.index, recent_data.values, 'b-o', linewidth=1.5, markersize=4, label='Historical')
ax2.plot(forecast_df.index, forecast_df['Forecast'], 'r-o', linewidth=2, markersize=6, label='Forecast')
ax2.fill_between(forecast_df.index, forecast_df['Lower_CI'], forecast_df['Upper_CI'],
                  color='red', alpha=0.2, label=f'{${params.confidenceLevel}*100:.0f}% CI')
ax2.axvline(x=series.index[-1], color='gray', linestyle='--', alpha=0.7)
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('${params.valueColumn}', fontsize=11)
ax2.set_title('Zoomed View: Recent Data + Forecast', fontsize=12)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Plot 3: Trend Component (if decomposition available)
ax3 = fig.add_subplot(3, 2, 3)
if decomposition_available:
    ax3.plot(stl_result.trend.index, stl_result.trend.values, 'b-', linewidth=1.5)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Trend Component', fontsize=11)
    ax3.set_title('Extracted Trend Component (STL)', fontsize=12)
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'Decomposition not available\\n(insufficient data)', 
             ha='center', va='center', fontsize=12)
    ax3.set_title('Trend Component', fontsize=12)

# Plot 4: Seasonal Component
ax4 = fig.add_subplot(3, 2, 4)
if decomposition_available:
    seasonal_period = ${params.seasonalPeriod}
    seasonal_data = stl_result.seasonal[:seasonal_period*2] if len(stl_result.seasonal) > seasonal_period*2 else stl_result.seasonal
    ax4.plot(range(len(seasonal_data)), seasonal_data.values, 'g-', linewidth=1.5)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Period', fontsize=11)
    ax4.set_ylabel('Seasonal Component', fontsize=11)
    ax4.set_title(f'Seasonal Pattern (Period = {seasonal_period})', fontsize=12)
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Seasonal analysis not available', ha='center', va='center', fontsize=12)
    ax4.set_title('Seasonal Component', fontsize=12)

# Plot 5: ACF Plot
ax5 = fig.add_subplot(3, 2, 5)
plot_acf(test_series.dropna(), lags=n_lags, ax=ax5, alpha=0.05)
ax5.set_xlabel('Lag', fontsize=11)
ax5.set_ylabel('Autocorrelation', fontsize=11)
ax5.set_title('Autocorrelation Function (ACF)', fontsize=12)

# Plot 6: Residuals Distribution
ax6 = fig.add_subplot(3, 2, 6)
if hasattr(full_fit, 'resid'):
    residuals = full_fit.resid.dropna()
elif best_model_name == 'Prophet':
    fitted = full_fit.predict(full_df)['yhat'].values[:len(series)]
    residuals = series.values - fitted
else:
    residuals = series.diff().dropna().values

ax6.hist(residuals, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='k')
x_range = np.linspace(np.nanmin(residuals), np.nanmax(residuals), 100)
ax6.plot(x_range, stats.norm.pdf(x_range, np.nanmean(residuals), np.nanstd(residuals)),
         'r-', linewidth=2, label='Normal Distribution')
ax6.axvline(x=0, color='black', linestyle='--', alpha=0.7)
ax6.set_xlabel('Residual Value', fontsize=11)
ax6.set_ylabel('Density', fontsize=11)
ax6.set_title('Residual Distribution', fontsize=12)
ax6.legend()

plt.tight_layout()
plt.savefig('forecast_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nVisualization saved as 'forecast_analysis.png'")

# Model Comparison Plot (if multiple models)
if len(comparison_df) > 1:
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metric comparison bar chart
    ax_bar = axes2[0]
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars1 = ax_bar.bar(x - width/2, comparison_df['RMSE'], width, label='RMSE', color='steelblue')
    ax_bar_twin = ax_bar.twinx()
    bars2 = ax_bar_twin.bar(x + width/2, comparison_df['MAPE'], width, label='MAPE (%)', color='coral')
    
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax_bar.set_ylabel('RMSE', fontsize=11, color='steelblue')
    ax_bar_twin.set_ylabel('MAPE (%)', fontsize=11, color='coral')
    ax_bar.set_title('Model Comparison: Error Metrics', fontsize=12)
    
    lines1, labels1 = ax_bar.get_legend_handles_labels()
    lines2, labels2 = ax_bar_twin.get_legend_handles_labels()
    ax_bar.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Forecast comparison on test set
    ax_fc = axes2[1]
    ax_fc.plot(test.index, test.values, 'k-', linewidth=2, label='Actual')
    colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts)))
    for (name, fc), color in zip(forecasts.items(), colors):
        if 'Naive' not in name:
            ax_fc.plot(test.index[:len(fc)], fc.values[:len(test)], 
                      linestyle='--', linewidth=1.5, color=color, label=name)
    ax_fc.set_xlabel('Date', fontsize=11)
    ax_fc.set_ylabel('${params.valueColumn}', fontsize=11)
    ax_fc.set_title('Model Comparison: Test Set Predictions', fontsize=12)
    ax_fc.legend(loc='upper left')
    ax_fc.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\\nModel comparison saved as 'model_comparison.png'")

`;
}

function generateForecastFunctionCode(params: ForecastingParameters): string {
  return `
# --- FORECAST FUNCTION ---

def generate_forecast(historical_data, periods=${params.forecastHorizon}, model_type='${params.modelType}'):
    """
    Generate forecast for new data.
    
    Parameters:
    -----------
    historical_data : pd.Series
        Time series data with datetime index
    periods : int
        Number of periods to forecast
    model_type : str
        'sarima', 'holtwinters', 'prophet', or 'auto'
    
    Returns:
    --------
    dict with:
        - forecast: pd.Series of point forecasts
        - lower_ci: pd.Series of lower confidence bound
        - upper_ci: pd.Series of upper confidence bound
        - model_info: dict with model details
    """
    
    # Ensure datetime index and frequency
    if not isinstance(historical_data.index, pd.DatetimeIndex):
        historical_data.index = pd.to_datetime(historical_data.index)
    
    if historical_data.index.freq is None:
        freq = pd.infer_freq(historical_data.index)
        if freq:
            historical_data = historical_data.asfreq(freq)
        else:
            historical_data = historical_data.asfreq('${params.frequency}')
    
    confidence_level = ${params.confidenceLevel}
    seasonal_period = ${params.seasonalPeriod}
    
    if model_type.lower() in ['sarima', 'arima']:
        if PMDARIMA_AVAILABLE:
            auto_fit = auto_arima(historical_data, seasonal=True, m=seasonal_period,
                                   suppress_warnings=True, error_action='ignore')
            fit_order = auto_fit.order
            fit_seasonal = auto_fit.seasonal_order
        else:
            fit_order = (1, 1, 1)
            fit_seasonal = (1, 1, 1, seasonal_period)
        
        model = SARIMAX(historical_data, order=fit_order, seasonal_order=fit_seasonal,
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        result = fit.get_forecast(steps=periods)
        forecast = result.predicted_mean
        ci = result.conf_int(alpha=1-confidence_level)
        lower = ci.iloc[:, 0]
        upper = ci.iloc[:, 1]
        
    elif model_type.lower() in ['holtwinters', 'hw', 'ets']:
        model = ExponentialSmoothing(historical_data, trend='add', seasonal='${params.decompositionType}',
                                     seasonal_periods=seasonal_period, damped_trend=True)
        fit = model.fit(optimized=True)
        forecast = fit.forecast(periods)
        
        std = fit.resid.std()
        z = stats.norm.ppf((1 + confidence_level) / 2)
        factor = np.sqrt(np.arange(1, periods + 1))
        lower = forecast - z * std * factor
        upper = forecast + z * std * factor
        
    elif model_type.lower() == 'prophet' and PROPHET_AVAILABLE:
        prophet = Prophet(seasonality_mode='${params.decompositionType}', interval_width=confidence_level)
        df = historical_data.reset_index()
        df.columns = ['ds', 'y']
        prophet.fit(df)
        future = prophet.make_future_dataframe(periods=periods, freq=historical_data.index.freq or '${params.frequency}')
        pred = prophet.predict(future)
        forecast = pred.iloc[-periods:]['yhat']
        lower = pred.iloc[-periods:]['yhat_lower']
        upper = pred.iloc[-periods:]['yhat_upper']
        forecast.index = future.iloc[-periods:]['ds']
        lower.index = forecast.index
        upper.index = forecast.index
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return {
        'forecast': forecast,
        'lower_ci': lower,
        'upper_ci': upper,
        'model_info': {
            'model_type': model_type,
            'periods': periods,
            'confidence_level': confidence_level
        }
    }

print("\\n--- FORECAST FUNCTION READY ---")
print(\"\"\"
Example usage:
result = generate_forecast(
    historical_data=series,
    periods=12,
    model_type='sarima'
)

print("Forecast:", result['forecast'])
print("Lower CI:", result['lower_ci'])
print("Upper CI:", result['upper_ci'])
\"\"\")

`;
}

function generateModelPersistenceCode(params: ForecastingParameters): string {
  return `
# --- SAVE MODEL ARTIFACTS ---
print("\\n" + "-" * 70)
print("SAVING MODEL ARTIFACTS")
print("-" * 70)

import joblib
import json

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save model
if best_model_name in ['SARIMA', 'ARIMA']:
    model_filename = f'forecast_sarima_{timestamp}.pkl'
elif best_model_name == 'Prophet':
    model_filename = f'forecast_prophet_{timestamp}.pkl'
else:
    model_filename = f'forecast_{best_model_name.lower().replace(" ", "_")}_{timestamp}.pkl'

joblib.dump(full_fit, model_filename)
print(f"Model saved: {model_filename}")

# Save forecast results
forecast_filename = f'forecast_results_{timestamp}.csv'
forecast_df.to_csv(forecast_filename)
print(f"Forecast saved: {forecast_filename}")

# Save configuration
config = {
    'model_type': best_model_name,
    'order': list(order) if best_model_name in ['SARIMA', 'ARIMA'] else None,
    'seasonal_order': list(seasonal_order) if best_model_name in ['SARIMA', 'ARIMA'] else None,
    'seasonal_period': ${params.seasonalPeriod},
    'frequency': '${params.frequency}',
    'decomposition_type': decomp_type,
    'confidence_level': ${params.confidenceLevel},
    'forecast_horizon': ${params.forecastHorizon},
    'value_column': '${params.valueColumn}',
    'timestamp': timestamp,
    'metrics': {
        'test_rmse': float(comparison_df.iloc[0]['RMSE']) if len(comparison_df) > 0 else None,
        'test_mape': float(comparison_df.iloc[0]['MAPE']) if len(comparison_df) > 0 else None
    }
}

config_filename = f'forecast_config_{timestamp}.json'
with open(config_filename, 'w') as f:
    json.dump(config, f, indent=2, default=str)
print(f"Configuration saved: {config_filename}")

# Save historical data
data_filename = f'historical_data_{timestamp}.csv'
series.to_csv(data_filename)
print(f"Historical data saved: {data_filename}")

print("\\n" + "=" * 70)
print("FORECASTING ANALYSIS COMPLETE")
print("=" * 70)
print(f"\\nBest Model: {best_model_name}")
if len(comparison_df) > 0:
    print(f"Test RMSE: {comparison_df.iloc[0]['RMSE']:.4f}")
    print(f"Test MAPE: {comparison_df.iloc[0]['MAPE']:.2f}%")
print(f"\\nForecast generated for next {${params.forecastHorizon}} periods")
print(f"Confidence level: {${params.confidenceLevel}*100:.0f}%")

print("\\nFiles generated:")
print(f"  - {model_filename}")
print(f"  - {forecast_filename}")
print(f"  - {config_filename}")
print(f"  - {data_filename}")
print(f"  - forecast_analysis.png")
if len(comparison_df) > 1:
    print(f"  - model_comparison.png")

# --- LOADING INSTRUCTIONS ---
print(\"\"\"
To load and use this model later:

import joblib
import json
import pandas as pd

# Load artifacts
model = joblib.load('[model_filename]')
with open('[config_filename]', 'r') as f:
    config = json.load(f)

# Load new data and generate forecast
# result = generate_forecast(new_data, periods=12)
\"\"\")
`;
}
