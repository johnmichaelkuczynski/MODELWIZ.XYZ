import ExcelJS from 'exceljs';
import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';

export type FinanceLLMProvider = 'zhi1' | 'zhi2' | 'zhi3' | 'zhi4' | 'zhi5';

interface ThreeStatementAssumptions {
  companyName: string;
  industry: string;
  fiscalYearEnd: string;
  currency: string;
  projectionYears: number;

  // Historical Data
  historicalRevenue: number;
  historicalCOGS: number;
  historicalGrossMargin: number;
  historicalSGA: number;
  historicalRD: number;
  historicalEBITDA: number;
  historicalDA: number;
  historicalInterestExpense: number;
  historicalNetIncome: number;
  historicalTotalAssets: number;
  historicalTotalDebt: number;
  historicalCash: number;
  historicalEquity: number;
  historicalSharesOutstanding: number;
  historicalPPE: number;

  // Revenue Assumptions
  revenueGrowthRates: number[];

  // Cost Structure Assumptions
  baseGrossMargin: number;
  targetGrossMargin: number;
  baseSGAPercent: number;
  targetSGAPercent: number;
  rdPercent: number;
  daPercent: number;

  // Working Capital Assumptions
  dso: number;
  dio: number;
  dpo: number;
  prepaidPercent: number;
  accruedPercent: number;
  otherCAPercent: number;
  otherCLPercent: number;

  // CapEx Assumptions
  capexPercent: number[];

  // Debt Assumptions
  existingDebtBalance: number;
  existingDebtRate: number;
  debtAmortization: number;
  revolverSize: number;
  revolverRate: number;
  revolverCommitmentFee: number;
  minimumCashBalance: number;
  interestOnCash: number;

  // Tax Assumptions
  effectiveTaxRate: number;
  nolCarryforward: number;

  // Equity Assumptions
  stockBasedCompPercent: number;
  dividendsPerShare: number;
  payoutRatio: number;
  shareRepurchases: number;
}

interface IncomeStatementData {
  years: string[];
  revenue: number[];
  revenueGrowth: number[];
  cogs: number[];
  grossProfit: number[];
  grossMargin: number[];
  sga: number[];
  sgaPercent: number[];
  rd: number[];
  rdPercent: number[];
  totalOpex: number[];
  ebitda: number[];
  ebitdaMargin: number[];
  da: number[];
  ebit: number[];
  ebitMargin: number[];
  interestExpense: number[];
  interestIncome: number[];
  netInterest: number[];
  otherIncomeExpense: number[];
  ebt: number[];
  incomeTax: number[];
  effectiveTaxRate: number[];
  netIncome: number[];
  netMargin: number[];
  sharesOutstanding: number[];
  eps: number[];
}

interface BalanceSheetData {
  years: string[];
  cash: number[];
  accountsReceivable: number[];
  inventory: number[];
  prepaidExpenses: number[];
  otherCurrentAssets: number[];
  totalCurrentAssets: number[];
  ppeGross: number[];
  accumulatedDepreciation: number[];
  ppeNet: number[];
  intangibleAssets: number[];
  goodwill: number[];
  otherLongTermAssets: number[];
  totalNonCurrentAssets: number[];
  totalAssets: number[];
  accountsPayable: number[];
  accruedExpenses: number[];
  deferredRevenue: number[];
  currentPortionDebt: number[];
  revolverBalance: number[];
  otherCurrentLiabilities: number[];
  totalCurrentLiabilities: number[];
  longTermDebt: number[];
  deferredTaxLiabilities: number[];
  otherLongTermLiabilities: number[];
  totalNonCurrentLiabilities: number[];
  totalLiabilities: number[];
  commonStock: number[];
  apic: number[];
  retainedEarnings: number[];
  treasuryStock: number[];
  aoci: number[];
  totalEquity: number[];
  totalLiabilitiesEquity: number[];
  balanceCheck: number[];
}

interface CashFlowData {
  years: string[];
  netIncome: number[];
  depreciation: number[];
  stockBasedComp: number[];
  deferredTaxes: number[];
  otherNonCash: number[];
  totalNonCashAdjustments: number[];
  changeInAR: number[];
  changeInInventory: number[];
  changeInPrepaid: number[];
  changeInOtherCA: number[];
  changeInAP: number[];
  changeInAccrued: number[];
  changeInDeferredRev: number[];
  changeInOtherCL: number[];
  totalWorkingCapitalChange: number[];
  cfo: number[];
  capex: number[];
  acquisitions: number[];
  assetSales: number[];
  otherInvesting: number[];
  cfi: number[];
  debtProceeds: number[];
  debtRepayments: number[];
  revolverChange: number[];
  equityProceeds: number[];
  shareRepurchases: number[];
  dividendsPaid: number[];
  otherFinancing: number[];
  cff: number[];
  netCashChange: number[];
  beginningCash: number[];
  endingCash: number[];
  freeCashFlow: number[];
}

interface DebtScheduleData {
  years: string[];
  termDebtBeginning: number[];
  termDebtAmortization: number[];
  termDebtOptionalPrepay: number[];
  termDebtEnding: number[];
  termDebtAverage: number[];
  termDebtRate: number[];
  termDebtInterest: number[];
  revolverBeginning: number[];
  revolverDraws: number[];
  revolverPaydowns: number[];
  revolverEnding: number[];
  revolverAvailable: number[];
  revolverAverage: number[];
  revolverRate: number[];
  revolverInterest: number[];
  revolverCommitmentFee: number[];
  totalDebtBeginning: number[];
  totalDebtEnding: number[];
  totalInterestExpense: number[];
  netDebt: number[];
  debtToEBITDA: number[];
  netDebtToEBITDA: number[];
  interestCoverage: number[];
}

interface WorkingCapitalSchedule {
  years: string[];
  arBalance: number[];
  arChange: number[];
  inventoryBalance: number[];
  inventoryChange: number[];
  prepaidBalance: number[];
  prepaidChange: number[];
  otherCABalance: number[];
  otherCAChange: number[];
  apBalance: number[];
  apChange: number[];
  accruedBalance: number[];
  accruedChange: number[];
  otherCLBalance: number[];
  otherCLChange: number[];
  nwc: number[];
  nwcPercent: number[];
  nwcChange: number[];
  cashConversionCycle: number[];
}

interface PPESchedule {
  years: string[];
  beginningPPE: number[];
  capex: number[];
  disposals: number[];
  depreciation: number[];
  endingPPE: number[];
  capexPercent: number[];
  capexToDA: number[];
}

interface EquitySchedule {
  years: string[];
  commonStockBeginning: number[];
  commonStockEnding: number[];
  apicBeginning: number[];
  stockBasedComp: number[];
  apicEnding: number[];
  retainedEarningsBeginning: number[];
  netIncome: number[];
  dividends: number[];
  retainedEarningsEnding: number[];
  treasuryStockBeginning: number[];
  shareRepurchases: number[];
  treasuryStockEnding: number[];
  totalEquity: number[];
  sharesBeginning: number[];
  sharesIssued: number[];
  sharesRepurchased: number[];
  sharesEnding: number[];
  dividendsPerShare: number[];
  payoutRatio: number[];
}

interface RatioAnalysis {
  years: string[];
  grossMargin: number[];
  ebitdaMargin: number[];
  ebitMargin: number[];
  netMargin: number[];
  roe: number[];
  roa: number[];
  roic: number[];
  currentRatio: number[];
  quickRatio: number[];
  cashRatio: number[];
  debtToEquity: number[];
  debtToEBITDA: number[];
  netDebtToEBITDA: number[];
  interestCoverage: number[];
  assetTurnover: number[];
  inventoryTurnover: number[];
  receivablesTurnover: number[];
  payablesTurnover: number[];
  cashConversionCycle: number[];
  revenueGrowth: number[];
  ebitdaGrowth: number[];
  netIncomeGrowth: number[];
  epsGrowth: number[];
  basicEPS: number[];
  bookValuePerShare: number[];
}

interface ThreeStatementModelResult {
  assumptions: ThreeStatementAssumptions;
  incomeStatement: IncomeStatementData;
  balanceSheet: BalanceSheetData;
  cashFlow: CashFlowData;
  debtSchedule: DebtScheduleData;
  workingCapital: WorkingCapitalSchedule;
  ppeSchedule: PPESchedule;
  equitySchedule: EquitySchedule;
  ratioAnalysis: RatioAnalysis;
  summary: {
    revenueCAGR: number;
    ebitdaCAGR: number;
    netIncomeCAGR: number;
    epsCAGR: number;
    endingNetDebtToEBITDA: number;
    endingDebtToEquity: number;
    averageROIC: number;
    isBalanced: boolean;
    cashFlowReconciled: boolean;
  };
}

export async function parseThreeStatementDescription(
  description: string,
  customInstructions?: string,
  llmProvider: FinanceLLMProvider = 'zhi5'
): Promise<{ assumptions: ThreeStatementAssumptions; providerUsed: string }> {
  const providerNames: Record<FinanceLLMProvider, string> = {
    'zhi1': 'Zhi 1 (GPT-4o)',
    'zhi2': 'Zhi 2 (Claude)',
    'zhi3': 'Zhi 3 (DeepSeek)',
    'zhi4': 'Zhi 4 (Perplexity)',
    'zhi5': 'Zhi 5 (Grok)'
  };

  const systemPrompt = `You are a financial analyst expert at extracting 3-statement financial model assumptions from natural language descriptions.
Extract ALL the following variables from the user's description. If a value is not explicitly stated, use reasonable defaults based on industry standards.

Return a JSON object with EXACTLY these fields (all numbers, no strings except companyName, industry, fiscalYearEnd, currency):
{
  "companyName": "string - company name or 'Target Company' if not specified",
  "industry": "string - industry/sector or 'General' if not specified",
  "fiscalYearEnd": "string - e.g., 'December' or 'Q4 2024'",
  "currency": "string - e.g., 'USD'",
  "projectionYears": 5,
  
  "historicalRevenue": number in millions,
  "historicalCOGS": number in millions,
  "historicalGrossMargin": decimal (e.g., 0.35 for 35%),
  "historicalSGA": number in millions,
  "historicalRD": number in millions (0 if not R&D intensive),
  "historicalEBITDA": number in millions,
  "historicalDA": number in millions,
  "historicalInterestExpense": number in millions,
  "historicalNetIncome": number in millions,
  "historicalTotalAssets": number in millions,
  "historicalTotalDebt": number in millions,
  "historicalCash": number in millions,
  "historicalEquity": number in millions,
  "historicalSharesOutstanding": number in millions,
  "historicalPPE": number in millions (estimate at 50% of assets if not given),
  
  "revenueGrowthRates": array of 5 decimals for Y1-Y5 growth (e.g., [0.08, 0.10, 0.06, 0.06, 0.06]),
  
  "baseGrossMargin": decimal,
  "targetGrossMargin": decimal,
  "baseSGAPercent": decimal (SG&A as % of revenue),
  "targetSGAPercent": decimal,
  "rdPercent": decimal (R&D as % of revenue),
  "daPercent": decimal (D&A as % of revenue),
  
  "dso": number (days sales outstanding),
  "dio": number (days inventory outstanding),
  "dpo": number (days payable outstanding),
  "prepaidPercent": decimal (prepaid expenses as % of revenue),
  "accruedPercent": decimal (accrued liabilities as % of operating expenses),
  "otherCAPercent": decimal (other current assets as % of revenue),
  "otherCLPercent": decimal (other current liabilities as % of revenue),
  
  "capexPercent": array of 5 decimals for Y1-Y5 CapEx % of revenue,
  
  "existingDebtBalance": number in millions,
  "existingDebtRate": decimal (e.g., 0.06 for 6%),
  "debtAmortization": number in millions per year,
  "revolverSize": number in millions,
  "revolverRate": decimal,
  "revolverCommitmentFee": decimal (e.g., 0.005 for 0.5%),
  "minimumCashBalance": number in millions,
  "interestOnCash": decimal,
  
  "effectiveTaxRate": decimal,
  "nolCarryforward": number in millions (0 if none),
  
  "stockBasedCompPercent": decimal (stock-based comp as % of revenue),
  "dividendsPerShare": number,
  "payoutRatio": decimal (0 if not specified),
  "shareRepurchases": number in millions per year (0 if none)
}

Default values if not specified:
- revenueGrowthRates: [0.05, 0.05, 0.04, 0.04, 0.03]
- baseGrossMargin: 0.35
- targetGrossMargin: 0.38
- baseSGAPercent: 0.20
- targetSGAPercent: 0.18
- rdPercent: 0.03
- daPercent: 0.05
- dso: 45
- dio: 60
- dpo: 40
- prepaidPercent: 0.01
- accruedPercent: 0.15
- otherCAPercent: 0.01
- otherCLPercent: 0.01
- capexPercent: [0.05, 0.05, 0.05, 0.05, 0.05]
- existingDebtRate: 0.06
- debtAmortization: 0 (calculate from debt/5 if term given)
- revolverSize: 0
- revolverRate: 0.07
- revolverCommitmentFee: 0.005
- minimumCashBalance: 50
- interestOnCash: 0.02
- effectiveTaxRate: 0.25
- nolCarryforward: 0
- stockBasedCompPercent: 0.01
- dividendsPerShare: 0
- payoutRatio: 0
- shareRepurchases: 0

Parse percentages carefully:
- "eight percent" or "8%" = 0.08
- "thirty-five percent" = 0.35
- Convert all spoken numbers to their numeric equivalents

IMPORTANT: Return ONLY valid JSON, no markdown, no explanations.`;

  let userPrompt = `Extract 3-statement model assumptions from this description:\n\n${description}`;

  if (customInstructions) {
    userPrompt += `\n\nAdditional instructions: ${customInstructions}`;
  }

  let responseText: string;

  if (llmProvider === 'zhi1') {
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      max_tokens: 4000,
      temperature: 0,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ]
    });
    responseText = response.choices[0]?.message?.content || '';

  } else if (llmProvider === 'zhi2') {
    const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    const response = await anthropic.messages.create({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 4000,
      temperature: 0,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }]
    });
    const content = response.content[0];
    if (content.type !== 'text') {
      throw new Error('Unexpected response type from Anthropic');
    }
    responseText = content.text;

  } else if (llmProvider === 'zhi3') {
    const openaiCompatible = new OpenAI({
      baseURL: 'https://api.deepseek.com/v1',
      apiKey: process.env.DEEPSEEK_API_KEY
    });
    const response = await openaiCompatible.chat.completions.create({
      model: 'deepseek-chat',
      max_tokens: 4000,
      temperature: 0,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ]
    });
    responseText = response.choices[0]?.message?.content || '';

  } else if (llmProvider === 'zhi4') {
    const perplexity = new OpenAI({
      baseURL: 'https://api.perplexity.ai',
      apiKey: process.env.PERPLEXITY_API_KEY
    });
    const response = await perplexity.chat.completions.create({
      model: 'llama-3.1-sonar-large-128k-online',
      max_tokens: 4000,
      temperature: 0,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ]
    });
    responseText = response.choices[0]?.message?.content || '';

  } else if (llmProvider === 'zhi5') {
    const grok = new OpenAI({
      baseURL: 'https://api.x.ai/v1',
      apiKey: process.env.GROK_API_KEY
    });
    const response = await grok.chat.completions.create({
      model: 'grok-3',
      max_tokens: 4000,
      temperature: 0,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ]
    });
    responseText = response.choices[0]?.message?.content || '';

  } else {
    throw new Error(`Unknown LLM provider: ${llmProvider}`);
  }

  let cleanedText = responseText.trim();
  if (cleanedText.startsWith('```json')) {
    cleanedText = cleanedText.slice(7);
  } else if (cleanedText.startsWith('```')) {
    cleanedText = cleanedText.slice(3);
  }
  if (cleanedText.endsWith('```')) {
    cleanedText = cleanedText.slice(0, -3);
  }
  cleanedText = cleanedText.trim();

  const parsed = JSON.parse(cleanedText);
  return { 
    assumptions: parsed as ThreeStatementAssumptions, 
    providerUsed: providerNames[llmProvider] 
  };
}

export function calculateThreeStatementModel(
  assumptions: ThreeStatementAssumptions
): ThreeStatementModelResult {
  const years = assumptions.projectionYears;
  const yearLabels: string[] = [];
  for (let i = 0; i <= years; i++) {
    yearLabels.push(i === 0 ? 'Historical' : `Year ${i}`);
  }

  // ============ INCOME STATEMENT CALCULATIONS ============
  const revenue: number[] = [assumptions.historicalRevenue];
  const revenueGrowth: number[] = [0];
  const grossMargin: number[] = [assumptions.historicalGrossMargin];
  const sgaPercent: number[] = [assumptions.baseSGAPercent];
  
  // Calculate linear interpolation for margins
  const grossMarginStep = (assumptions.targetGrossMargin - assumptions.baseGrossMargin) / years;
  const sgaStep = (assumptions.targetSGAPercent - assumptions.baseSGAPercent) / years;

  for (let i = 1; i <= years; i++) {
    const growthRate = assumptions.revenueGrowthRates[i - 1] || 0.05;
    revenue.push(revenue[i - 1] * (1 + growthRate));
    revenueGrowth.push(growthRate);
    grossMargin.push(assumptions.baseGrossMargin + grossMarginStep * i);
    sgaPercent.push(assumptions.baseSGAPercent + sgaStep * i);
  }

  const cogs: number[] = revenue.map((r, i) => r * (1 - grossMargin[i]));
  const grossProfit: number[] = revenue.map((r, i) => r * grossMargin[i]);
  const sga: number[] = revenue.map((r, i) => r * sgaPercent[i]);
  const rd: number[] = revenue.map(r => r * assumptions.rdPercent);
  const totalOpex: number[] = sga.map((s, i) => s + rd[i]);
  const ebitda: number[] = grossProfit.map((gp, i) => gp - totalOpex[i]);
  const ebitdaMargin: number[] = ebitda.map((e, i) => e / revenue[i]);
  const da: number[] = revenue.map(r => r * assumptions.daPercent);
  const ebit: number[] = ebitda.map((e, i) => e - da[i]);
  const ebitMargin: number[] = ebit.map((e, i) => e / revenue[i]);

  // ============ DEBT SCHEDULE ============
  const termDebtBeginning: number[] = [assumptions.existingDebtBalance];
  const termDebtAmortization: number[] = [0];
  const termDebtEnding: number[] = [assumptions.existingDebtBalance];
  const revolverBeginning: number[] = [0];
  const revolverEnding: number[] = [0];
  const revolverDraws: number[] = [0];
  const revolverPaydowns: number[] = [0];

  for (let i = 1; i <= years; i++) {
    termDebtBeginning.push(termDebtEnding[i - 1]);
    const amort = Math.min(assumptions.debtAmortization, termDebtBeginning[i]);
    termDebtAmortization.push(amort);
    termDebtEnding.push(termDebtBeginning[i] - amort);
    revolverBeginning.push(revolverEnding[i - 1]);
    revolverEnding.push(0); // Will be solved in circular reference handling
    revolverDraws.push(0);
    revolverPaydowns.push(0);
  }

  const termDebtAverage: number[] = termDebtBeginning.map((b, i) => (b + termDebtEnding[i]) / 2);
  const termDebtInterest: number[] = termDebtAverage.map(avg => avg * assumptions.existingDebtRate);
  const revolverAverage: number[] = revolverBeginning.map((b, i) => (b + revolverEnding[i]) / 2);
  const revolverInterest: number[] = revolverAverage.map(avg => avg * assumptions.revolverRate);
  const revolverCommitmentFee: number[] = revolverEnding.map(e => 
    (assumptions.revolverSize - e) * assumptions.revolverCommitmentFee
  );
  const totalInterest: number[] = termDebtInterest.map((t, i) => t + revolverInterest[i] + revolverCommitmentFee[i]);

  // Interest income on cash (will be updated after cash is solved)
  const interestIncome: number[] = new Array(years + 1).fill(0);
  const netInterest: number[] = totalInterest.map((ti, i) => ti - interestIncome[i]);
  const otherIncomeExpense: number[] = new Array(years + 1).fill(0);
  const ebt: number[] = ebit.map((e, i) => e - netInterest[i] + otherIncomeExpense[i]);
  
  // Tax calculation with NOL handling
  let nolRemaining = assumptions.nolCarryforward;
  const incomeTax: number[] = ebt.map(e => {
    if (e <= 0) return 0;
    const taxableIncome = Math.max(0, e - nolRemaining);
    nolRemaining = Math.max(0, nolRemaining - e);
    return taxableIncome * assumptions.effectiveTaxRate;
  });
  const effectiveTaxRateArr: number[] = ebt.map((e, i) => e > 0 ? incomeTax[i] / e : 0);
  const netIncome: number[] = ebt.map((e, i) => e - incomeTax[i]);
  const netMargin: number[] = netIncome.map((ni, i) => ni / revenue[i]);

  // Shares and EPS
  const sharesOutstanding: number[] = new Array(years + 1).fill(assumptions.historicalSharesOutstanding);
  const eps: number[] = netIncome.map((ni, i) => ni / sharesOutstanding[i]);

  // ============ WORKING CAPITAL SCHEDULE ============
  const arBalance: number[] = [revenue[0] * (assumptions.dso / 365)];
  const inventoryBalance: number[] = [cogs[0] * (assumptions.dio / 365)];
  const prepaidBalance: number[] = [revenue[0] * assumptions.prepaidPercent];
  const otherCABalance: number[] = [revenue[0] * assumptions.otherCAPercent];
  const apBalance: number[] = [cogs[0] * (assumptions.dpo / 365)];
  const accruedBalance: number[] = [totalOpex[0] * assumptions.accruedPercent];
  const otherCLBalance: number[] = [revenue[0] * assumptions.otherCLPercent];

  for (let i = 1; i <= years; i++) {
    arBalance.push(revenue[i] * (assumptions.dso / 365));
    inventoryBalance.push(cogs[i] * (assumptions.dio / 365));
    prepaidBalance.push(revenue[i] * assumptions.prepaidPercent);
    otherCABalance.push(revenue[i] * assumptions.otherCAPercent);
    apBalance.push(cogs[i] * (assumptions.dpo / 365));
    accruedBalance.push(totalOpex[i] * assumptions.accruedPercent);
    otherCLBalance.push(revenue[i] * assumptions.otherCLPercent);
  }

  const arChange: number[] = arBalance.map((ar, i) => i === 0 ? 0 : ar - arBalance[i - 1]);
  const inventoryChange: number[] = inventoryBalance.map((inv, i) => i === 0 ? 0 : inv - inventoryBalance[i - 1]);
  const prepaidChange: number[] = prepaidBalance.map((p, i) => i === 0 ? 0 : p - prepaidBalance[i - 1]);
  const otherCAChange: number[] = otherCABalance.map((o, i) => i === 0 ? 0 : o - otherCABalance[i - 1]);
  const apChange: number[] = apBalance.map((ap, i) => i === 0 ? 0 : ap - apBalance[i - 1]);
  const accruedChange: number[] = accruedBalance.map((a, i) => i === 0 ? 0 : a - accruedBalance[i - 1]);
  const otherCLChange: number[] = otherCLBalance.map((o, i) => i === 0 ? 0 : o - otherCLBalance[i - 1]);

  const nwc: number[] = arBalance.map((ar, i) => 
    ar + inventoryBalance[i] + prepaidBalance[i] + otherCABalance[i] -
    apBalance[i] - accruedBalance[i] - otherCLBalance[i]
  );
  const nwcPercent: number[] = nwc.map((n, i) => n / revenue[i]);
  const nwcChange: number[] = nwc.map((n, i) => i === 0 ? 0 : n - nwc[i - 1]);
  const cashConversionCycle: number[] = new Array(years + 1).fill(
    assumptions.dso + assumptions.dio - assumptions.dpo
  );

  // ============ PP&E SCHEDULE ============
  const capexArr: number[] = [assumptions.historicalPPE * 0.1]; // Estimate historical CapEx
  const depreciation: number[] = [assumptions.historicalDA];
  const ppeNet: number[] = [assumptions.historicalPPE];
  const ppeGross: number[] = [assumptions.historicalPPE * 1.5]; // Estimate gross
  const accumulatedDepreciation: number[] = [ppeGross[0] - ppeNet[0]];

  for (let i = 1; i <= years; i++) {
    const capexRate = assumptions.capexPercent[i - 1] || 0.05;
    capexArr.push(revenue[i] * capexRate);
    depreciation.push(da[i]);
    ppeGross.push(ppeGross[i - 1] + capexArr[i]);
    accumulatedDepreciation.push(accumulatedDepreciation[i - 1] + depreciation[i]);
    ppeNet.push(ppeGross[i] - accumulatedDepreciation[i]);
  }

  const capexPercent: number[] = capexArr.map((c, i) => c / revenue[i]);
  const capexToDA: number[] = capexArr.map((c, i) => depreciation[i] > 0 ? c / depreciation[i] : 0);

  // ============ EQUITY SCHEDULE ============
  const commonStock: number[] = new Array(years + 1).fill(5); // Nominal par value
  const stockBasedComp: number[] = revenue.map(r => r * assumptions.stockBasedCompPercent);
  const apic: number[] = [assumptions.historicalEquity * 0.4]; // Estimate APIC
  const retainedEarnings: number[] = [assumptions.historicalEquity - commonStock[0] - apic[0]];
  const treasuryStock: number[] = new Array(years + 1).fill(0);
  const aoci: number[] = new Array(years + 1).fill(0);
  const dividendsPaid: number[] = [0];

  for (let i = 1; i <= years; i++) {
    apic.push(apic[i - 1] + stockBasedComp[i]);
    const dividend = assumptions.dividendsPerShare > 0 
      ? assumptions.dividendsPerShare * sharesOutstanding[i]
      : netIncome[i] * assumptions.payoutRatio;
    dividendsPaid.push(dividend);
    retainedEarnings.push(retainedEarnings[i - 1] + netIncome[i] - dividend);
    treasuryStock.push(treasuryStock[i - 1] + assumptions.shareRepurchases);
  }

  const totalEquity: number[] = commonStock.map((cs, i) => 
    cs + apic[i] + retainedEarnings[i] - treasuryStock[i] + aoci[i]
  );

  // ============ CASH FLOW STATEMENT ============
  const cfoNetIncome: number[] = [...netIncome];
  const cfoDepreciation: number[] = [...depreciation];
  const cfoStockBasedComp: number[] = [...stockBasedComp];
  const cfoDeferredTaxes: number[] = new Array(years + 1).fill(0);
  const cfoOtherNonCash: number[] = new Array(years + 1).fill(0);
  const totalNonCashAdj: number[] = cfoDepreciation.map((d, i) => 
    d + cfoStockBasedComp[i] + cfoDeferredTaxes[i] + cfoOtherNonCash[i]
  );

  const wcChangeAR: number[] = arChange.map(c => -c);
  const wcChangeInv: number[] = inventoryChange.map(c => -c);
  const wcChangePrepaid: number[] = prepaidChange.map(c => -c);
  const wcChangeOtherCA: number[] = otherCAChange.map(c => -c);
  const wcChangeAP: number[] = [...apChange];
  const wcChangeAccrued: number[] = [...accruedChange];
  const wcChangeDeferredRev: number[] = new Array(years + 1).fill(0);
  const wcChangeOtherCL: number[] = [...otherCLChange];
  const totalWCChange: number[] = wcChangeAR.map((ar, i) => 
    ar + wcChangeInv[i] + wcChangePrepaid[i] + wcChangeOtherCA[i] +
    wcChangeAP[i] + wcChangeAccrued[i] + wcChangeDeferredRev[i] + wcChangeOtherCL[i]
  );

  const cfo: number[] = cfoNetIncome.map((ni, i) => 
    ni + totalNonCashAdj[i] + totalWCChange[i]
  );

  const cfiCapex: number[] = capexArr.map(c => -c);
  const cfiAcquisitions: number[] = new Array(years + 1).fill(0);
  const cfiAssetSales: number[] = new Array(years + 1).fill(0);
  const cfiOther: number[] = new Array(years + 1).fill(0);
  const cfi: number[] = cfiCapex.map((c, i) => 
    c + cfiAcquisitions[i] + cfiAssetSales[i] + cfiOther[i]
  );

  const cffDebtProceeds: number[] = new Array(years + 1).fill(0);
  const cffDebtRepayments: number[] = termDebtAmortization.map(a => -a);
  const cffRevolverChange: number[] = revolverDraws.map((d, i) => d - revolverPaydowns[i]);
  const cffEquityProceeds: number[] = new Array(years + 1).fill(0);
  const cffShareRepurchases: number[] = new Array(years + 1).fill(-assumptions.shareRepurchases);
  cffShareRepurchases[0] = 0;
  const cffDividends: number[] = dividendsPaid.map(d => -d);
  const cffOther: number[] = new Array(years + 1).fill(0);
  const cff: number[] = cffDebtProceeds.map((d, i) =>
    d + cffDebtRepayments[i] + cffRevolverChange[i] + cffEquityProceeds[i] +
    cffShareRepurchases[i] + cffDividends[i] + cffOther[i]
  );

  const netCashChange: number[] = cfo.map((o, i) => o + cfi[i] + cff[i]);
  const beginningCash: number[] = [assumptions.historicalCash];
  const endingCash: number[] = [assumptions.historicalCash];

  // Solve for ending cash iteratively (circular reference handling)
  for (let i = 1; i <= years; i++) {
    beginningCash.push(endingCash[i - 1]);
    let preliminaryCash = beginningCash[i] + netCashChange[i];
    
    // Check if we need revolver
    if (preliminaryCash < assumptions.minimumCashBalance) {
      const shortfall = assumptions.minimumCashBalance - preliminaryCash;
      const draw = Math.min(shortfall, assumptions.revolverSize - revolverEnding[i - 1]);
      revolverDraws[i] = draw;
      revolverEnding[i] = revolverEnding[i - 1] + draw;
      preliminaryCash += draw;
    } else if (preliminaryCash > assumptions.minimumCashBalance && revolverEnding[i - 1] > 0) {
      // Pay down revolver with excess cash
      const excess = preliminaryCash - assumptions.minimumCashBalance;
      const paydown = Math.min(excess, revolverEnding[i - 1]);
      revolverPaydowns[i] = paydown;
      revolverEnding[i] = revolverEnding[i - 1] - paydown;
      preliminaryCash -= paydown;
    }
    
    endingCash[i] = Math.max(preliminaryCash, assumptions.minimumCashBalance);
  }

  // Update interest income based on cash
  for (let i = 0; i <= years; i++) {
    const avgCash = i === 0 ? endingCash[i] : (beginningCash[i] + endingCash[i]) / 2;
    interestIncome[i] = avgCash * assumptions.interestOnCash;
  }

  const freeCashFlow: number[] = cfo.map((o, i) => o + cfiCapex[i]);

  // ============ BALANCE SHEET ============
  const cash: number[] = [...endingCash];
  const totalCurrentAssets: number[] = cash.map((c, i) => 
    c + arBalance[i] + inventoryBalance[i] + prepaidBalance[i] + otherCABalance[i]
  );
  const intangibleAssets: number[] = new Array(years + 1).fill(0);
  const goodwill: number[] = new Array(years + 1).fill(0);
  const otherLongTermAssets: number[] = new Array(years + 1).fill(0);
  const totalNonCurrentAssets: number[] = ppeNet.map((ppe, i) => 
    ppe + intangibleAssets[i] + goodwill[i] + otherLongTermAssets[i]
  );
  const totalAssets: number[] = totalCurrentAssets.map((ca, i) => ca + totalNonCurrentAssets[i]);

  const deferredRevenue: number[] = new Array(years + 1).fill(0);
  const currentPortionDebt: number[] = termDebtAmortization.map((a, i) => 
    i < years ? termDebtAmortization[i + 1] : 0
  );
  const totalCurrentLiabilities: number[] = apBalance.map((ap, i) => 
    ap + accruedBalance[i] + deferredRevenue[i] + currentPortionDebt[i] + revolverEnding[i] + otherCLBalance[i]
  );
  const longTermDebt: number[] = termDebtEnding.map((d, i) => Math.max(0, d - currentPortionDebt[i]));
  const deferredTaxLiabilities: number[] = new Array(years + 1).fill(0);
  const otherLongTermLiabilities: number[] = new Array(years + 1).fill(0);
  const totalNonCurrentLiabilities: number[] = longTermDebt.map((ltd, i) => 
    ltd + deferredTaxLiabilities[i] + otherLongTermLiabilities[i]
  );
  const totalLiabilities: number[] = totalCurrentLiabilities.map((cl, i) => cl + totalNonCurrentLiabilities[i]);
  const totalLiabilitiesEquity: number[] = totalLiabilities.map((l, i) => l + totalEquity[i]);
  const balanceCheck: number[] = totalAssets.map((a, i) => Math.abs(a - totalLiabilitiesEquity[i]));

  // ============ RATIO ANALYSIS ============
  const roe: number[] = netIncome.map((ni, i) => {
    const avgEquity = i === 0 ? totalEquity[i] : (totalEquity[i] + totalEquity[i - 1]) / 2;
    return avgEquity > 0 ? ni / avgEquity : 0;
  });
  const roa: number[] = netIncome.map((ni, i) => {
    const avgAssets = i === 0 ? totalAssets[i] : (totalAssets[i] + totalAssets[i - 1]) / 2;
    return avgAssets > 0 ? ni / avgAssets : 0;
  });
  const nopat: number[] = ebit.map((e, i) => e * (1 - assumptions.effectiveTaxRate));
  const investedCapital: number[] = totalEquity.map((eq, i) => eq + termDebtEnding[i] + revolverEnding[i] - cash[i]);
  const roic: number[] = nopat.map((n, i) => {
    const avgIC = i === 0 ? investedCapital[i] : (investedCapital[i] + investedCapital[i - 1]) / 2;
    return avgIC > 0 ? n / avgIC : 0;
  });

  const currentRatio: number[] = totalCurrentAssets.map((ca, i) => 
    totalCurrentLiabilities[i] > 0 ? ca / totalCurrentLiabilities[i] : 0
  );
  const quickRatio: number[] = totalCurrentAssets.map((ca, i) => 
    totalCurrentLiabilities[i] > 0 ? (ca - inventoryBalance[i]) / totalCurrentLiabilities[i] : 0
  );
  const cashRatio: number[] = cash.map((c, i) => 
    totalCurrentLiabilities[i] > 0 ? c / totalCurrentLiabilities[i] : 0
  );

  const totalDebt: number[] = termDebtEnding.map((t, i) => t + revolverEnding[i]);
  const debtToEquity: number[] = totalDebt.map((d, i) => totalEquity[i] > 0 ? d / totalEquity[i] : 0);
  const debtToEBITDA: number[] = totalDebt.map((d, i) => ebitda[i] > 0 ? d / ebitda[i] : 0);
  const netDebt: number[] = totalDebt.map((d, i) => d - cash[i]);
  const netDebtToEBITDA: number[] = netDebt.map((nd, i) => ebitda[i] > 0 ? nd / ebitda[i] : 0);
  const interestCoverage: number[] = ebit.map((e, i) => totalInterest[i] > 0 ? e / totalInterest[i] : 0);

  const assetTurnover: number[] = revenue.map((r, i) => {
    const avgAssets = i === 0 ? totalAssets[i] : (totalAssets[i] + totalAssets[i - 1]) / 2;
    return avgAssets > 0 ? r / avgAssets : 0;
  });
  const inventoryTurnover: number[] = cogs.map((c, i) => {
    const avgInv = i === 0 ? inventoryBalance[i] : (inventoryBalance[i] + inventoryBalance[i - 1]) / 2;
    return avgInv > 0 ? c / avgInv : 0;
  });
  const receivablesTurnover: number[] = revenue.map((r, i) => {
    const avgAR = i === 0 ? arBalance[i] : (arBalance[i] + arBalance[i - 1]) / 2;
    return avgAR > 0 ? r / avgAR : 0;
  });
  const payablesTurnover: number[] = cogs.map((c, i) => {
    const avgAP = i === 0 ? apBalance[i] : (apBalance[i] + apBalance[i - 1]) / 2;
    return avgAP > 0 ? c / avgAP : 0;
  });

  const revenueGrowthArr: number[] = revenue.map((r, i) => i === 0 ? 0 : (r - revenue[i - 1]) / revenue[i - 1]);
  const ebitdaGrowth: number[] = ebitda.map((e, i) => i === 0 ? 0 : ebitda[i - 1] > 0 ? (e - ebitda[i - 1]) / ebitda[i - 1] : 0);
  const netIncomeGrowth: number[] = netIncome.map((n, i) => i === 0 ? 0 : netIncome[i - 1] > 0 ? (n - netIncome[i - 1]) / netIncome[i - 1] : 0);
  const epsGrowth: number[] = eps.map((e, i) => i === 0 ? 0 : eps[i - 1] > 0 ? (e - eps[i - 1]) / eps[i - 1] : 0);
  const bookValuePerShare: number[] = totalEquity.map((eq, i) => eq / sharesOutstanding[i]);

  // ============ SUMMARY METRICS ============
  const calcCAGR = (start: number, end: number, periods: number) => 
    start > 0 && end > 0 ? Math.pow(end / start, 1 / periods) - 1 : 0;

  const revenueCAGR = calcCAGR(revenue[0], revenue[years], years);
  const ebitdaCAGR = calcCAGR(ebitda[0], ebitda[years], years);
  const netIncomeCAGR = calcCAGR(Math.max(netIncome[0], 1), Math.max(netIncome[years], 1), years);
  const epsCAGR = calcCAGR(Math.max(eps[0], 0.01), Math.max(eps[years], 0.01), years);
  const endingNetDebtToEBITDA = netDebtToEBITDA[years];
  const endingDebtToEquity = debtToEquity[years];
  const averageROIC = roic.slice(1).reduce((a, b) => a + b, 0) / years;
  const isBalanced = balanceCheck.every(c => c < 0.01);
  const cashFlowReconciled = endingCash.every((c, i) => i === 0 || Math.abs(c - (beginningCash[i] + netCashChange[i])) < 0.01);

  return {
    assumptions,
    incomeStatement: {
      years: yearLabels,
      revenue,
      revenueGrowth,
      cogs,
      grossProfit,
      grossMargin,
      sga,
      sgaPercent,
      rd,
      rdPercent: rd.map((r, i) => r / revenue[i]),
      totalOpex,
      ebitda,
      ebitdaMargin,
      da,
      ebit,
      ebitMargin,
      interestExpense: totalInterest,
      interestIncome,
      netInterest,
      otherIncomeExpense,
      ebt,
      incomeTax,
      effectiveTaxRate: effectiveTaxRateArr,
      netIncome,
      netMargin,
      sharesOutstanding,
      eps,
    },
    balanceSheet: {
      years: yearLabels,
      cash,
      accountsReceivable: arBalance,
      inventory: inventoryBalance,
      prepaidExpenses: prepaidBalance,
      otherCurrentAssets: otherCABalance,
      totalCurrentAssets,
      ppeGross,
      accumulatedDepreciation,
      ppeNet,
      intangibleAssets,
      goodwill,
      otherLongTermAssets,
      totalNonCurrentAssets,
      totalAssets,
      accountsPayable: apBalance,
      accruedExpenses: accruedBalance,
      deferredRevenue,
      currentPortionDebt,
      revolverBalance: revolverEnding,
      otherCurrentLiabilities: otherCLBalance,
      totalCurrentLiabilities,
      longTermDebt,
      deferredTaxLiabilities,
      otherLongTermLiabilities,
      totalNonCurrentLiabilities,
      totalLiabilities,
      commonStock,
      apic,
      retainedEarnings,
      treasuryStock,
      aoci,
      totalEquity,
      totalLiabilitiesEquity,
      balanceCheck,
    },
    cashFlow: {
      years: yearLabels,
      netIncome: cfoNetIncome,
      depreciation: cfoDepreciation,
      stockBasedComp: cfoStockBasedComp,
      deferredTaxes: cfoDeferredTaxes,
      otherNonCash: cfoOtherNonCash,
      totalNonCashAdjustments: totalNonCashAdj,
      changeInAR: wcChangeAR,
      changeInInventory: wcChangeInv,
      changeInPrepaid: wcChangePrepaid,
      changeInOtherCA: wcChangeOtherCA,
      changeInAP: wcChangeAP,
      changeInAccrued: wcChangeAccrued,
      changeInDeferredRev: wcChangeDeferredRev,
      changeInOtherCL: wcChangeOtherCL,
      totalWorkingCapitalChange: totalWCChange,
      cfo,
      capex: cfiCapex,
      acquisitions: cfiAcquisitions,
      assetSales: cfiAssetSales,
      otherInvesting: cfiOther,
      cfi,
      debtProceeds: cffDebtProceeds,
      debtRepayments: cffDebtRepayments,
      revolverChange: cffRevolverChange,
      equityProceeds: cffEquityProceeds,
      shareRepurchases: cffShareRepurchases,
      dividendsPaid: cffDividends,
      otherFinancing: cffOther,
      cff,
      netCashChange,
      beginningCash,
      endingCash,
      freeCashFlow,
    },
    debtSchedule: {
      years: yearLabels,
      termDebtBeginning,
      termDebtAmortization,
      termDebtOptionalPrepay: new Array(years + 1).fill(0),
      termDebtEnding,
      termDebtAverage,
      termDebtRate: new Array(years + 1).fill(assumptions.existingDebtRate),
      termDebtInterest,
      revolverBeginning,
      revolverDraws,
      revolverPaydowns,
      revolverEnding,
      revolverAvailable: revolverEnding.map(e => assumptions.revolverSize - e),
      revolverAverage,
      revolverRate: new Array(years + 1).fill(assumptions.revolverRate),
      revolverInterest,
      revolverCommitmentFee,
      totalDebtBeginning: termDebtBeginning.map((t, i) => t + revolverBeginning[i]),
      totalDebtEnding: totalDebt,
      totalInterestExpense: totalInterest,
      netDebt,
      debtToEBITDA,
      netDebtToEBITDA,
      interestCoverage,
    },
    workingCapital: {
      years: yearLabels,
      arBalance,
      arChange,
      inventoryBalance,
      inventoryChange,
      prepaidBalance,
      prepaidChange,
      otherCABalance,
      otherCAChange,
      apBalance,
      apChange,
      accruedBalance,
      accruedChange,
      otherCLBalance,
      otherCLChange,
      nwc,
      nwcPercent,
      nwcChange,
      cashConversionCycle,
    },
    ppeSchedule: {
      years: yearLabels,
      beginningPPE: ppeNet.map((p, i) => i === 0 ? p : ppeNet[i - 1]),
      capex: capexArr,
      disposals: new Array(years + 1).fill(0),
      depreciation,
      endingPPE: ppeNet,
      capexPercent,
      capexToDA,
    },
    equitySchedule: {
      years: yearLabels,
      commonStockBeginning: commonStock,
      commonStockEnding: commonStock,
      apicBeginning: apic.map((a, i) => i === 0 ? a : apic[i - 1]),
      stockBasedComp,
      apicEnding: apic,
      retainedEarningsBeginning: retainedEarnings.map((r, i) => i === 0 ? r : retainedEarnings[i - 1]),
      netIncome,
      dividends: dividendsPaid,
      retainedEarningsEnding: retainedEarnings,
      treasuryStockBeginning: treasuryStock,
      shareRepurchases: new Array(years + 1).fill(assumptions.shareRepurchases),
      treasuryStockEnding: treasuryStock,
      totalEquity,
      sharesBeginning: sharesOutstanding,
      sharesIssued: new Array(years + 1).fill(0),
      sharesRepurchased: new Array(years + 1).fill(0),
      sharesEnding: sharesOutstanding,
      dividendsPerShare: dividendsPaid.map((d, i) => d / sharesOutstanding[i]),
      payoutRatio: dividendsPaid.map((d, i) => netIncome[i] > 0 ? d / netIncome[i] : 0),
    },
    ratioAnalysis: {
      years: yearLabels,
      grossMargin,
      ebitdaMargin,
      ebitMargin,
      netMargin,
      roe,
      roa,
      roic,
      currentRatio,
      quickRatio,
      cashRatio,
      debtToEquity,
      debtToEBITDA,
      netDebtToEBITDA,
      interestCoverage,
      assetTurnover,
      inventoryTurnover,
      receivablesTurnover,
      payablesTurnover,
      cashConversionCycle,
      revenueGrowth: revenueGrowthArr,
      ebitdaGrowth,
      netIncomeGrowth,
      epsGrowth,
      basicEPS: eps,
      bookValuePerShare,
    },
    summary: {
      revenueCAGR,
      ebitdaCAGR,
      netIncomeCAGR,
      epsCAGR,
      endingNetDebtToEBITDA,
      endingDebtToEquity,
      averageROIC,
      isBalanced,
      cashFlowReconciled,
    },
  };
}

export async function generateThreeStatementExcel(
  result: ThreeStatementModelResult
): Promise<Buffer> {
  const workbook = new ExcelJS.Workbook();
  workbook.creator = 'Cognitive Analysis Platform - Finance Panel';
  workbook.created = new Date();

  const currencyFormat = '"$"#,##0';
  const percentFormat = '0.0%';
  const ratioFormat = '0.0"x"';
  const numberFormat = '#,##0.0';

  const { assumptions, incomeStatement, balanceSheet, cashFlow, debtSchedule, workingCapital, ppeSchedule, equitySchedule, ratioAnalysis, summary } = result;
  const years = assumptions.projectionYears;

  // ============ TAB 1: EXECUTIVE SUMMARY ============
  const summarySheet = workbook.addWorksheet('Executive_Summary');
  summarySheet.columns = [
    { width: 30 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }
  ];

  summarySheet.getCell('A1').value = assumptions.companyName;
  summarySheet.getCell('A1').font = { bold: true, size: 16 };
  summarySheet.getCell('A2').value = 'Three-Statement Integrated Financial Model';
  summarySheet.getCell('A2').font = { bold: true, size: 14 };
  summarySheet.getCell('A3').value = `Model Date: ${new Date().toLocaleDateString()}`;
  summarySheet.getCell('A4').value = `Projection Period: ${years} Years`;
  summarySheet.getCell('A5').value = `Currency: ${assumptions.currency} (millions)`;

  summarySheet.getCell('A7').value = 'KEY FINANCIAL HIGHLIGHTS';
  summarySheet.getCell('A7').font = { bold: true };
  summarySheet.getRow(7).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };

  let row = 8;
  const headers = ['Metric', ...incomeStatement.years];
  headers.forEach((h, col) => {
    summarySheet.getCell(row, col + 1).value = h;
    summarySheet.getCell(row, col + 1).font = { bold: true };
  });
  row++;

  const summaryMetrics = [
    { label: 'Revenue', data: incomeStatement.revenue, format: currencyFormat },
    { label: 'Revenue Growth %', data: incomeStatement.revenueGrowth, format: percentFormat },
    { label: 'Gross Profit', data: incomeStatement.grossProfit, format: currencyFormat },
    { label: 'Gross Margin %', data: incomeStatement.grossMargin, format: percentFormat },
    { label: 'EBITDA', data: incomeStatement.ebitda, format: currencyFormat },
    { label: 'EBITDA Margin %', data: incomeStatement.ebitdaMargin, format: percentFormat },
    { label: 'Net Income', data: incomeStatement.netIncome, format: currencyFormat },
    { label: 'Net Margin %', data: incomeStatement.netMargin, format: percentFormat },
    { label: 'EPS', data: incomeStatement.eps, format: '"$"#,##0.00' },
    { label: 'Total Debt', data: debtSchedule.totalDebtEnding, format: currencyFormat },
    { label: 'Cash', data: balanceSheet.cash, format: currencyFormat },
    { label: 'Net Debt', data: debtSchedule.netDebt, format: currencyFormat },
    { label: 'Total Equity', data: balanceSheet.totalEquity, format: currencyFormat },
  ];

  summaryMetrics.forEach(metric => {
    summarySheet.getCell(row, 1).value = metric.label;
    metric.data.forEach((val, col) => {
      summarySheet.getCell(row, col + 2).value = val;
      summarySheet.getCell(row, col + 2).numFmt = metric.format;
    });
    row++;
  });

  row += 2;
  summarySheet.getCell(row, 1).value = 'KEY METRICS SUMMARY';
  summarySheet.getCell(row, 1).font = { bold: true };
  row++;

  const keyMetrics = [
    { label: 'Revenue CAGR (5-Year)', value: summary.revenueCAGR, format: percentFormat },
    { label: 'EBITDA CAGR (5-Year)', value: summary.ebitdaCAGR, format: percentFormat },
    { label: 'Net Income CAGR (5-Year)', value: summary.netIncomeCAGR, format: percentFormat },
    { label: 'EPS CAGR (5-Year)', value: summary.epsCAGR, format: percentFormat },
    { label: 'Ending Net Debt / EBITDA', value: summary.endingNetDebtToEBITDA, format: ratioFormat },
    { label: 'Ending Debt / Equity', value: summary.endingDebtToEquity, format: ratioFormat },
    { label: 'Average ROIC', value: summary.averageROIC, format: percentFormat },
  ];

  keyMetrics.forEach(metric => {
    summarySheet.getCell(row, 1).value = metric.label;
    summarySheet.getCell(row, 2).value = metric.value;
    summarySheet.getCell(row, 2).numFmt = metric.format;
    row++;
  });

  row += 2;
  summarySheet.getCell(row, 1).value = 'BALANCE SHEET CHECK';
  summarySheet.getCell(row, 1).font = { bold: true };
  row++;
  summarySheet.getCell(row, 1).value = 'Assets = Liabilities + Equity:';
  summarySheet.getCell(row, 2).value = summary.isBalanced ? 'BALANCED' : 'NOT BALANCED';
  summarySheet.getCell(row, 2).font = { bold: true, color: { argb: summary.isBalanced ? 'FF008000' : 'FFFF0000' } };
  row++;
  summarySheet.getCell(row, 1).value = 'Cash Flow Reconciliation:';
  summarySheet.getCell(row, 2).value = summary.cashFlowReconciled ? 'VERIFIED' : 'ERROR';
  summarySheet.getCell(row, 2).font = { bold: true, color: { argb: summary.cashFlowReconciled ? 'FF008000' : 'FFFF0000' } };

  // ============ TAB 2: ASSUMPTIONS ============
  const assumpSheet = workbook.addWorksheet('Assumptions');
  assumpSheet.columns = [{ width: 40 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }];

  row = 1;
  assumpSheet.getCell(row, 1).value = 'MODEL ASSUMPTIONS';
  assumpSheet.getCell(row, 1).font = { bold: true, size: 14 };
  row += 2;

  const addAssumptionSection = (title: string, items: { label: string; value: any; format?: string }[]) => {
    assumpSheet.getCell(row, 1).value = title;
    assumpSheet.getCell(row, 1).font = { bold: true };
    assumpSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
    row++;
    items.forEach(item => {
      assumpSheet.getCell(row, 1).value = item.label;
      if (Array.isArray(item.value)) {
        item.value.forEach((v, col) => {
          assumpSheet.getCell(row, col + 2).value = v;
          if (item.format) assumpSheet.getCell(row, col + 2).numFmt = item.format;
        });
      } else {
        assumpSheet.getCell(row, 2).value = item.value;
        if (item.format) assumpSheet.getCell(row, 2).numFmt = item.format;
      }
      assumpSheet.getCell(row, 1).font = { color: { argb: 'FF0000FF' } };
      row++;
    });
    row++;
  };

  addAssumptionSection('REVENUE ASSUMPTIONS', [
    { label: 'Base Year Revenue ($M)', value: assumptions.historicalRevenue, format: currencyFormat },
    { label: 'Revenue Growth Rates (Y1-Y5)', value: assumptions.revenueGrowthRates, format: percentFormat },
  ]);

  addAssumptionSection('COST STRUCTURE ASSUMPTIONS', [
    { label: 'Base Gross Margin', value: assumptions.baseGrossMargin, format: percentFormat },
    { label: 'Target Gross Margin (Year 5)', value: assumptions.targetGrossMargin, format: percentFormat },
    { label: 'Base SG&A % of Revenue', value: assumptions.baseSGAPercent, format: percentFormat },
    { label: 'Target SG&A % of Revenue', value: assumptions.targetSGAPercent, format: percentFormat },
    { label: 'R&D % of Revenue', value: assumptions.rdPercent, format: percentFormat },
    { label: 'D&A % of Revenue', value: assumptions.daPercent, format: percentFormat },
  ]);

  addAssumptionSection('WORKING CAPITAL ASSUMPTIONS', [
    { label: 'Days Sales Outstanding (DSO)', value: assumptions.dso },
    { label: 'Days Inventory Outstanding (DIO)', value: assumptions.dio },
    { label: 'Days Payable Outstanding (DPO)', value: assumptions.dpo },
    { label: 'Prepaid Expenses % of Revenue', value: assumptions.prepaidPercent, format: percentFormat },
    { label: 'Accrued Liabilities % of OpEx', value: assumptions.accruedPercent, format: percentFormat },
  ]);

  addAssumptionSection('CAPITAL EXPENDITURE ASSUMPTIONS', [
    { label: 'CapEx % of Revenue (Y1-Y5)', value: assumptions.capexPercent, format: percentFormat },
  ]);

  addAssumptionSection('DEBT ASSUMPTIONS', [
    { label: 'Beginning Term Debt ($M)', value: assumptions.existingDebtBalance, format: currencyFormat },
    { label: 'Term Debt Interest Rate', value: assumptions.existingDebtRate, format: percentFormat },
    { label: 'Mandatory Amortization ($M/year)', value: assumptions.debtAmortization, format: currencyFormat },
    { label: 'Revolver Size ($M)', value: assumptions.revolverSize, format: currencyFormat },
    { label: 'Revolver Interest Rate', value: assumptions.revolverRate, format: percentFormat },
    { label: 'Revolver Commitment Fee', value: assumptions.revolverCommitmentFee, format: percentFormat },
    { label: 'Minimum Cash Balance ($M)', value: assumptions.minimumCashBalance, format: currencyFormat },
  ]);

  addAssumptionSection('TAX & EQUITY ASSUMPTIONS', [
    { label: 'Effective Tax Rate', value: assumptions.effectiveTaxRate, format: percentFormat },
    { label: 'Stock-Based Comp % of Revenue', value: assumptions.stockBasedCompPercent, format: percentFormat },
    { label: 'Dividends Per Share', value: assumptions.dividendsPerShare, format: '"$"#,##0.00' },
    { label: 'Shares Outstanding (M)', value: assumptions.historicalSharesOutstanding },
  ]);

  // ============ TAB 3: INCOME STATEMENT ============
  const isSheet = workbook.addWorksheet('Income_Statement');
  isSheet.columns = [{ width: 35 }, ...Array(years + 1).fill({ width: 14 })];

  row = 1;
  isSheet.getCell(row, 1).value = 'INCOME STATEMENT';
  isSheet.getCell(row, 1).font = { bold: true, size: 14 };
  row++;
  isSheet.getCell(row, 1).value = '($ in millions)';
  row += 2;

  // Headers
  incomeStatement.years.forEach((y, col) => {
    isSheet.getCell(row, col + 2).value = y;
    isSheet.getCell(row, col + 2).font = { bold: true };
  });
  row++;

  const addISRow = (label: string, data: number[], format: string, indent: number = 0, bold: boolean = false) => {
    isSheet.getCell(row, 1).value = '  '.repeat(indent) + label;
    if (bold) isSheet.getCell(row, 1).font = { bold: true };
    data.forEach((val, col) => {
      isSheet.getCell(row, col + 2).value = val;
      isSheet.getCell(row, col + 2).numFmt = format;
      if (bold) isSheet.getCell(row, col + 2).font = { bold: true };
    });
    row++;
  };

  addISRow('Revenue', incomeStatement.revenue, currencyFormat, 0, true);
  addISRow('Growth %', incomeStatement.revenueGrowth, percentFormat, 1);
  row++;
  addISRow('Cost of Goods Sold', incomeStatement.cogs, currencyFormat);
  addISRow('Gross Profit', incomeStatement.grossProfit, currencyFormat, 0, true);
  addISRow('Gross Margin %', incomeStatement.grossMargin, percentFormat, 1);
  row++;
  addISRow('Selling, General & Administrative', incomeStatement.sga, currencyFormat);
  addISRow('SG&A % of Revenue', incomeStatement.sgaPercent, percentFormat, 1);
  addISRow('Research & Development', incomeStatement.rd, currencyFormat);
  addISRow('Total Operating Expenses', incomeStatement.totalOpex, currencyFormat, 0, true);
  row++;
  addISRow('EBITDA', incomeStatement.ebitda, currencyFormat, 0, true);
  addISRow('EBITDA Margin %', incomeStatement.ebitdaMargin, percentFormat, 1);
  row++;
  addISRow('Depreciation & Amortization', incomeStatement.da, currencyFormat);
  addISRow('EBIT (Operating Income)', incomeStatement.ebit, currencyFormat, 0, true);
  addISRow('EBIT Margin %', incomeStatement.ebitMargin, percentFormat, 1);
  row++;
  addISRow('Interest Expense', incomeStatement.interestExpense, currencyFormat);
  addISRow('Interest Income', incomeStatement.interestIncome, currencyFormat);
  addISRow('Net Interest Expense', incomeStatement.netInterest, currencyFormat);
  row++;
  addISRow('EBT (Earnings Before Tax)', incomeStatement.ebt, currencyFormat, 0, true);
  addISRow('Income Tax Expense', incomeStatement.incomeTax, currencyFormat);
  addISRow('Effective Tax Rate', incomeStatement.effectiveTaxRate, percentFormat, 1);
  row++;
  addISRow('NET INCOME', incomeStatement.netIncome, currencyFormat, 0, true);
  addISRow('Net Margin %', incomeStatement.netMargin, percentFormat, 1);
  row++;
  addISRow('Shares Outstanding', incomeStatement.sharesOutstanding, numberFormat);
  addISRow('Earnings Per Share (EPS)', incomeStatement.eps, '"$"#,##0.00', 0, true);

  // ============ TAB 4: BALANCE SHEET ============
  const bsSheet = workbook.addWorksheet('Balance_Sheet');
  bsSheet.columns = [{ width: 35 }, ...Array(years + 1).fill({ width: 14 })];

  row = 1;
  bsSheet.getCell(row, 1).value = 'BALANCE SHEET';
  bsSheet.getCell(row, 1).font = { bold: true, size: 14 };
  row++;
  bsSheet.getCell(row, 1).value = '($ in millions)';
  row += 2;

  balanceSheet.years.forEach((y, col) => {
    bsSheet.getCell(row, col + 2).value = y;
    bsSheet.getCell(row, col + 2).font = { bold: true };
  });
  row++;

  const addBSRow = (label: string, data: number[], format: string, indent: number = 0, bold: boolean = false) => {
    bsSheet.getCell(row, 1).value = '  '.repeat(indent) + label;
    if (bold) bsSheet.getCell(row, 1).font = { bold: true };
    data.forEach((val, col) => {
      bsSheet.getCell(row, col + 2).value = val;
      bsSheet.getCell(row, col + 2).numFmt = format;
      if (bold) bsSheet.getCell(row, col + 2).font = { bold: true };
    });
    row++;
  };

  bsSheet.getCell(row, 1).value = 'ASSETS';
  bsSheet.getCell(row, 1).font = { bold: true };
  bsSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addBSRow('Current Assets:', [], '', 0, true);
  addBSRow('Cash and Cash Equivalents', balanceSheet.cash, currencyFormat, 1);
  addBSRow('Accounts Receivable', balanceSheet.accountsReceivable, currencyFormat, 1);
  addBSRow('Inventory', balanceSheet.inventory, currencyFormat, 1);
  addBSRow('Prepaid Expenses', balanceSheet.prepaidExpenses, currencyFormat, 1);
  addBSRow('Other Current Assets', balanceSheet.otherCurrentAssets, currencyFormat, 1);
  addBSRow('Total Current Assets', balanceSheet.totalCurrentAssets, currencyFormat, 0, true);
  row++;
  addBSRow('Non-Current Assets:', [], '', 0, true);
  addBSRow('Property, Plant & Equipment, Net', balanceSheet.ppeNet, currencyFormat, 1);
  addBSRow('Intangible Assets', balanceSheet.intangibleAssets, currencyFormat, 1);
  addBSRow('Goodwill', balanceSheet.goodwill, currencyFormat, 1);
  addBSRow('Other Long-Term Assets', balanceSheet.otherLongTermAssets, currencyFormat, 1);
  addBSRow('Total Non-Current Assets', balanceSheet.totalNonCurrentAssets, currencyFormat, 0, true);
  row++;
  addBSRow('TOTAL ASSETS', balanceSheet.totalAssets, currencyFormat, 0, true);
  row += 2;

  bsSheet.getCell(row, 1).value = 'LIABILITIES';
  bsSheet.getCell(row, 1).font = { bold: true };
  bsSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addBSRow('Current Liabilities:', [], '', 0, true);
  addBSRow('Accounts Payable', balanceSheet.accountsPayable, currencyFormat, 1);
  addBSRow('Accrued Expenses', balanceSheet.accruedExpenses, currencyFormat, 1);
  addBSRow('Current Portion of Long-Term Debt', balanceSheet.currentPortionDebt, currencyFormat, 1);
  addBSRow('Revolver Balance', balanceSheet.revolverBalance, currencyFormat, 1);
  addBSRow('Other Current Liabilities', balanceSheet.otherCurrentLiabilities, currencyFormat, 1);
  addBSRow('Total Current Liabilities', balanceSheet.totalCurrentLiabilities, currencyFormat, 0, true);
  row++;
  addBSRow('Non-Current Liabilities:', [], '', 0, true);
  addBSRow('Long-Term Debt', balanceSheet.longTermDebt, currencyFormat, 1);
  addBSRow('Other Long-Term Liabilities', balanceSheet.otherLongTermLiabilities, currencyFormat, 1);
  addBSRow('Total Non-Current Liabilities', balanceSheet.totalNonCurrentLiabilities, currencyFormat, 0, true);
  row++;
  addBSRow('TOTAL LIABILITIES', balanceSheet.totalLiabilities, currencyFormat, 0, true);
  row += 2;

  bsSheet.getCell(row, 1).value = "SHAREHOLDERS' EQUITY";
  bsSheet.getCell(row, 1).font = { bold: true };
  bsSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addBSRow('Common Stock', balanceSheet.commonStock, currencyFormat, 1);
  addBSRow('Additional Paid-In Capital', balanceSheet.apic, currencyFormat, 1);
  addBSRow('Retained Earnings', balanceSheet.retainedEarnings, currencyFormat, 1);
  addBSRow('Treasury Stock', balanceSheet.treasuryStock, currencyFormat, 1);
  addBSRow("Total Shareholders' Equity", balanceSheet.totalEquity, currencyFormat, 0, true);
  row++;
  addBSRow('TOTAL LIABILITIES + EQUITY', balanceSheet.totalLiabilitiesEquity, currencyFormat, 0, true);
  row += 2;

  bsSheet.getCell(row, 1).value = 'BALANCE CHECK (Assets - L - E)';
  bsSheet.getCell(row, 1).font = { bold: true };
  balanceSheet.balanceCheck.forEach((val, col) => {
    bsSheet.getCell(row, col + 2).value = val < 0.01 ? 'BALANCED' : 'ERROR';
    bsSheet.getCell(row, col + 2).font = { 
      bold: true, 
      color: { argb: val < 0.01 ? 'FF008000' : 'FFFF0000' } 
    };
  });

  // ============ TAB 5: CASH FLOW STATEMENT ============
  const cfSheet = workbook.addWorksheet('Cash_Flow_Statement');
  cfSheet.columns = [{ width: 40 }, ...Array(years + 1).fill({ width: 14 })];

  row = 1;
  cfSheet.getCell(row, 1).value = 'CASH FLOW STATEMENT';
  cfSheet.getCell(row, 1).font = { bold: true, size: 14 };
  row++;
  cfSheet.getCell(row, 1).value = '($ in millions)';
  row += 2;

  cashFlow.years.forEach((y, col) => {
    cfSheet.getCell(row, col + 2).value = y;
    cfSheet.getCell(row, col + 2).font = { bold: true };
  });
  row++;

  const addCFRow = (label: string, data: number[], format: string, indent: number = 0, bold: boolean = false) => {
    cfSheet.getCell(row, 1).value = '  '.repeat(indent) + label;
    if (bold) cfSheet.getCell(row, 1).font = { bold: true };
    data.forEach((val, col) => {
      cfSheet.getCell(row, col + 2).value = val;
      cfSheet.getCell(row, col + 2).numFmt = format;
      if (bold) cfSheet.getCell(row, col + 2).font = { bold: true };
    });
    row++;
  };

  cfSheet.getCell(row, 1).value = 'OPERATING ACTIVITIES';
  cfSheet.getCell(row, 1).font = { bold: true };
  cfSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addCFRow('Net Income', cashFlow.netIncome, currencyFormat);
  row++;
  addCFRow('Adjustments for Non-Cash Items:', [], '');
  addCFRow('Depreciation & Amortization', cashFlow.depreciation, currencyFormat, 1);
  addCFRow('Stock-Based Compensation', cashFlow.stockBasedComp, currencyFormat, 1);
  addCFRow('Total Non-Cash Adjustments', cashFlow.totalNonCashAdjustments, currencyFormat, 0, true);
  row++;
  addCFRow('Changes in Working Capital:', [], '');
  addCFRow('(Increase)/Decrease in Accounts Receivable', cashFlow.changeInAR, currencyFormat, 1);
  addCFRow('(Increase)/Decrease in Inventory', cashFlow.changeInInventory, currencyFormat, 1);
  addCFRow('(Increase)/Decrease in Prepaid Expenses', cashFlow.changeInPrepaid, currencyFormat, 1);
  addCFRow('Increase/(Decrease) in Accounts Payable', cashFlow.changeInAP, currencyFormat, 1);
  addCFRow('Increase/(Decrease) in Accrued Expenses', cashFlow.changeInAccrued, currencyFormat, 1);
  addCFRow('Total Change in Working Capital', cashFlow.totalWorkingCapitalChange, currencyFormat, 0, true);
  row++;
  addCFRow('Cash Flow from Operating Activities', cashFlow.cfo, currencyFormat, 0, true);
  row += 2;

  cfSheet.getCell(row, 1).value = 'INVESTING ACTIVITIES';
  cfSheet.getCell(row, 1).font = { bold: true };
  cfSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addCFRow('Capital Expenditures', cashFlow.capex, currencyFormat, 1);
  addCFRow('Cash Flow from Investing Activities', cashFlow.cfi, currencyFormat, 0, true);
  row += 2;

  cfSheet.getCell(row, 1).value = 'FINANCING ACTIVITIES';
  cfSheet.getCell(row, 1).font = { bold: true };
  cfSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addCFRow('Debt Repayments', cashFlow.debtRepayments, currencyFormat, 1);
  addCFRow('Revolver Draws / (Paydowns)', cashFlow.revolverChange, currencyFormat, 1);
  addCFRow('Share Repurchases', cashFlow.shareRepurchases, currencyFormat, 1);
  addCFRow('Dividends Paid', cashFlow.dividendsPaid, currencyFormat, 1);
  addCFRow('Cash Flow from Financing Activities', cashFlow.cff, currencyFormat, 0, true);
  row += 2;

  addCFRow('NET CHANGE IN CASH', cashFlow.netCashChange, currencyFormat, 0, true);
  row++;
  addCFRow('Beginning Cash Balance', cashFlow.beginningCash, currencyFormat);
  addCFRow('Ending Cash Balance', cashFlow.endingCash, currencyFormat, 0, true);
  row += 2;
  addCFRow('Free Cash Flow (CFO - CapEx)', cashFlow.freeCashFlow, currencyFormat, 0, true);

  // ============ TAB 6: DEBT SCHEDULE ============
  const debtSheet = workbook.addWorksheet('Debt_Schedule');
  debtSheet.columns = [{ width: 35 }, ...Array(years + 1).fill({ width: 14 })];

  row = 1;
  debtSheet.getCell(row, 1).value = 'DEBT SCHEDULE';
  debtSheet.getCell(row, 1).font = { bold: true, size: 14 };
  row++;
  debtSheet.getCell(row, 1).value = '($ in millions)';
  row += 2;

  debtSchedule.years.forEach((y, col) => {
    debtSheet.getCell(row, col + 2).value = y;
    debtSheet.getCell(row, col + 2).font = { bold: true };
  });
  row++;

  const addDebtRow = (label: string, data: number[], format: string, indent: number = 0, bold: boolean = false) => {
    debtSheet.getCell(row, 1).value = '  '.repeat(indent) + label;
    if (bold) debtSheet.getCell(row, 1).font = { bold: true };
    data.forEach((val, col) => {
      debtSheet.getCell(row, col + 2).value = val;
      debtSheet.getCell(row, col + 2).numFmt = format;
      if (bold) debtSheet.getCell(row, col + 2).font = { bold: true };
    });
    row++;
  };

  debtSheet.getCell(row, 1).value = 'EXISTING TERM DEBT';
  debtSheet.getCell(row, 1).font = { bold: true };
  debtSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addDebtRow('Beginning Balance', debtSchedule.termDebtBeginning, currencyFormat);
  addDebtRow('Mandatory Amortization', debtSchedule.termDebtAmortization.map(v => -v), currencyFormat, 1);
  addDebtRow('Ending Balance', debtSchedule.termDebtEnding, currencyFormat, 0, true);
  addDebtRow('Average Balance', debtSchedule.termDebtAverage, currencyFormat);
  addDebtRow('Interest Rate', debtSchedule.termDebtRate, percentFormat);
  addDebtRow('Interest Expense', debtSchedule.termDebtInterest, currencyFormat);
  row++;

  debtSheet.getCell(row, 1).value = 'REVOLVING CREDIT FACILITY';
  debtSheet.getCell(row, 1).font = { bold: true };
  debtSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addDebtRow('Facility Size', new Array(years + 1).fill(assumptions.revolverSize), currencyFormat);
  addDebtRow('Beginning Balance', debtSchedule.revolverBeginning, currencyFormat);
  addDebtRow('Draws', debtSchedule.revolverDraws, currencyFormat, 1);
  addDebtRow('Paydowns', debtSchedule.revolverPaydowns.map(v => -v), currencyFormat, 1);
  addDebtRow('Ending Balance', debtSchedule.revolverEnding, currencyFormat, 0, true);
  addDebtRow('Available Capacity', debtSchedule.revolverAvailable, currencyFormat);
  addDebtRow('Interest Expense', debtSchedule.revolverInterest, currencyFormat);
  addDebtRow('Commitment Fee', debtSchedule.revolverCommitmentFee, currencyFormat);
  row++;

  debtSheet.getCell(row, 1).value = 'TOTAL DEBT SUMMARY';
  debtSheet.getCell(row, 1).font = { bold: true };
  debtSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addDebtRow('Total Debt (Ending)', debtSchedule.totalDebtEnding, currencyFormat, 0, true);
  addDebtRow('Total Interest Expense', debtSchedule.totalInterestExpense, currencyFormat, 0, true);
  addDebtRow('Net Debt (Debt - Cash)', debtSchedule.netDebt, currencyFormat);
  row++;
  addDebtRow('Total Debt / EBITDA', debtSchedule.debtToEBITDA, ratioFormat);
  addDebtRow('Net Debt / EBITDA', debtSchedule.netDebtToEBITDA, ratioFormat);
  addDebtRow('Interest Coverage (EBIT / Interest)', debtSchedule.interestCoverage, ratioFormat);

  // ============ TAB 7: WORKING CAPITAL SCHEDULE ============
  const wcSheet = workbook.addWorksheet('Working_Capital');
  wcSheet.columns = [{ width: 35 }, ...Array(years + 1).fill({ width: 14 })];

  row = 1;
  wcSheet.getCell(row, 1).value = 'WORKING CAPITAL SCHEDULE';
  wcSheet.getCell(row, 1).font = { bold: true, size: 14 };
  row++;
  wcSheet.getCell(row, 1).value = '($ in millions)';
  row += 2;

  workingCapital.years.forEach((y, col) => {
    wcSheet.getCell(row, col + 2).value = y;
    wcSheet.getCell(row, col + 2).font = { bold: true };
  });
  row++;

  const addWCRow = (label: string, data: number[], format: string, indent: number = 0, bold: boolean = false) => {
    wcSheet.getCell(row, 1).value = '  '.repeat(indent) + label;
    if (bold) wcSheet.getCell(row, 1).font = { bold: true };
    data.forEach((val, col) => {
      wcSheet.getCell(row, col + 2).value = val;
      wcSheet.getCell(row, col + 2).numFmt = format;
      if (bold) wcSheet.getCell(row, col + 2).font = { bold: true };
    });
    row++;
  };

  addWCRow('ACCOUNTS RECEIVABLE', [], '', 0, true);
  addWCRow('A/R Balance', workingCapital.arBalance, currencyFormat, 1);
  addWCRow('Change in A/R', workingCapital.arChange, currencyFormat, 1);
  row++;
  addWCRow('INVENTORY', [], '', 0, true);
  addWCRow('Inventory Balance', workingCapital.inventoryBalance, currencyFormat, 1);
  addWCRow('Change in Inventory', workingCapital.inventoryChange, currencyFormat, 1);
  row++;
  addWCRow('ACCOUNTS PAYABLE', [], '', 0, true);
  addWCRow('A/P Balance', workingCapital.apBalance, currencyFormat, 1);
  addWCRow('Change in A/P', workingCapital.apChange, currencyFormat, 1);
  row++;
  addWCRow('NET WORKING CAPITAL SUMMARY', [], '', 0, true);
  addWCRow('Net Working Capital', workingCapital.nwc, currencyFormat, 1);
  addWCRow('NWC as % of Revenue', workingCapital.nwcPercent, percentFormat, 1);
  addWCRow('Change in NWC', workingCapital.nwcChange, currencyFormat, 1);
  addWCRow('Cash Conversion Cycle (days)', workingCapital.cashConversionCycle, '0');

  // ============ TAB 8: PP&E SCHEDULE ============
  const ppeSheet = workbook.addWorksheet('PPE_Schedule');
  ppeSheet.columns = [{ width: 35 }, ...Array(years + 1).fill({ width: 14 })];

  row = 1;
  ppeSheet.getCell(row, 1).value = 'PP&E SCHEDULE';
  ppeSheet.getCell(row, 1).font = { bold: true, size: 14 };
  row++;
  ppeSheet.getCell(row, 1).value = '($ in millions)';
  row += 2;

  ppeSchedule.years.forEach((y, col) => {
    ppeSheet.getCell(row, col + 2).value = y;
    ppeSheet.getCell(row, col + 2).font = { bold: true };
  });
  row++;

  const addPPERow = (label: string, data: number[], format: string, indent: number = 0, bold: boolean = false) => {
    ppeSheet.getCell(row, 1).value = '  '.repeat(indent) + label;
    if (bold) ppeSheet.getCell(row, 1).font = { bold: true };
    data.forEach((val, col) => {
      ppeSheet.getCell(row, col + 2).value = val;
      ppeSheet.getCell(row, col + 2).numFmt = format;
      if (bold) ppeSheet.getCell(row, col + 2).font = { bold: true };
    });
    row++;
  };

  addPPERow('PP&E ROLLFORWARD', [], '', 0, true);
  addPPERow('Beginning PP&E (Net)', ppeSchedule.beginningPPE, currencyFormat, 1);
  addPPERow('Capital Expenditures', ppeSchedule.capex, currencyFormat, 1);
  addPPERow('Depreciation', ppeSchedule.depreciation.map(d => -d), currencyFormat, 1);
  addPPERow('Ending PP&E (Net)', ppeSchedule.endingPPE, currencyFormat, 0, true);
  row++;
  addPPERow('CAPEX ANALYSIS', [], '', 0, true);
  addPPERow('CapEx % of Revenue', ppeSchedule.capexPercent, percentFormat, 1);
  addPPERow('CapEx / D&A Ratio', ppeSchedule.capexToDA, ratioFormat, 1);

  // ============ TAB 9: SHAREHOLDERS' EQUITY ============
  const eqSheet = workbook.addWorksheet('Shareholders_Equity');
  eqSheet.columns = [{ width: 35 }, ...Array(years + 1).fill({ width: 14 })];

  row = 1;
  eqSheet.getCell(row, 1).value = "SHAREHOLDERS' EQUITY SCHEDULE";
  eqSheet.getCell(row, 1).font = { bold: true, size: 14 };
  row++;
  eqSheet.getCell(row, 1).value = '($ in millions)';
  row += 2;

  equitySchedule.years.forEach((y, col) => {
    eqSheet.getCell(row, col + 2).value = y;
    eqSheet.getCell(row, col + 2).font = { bold: true };
  });
  row++;

  const addEqRow = (label: string, data: number[], format: string, indent: number = 0, bold: boolean = false) => {
    eqSheet.getCell(row, 1).value = '  '.repeat(indent) + label;
    if (bold) eqSheet.getCell(row, 1).font = { bold: true };
    data.forEach((val, col) => {
      eqSheet.getCell(row, col + 2).value = val;
      eqSheet.getCell(row, col + 2).numFmt = format;
      if (bold) eqSheet.getCell(row, col + 2).font = { bold: true };
    });
    row++;
  };

  addEqRow('RETAINED EARNINGS', [], '', 0, true);
  addEqRow('Beginning Balance', equitySchedule.retainedEarningsBeginning, currencyFormat, 1);
  addEqRow('Net Income', equitySchedule.netIncome, currencyFormat, 1);
  addEqRow('Dividends', equitySchedule.dividends.map(d => -d), currencyFormat, 1);
  addEqRow('Ending Balance', equitySchedule.retainedEarningsEnding, currencyFormat, 0, true);
  row++;
  addEqRow('ADDITIONAL PAID-IN CAPITAL', [], '', 0, true);
  addEqRow('Beginning Balance', equitySchedule.apicBeginning, currencyFormat, 1);
  addEqRow('Stock-Based Compensation', equitySchedule.stockBasedComp, currencyFormat, 1);
  addEqRow('Ending Balance', equitySchedule.apicEnding, currencyFormat, 0, true);
  row++;
  addEqRow("TOTAL SHAREHOLDERS' EQUITY", equitySchedule.totalEquity, currencyFormat, 0, true);
  row++;
  addEqRow('SHARE COUNT', [], '', 0, true);
  addEqRow('Shares Outstanding (M)', equitySchedule.sharesEnding, numberFormat, 1);
  addEqRow('Dividends Per Share', equitySchedule.dividendsPerShare, '"$"#,##0.00', 1);
  addEqRow('Payout Ratio', equitySchedule.payoutRatio, percentFormat, 1);

  // ============ TAB 10: RATIO ANALYSIS ============
  const ratioSheet = workbook.addWorksheet('Ratio_Analysis');
  ratioSheet.columns = [{ width: 35 }, ...Array(years + 1).fill({ width: 14 })];

  row = 1;
  ratioSheet.getCell(row, 1).value = 'RATIO ANALYSIS';
  ratioSheet.getCell(row, 1).font = { bold: true, size: 14 };
  row += 2;

  ratioAnalysis.years.forEach((y, col) => {
    ratioSheet.getCell(row, col + 2).value = y;
    ratioSheet.getCell(row, col + 2).font = { bold: true };
  });
  row++;

  const addRatioRow = (label: string, data: number[], format: string, indent: number = 0, bold: boolean = false) => {
    ratioSheet.getCell(row, 1).value = '  '.repeat(indent) + label;
    if (bold) ratioSheet.getCell(row, 1).font = { bold: true };
    data.forEach((val, col) => {
      ratioSheet.getCell(row, col + 2).value = val;
      ratioSheet.getCell(row, col + 2).numFmt = format;
      if (bold) ratioSheet.getCell(row, col + 2).font = { bold: true };
    });
    row++;
  };

  ratioSheet.getCell(row, 1).value = 'PROFITABILITY RATIOS';
  ratioSheet.getCell(row, 1).font = { bold: true };
  ratioSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addRatioRow('Gross Margin %', ratioAnalysis.grossMargin, percentFormat);
  addRatioRow('EBITDA Margin %', ratioAnalysis.ebitdaMargin, percentFormat);
  addRatioRow('EBIT Margin %', ratioAnalysis.ebitMargin, percentFormat);
  addRatioRow('Net Margin %', ratioAnalysis.netMargin, percentFormat);
  addRatioRow('Return on Equity (ROE)', ratioAnalysis.roe, percentFormat);
  addRatioRow('Return on Assets (ROA)', ratioAnalysis.roa, percentFormat);
  addRatioRow('Return on Invested Capital (ROIC)', ratioAnalysis.roic, percentFormat);
  row++;

  ratioSheet.getCell(row, 1).value = 'LIQUIDITY RATIOS';
  ratioSheet.getCell(row, 1).font = { bold: true };
  ratioSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addRatioRow('Current Ratio', ratioAnalysis.currentRatio, ratioFormat);
  addRatioRow('Quick Ratio', ratioAnalysis.quickRatio, ratioFormat);
  addRatioRow('Cash Ratio', ratioAnalysis.cashRatio, ratioFormat);
  row++;

  ratioSheet.getCell(row, 1).value = 'LEVERAGE RATIOS';
  ratioSheet.getCell(row, 1).font = { bold: true };
  ratioSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addRatioRow('Debt / Equity', ratioAnalysis.debtToEquity, ratioFormat);
  addRatioRow('Total Debt / EBITDA', ratioAnalysis.debtToEBITDA, ratioFormat);
  addRatioRow('Net Debt / EBITDA', ratioAnalysis.netDebtToEBITDA, ratioFormat);
  addRatioRow('Interest Coverage Ratio', ratioAnalysis.interestCoverage, ratioFormat);
  row++;

  ratioSheet.getCell(row, 1).value = 'EFFICIENCY RATIOS';
  ratioSheet.getCell(row, 1).font = { bold: true };
  ratioSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addRatioRow('Asset Turnover', ratioAnalysis.assetTurnover, ratioFormat);
  addRatioRow('Inventory Turnover', ratioAnalysis.inventoryTurnover, ratioFormat);
  addRatioRow('Receivables Turnover', ratioAnalysis.receivablesTurnover, ratioFormat);
  addRatioRow('Payables Turnover', ratioAnalysis.payablesTurnover, ratioFormat);
  addRatioRow('Cash Conversion Cycle (days)', ratioAnalysis.cashConversionCycle, '0');
  row++;

  ratioSheet.getCell(row, 1).value = 'GROWTH METRICS';
  ratioSheet.getCell(row, 1).font = { bold: true };
  ratioSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addRatioRow('Revenue Growth %', ratioAnalysis.revenueGrowth, percentFormat);
  addRatioRow('EBITDA Growth %', ratioAnalysis.ebitdaGrowth, percentFormat);
  addRatioRow('Net Income Growth %', ratioAnalysis.netIncomeGrowth, percentFormat);
  addRatioRow('EPS Growth %', ratioAnalysis.epsGrowth, percentFormat);
  row++;

  ratioSheet.getCell(row, 1).value = 'PER SHARE METRICS';
  ratioSheet.getCell(row, 1).font = { bold: true };
  ratioSheet.getRow(row).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  row++;

  addRatioRow('Basic EPS', ratioAnalysis.basicEPS, '"$"#,##0.00');
  addRatioRow('Book Value per Share', ratioAnalysis.bookValuePerShare, '"$"#,##0.00');

  // ============ TAB 11: CHARTS DATA ============
  const chartsSheet = workbook.addWorksheet('Charts_Data');
  chartsSheet.columns = [{ width: 20 }, ...Array(years + 1).fill({ width: 14 })];

  row = 1;
  chartsSheet.getCell(row, 1).value = 'CHARTS DATA';
  chartsSheet.getCell(row, 1).font = { bold: true, size: 14 };
  chartsSheet.getCell(row + 1, 1).value = 'Data tables for creating charts';
  row += 3;

  // Revenue & EBITDA Margin chart data
  chartsSheet.getCell(row, 1).value = 'Revenue & EBITDA Margin';
  chartsSheet.getCell(row, 1).font = { bold: true };
  row++;
  incomeStatement.years.forEach((y, col) => {
    chartsSheet.getCell(row, col + 2).value = y;
    chartsSheet.getCell(row, col + 2).font = { bold: true };
  });
  row++;
  chartsSheet.getCell(row, 1).value = 'Revenue ($M)';
  incomeStatement.revenue.forEach((val, col) => {
    chartsSheet.getCell(row, col + 2).value = val;
    chartsSheet.getCell(row, col + 2).numFmt = currencyFormat;
  });
  row++;
  chartsSheet.getCell(row, 1).value = 'EBITDA Margin %';
  incomeStatement.ebitdaMargin.forEach((val, col) => {
    chartsSheet.getCell(row, col + 2).value = val;
    chartsSheet.getCell(row, col + 2).numFmt = percentFormat;
  });
  row += 2;

  // Leverage chart data
  chartsSheet.getCell(row, 1).value = 'Leverage Trajectory';
  chartsSheet.getCell(row, 1).font = { bold: true };
  row++;
  debtSchedule.years.forEach((y, col) => {
    chartsSheet.getCell(row, col + 2).value = y;
    chartsSheet.getCell(row, col + 2).font = { bold: true };
  });
  row++;
  chartsSheet.getCell(row, 1).value = 'Debt / EBITDA';
  debtSchedule.debtToEBITDA.forEach((val, col) => {
    chartsSheet.getCell(row, col + 2).value = val;
    chartsSheet.getCell(row, col + 2).numFmt = ratioFormat;
  });
  row++;
  chartsSheet.getCell(row, 1).value = 'Net Debt / EBITDA';
  debtSchedule.netDebtToEBITDA.forEach((val, col) => {
    chartsSheet.getCell(row, col + 2).value = val;
    chartsSheet.getCell(row, col + 2).numFmt = ratioFormat;
  });
  row += 2;

  // EPS chart data
  chartsSheet.getCell(row, 1).value = 'EPS Growth';
  chartsSheet.getCell(row, 1).font = { bold: true };
  row++;
  incomeStatement.years.forEach((y, col) => {
    chartsSheet.getCell(row, col + 2).value = y;
    chartsSheet.getCell(row, col + 2).font = { bold: true };
  });
  row++;
  chartsSheet.getCell(row, 1).value = 'EPS';
  incomeStatement.eps.forEach((val, col) => {
    chartsSheet.getCell(row, col + 2).value = val;
    chartsSheet.getCell(row, col + 2).numFmt = '"$"#,##0.00';
  });
  row++;
  chartsSheet.getCell(row, 1).value = 'EPS Growth %';
  ratioAnalysis.epsGrowth.forEach((val, col) => {
    chartsSheet.getCell(row, col + 2).value = val;
    chartsSheet.getCell(row, col + 2).numFmt = percentFormat;
  });

  // Return workbook as buffer
  const buffer = await workbook.xlsx.writeBuffer();
  return Buffer.from(buffer);
}
