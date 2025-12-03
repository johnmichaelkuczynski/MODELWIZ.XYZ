import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import ExcelJS from "exceljs";

export interface MAAssumptions {
  acquirerName: string;
  targetName: string;
  transactionDate: string;
  
  // Acquirer Financials
  acquirerRevenue: number;
  acquirerRevenueGrowth: number[];
  acquirerEBITDAMargin: number;
  acquirerDAPercent: number;
  acquirerInterestExpense: number;
  acquirerTaxRate: number;
  acquirerSharesOutstanding: number;
  acquirerStockPrice: number;
  
  // Target Financials
  targetRevenue: number;
  targetRevenueGrowth: number[];
  targetEBITDAMargin: number;
  targetDAPercent: number;
  targetInterestExpense: number;
  targetTaxRate: number;
  targetNetDebt: number;
  
  // Transaction Structure
  purchasePrice: number;
  cashPercent: number;
  stockPercent: number;
  premium: number;
  transactionFeePercent: number;
  
  // Financing
  cashFromBalance: number;
  newDebtAmount: number;
  newDebtRate: number;
  
  // Synergies - Revenue (typically slower realization)
  revenueSynergies: number;
  revenueSynergyRealizationY1: number;
  revenueSynergyRealizationY2: number;
  revenueSynergyRealizationY3: number;
  revenueSynergyRealizationY4: number;
  revenueSynergyRealizationY5: number;
  
  // Synergies - Cost (typically faster realization)
  costSynergies: number;
  costSynergyRealizationY1: number;
  costSynergyRealizationY2: number;
  costSynergyRealizationY3: number;
  costSynergyRealizationY4: number;
  costSynergyRealizationY5: number;
  
  // Integration Costs
  integrationCostsY1: number;
  integrationCostsY2: number;
  integrationCostsY3: number;
  
  // Purchase Price Allocation
  intangibleAssets: number;
  intangibleAmortYears: number;
}

const MA_PARSING_PROMPT = `You are a financial analyst expert in M&A transactions. Parse the following natural language description of a merger or acquisition and extract all relevant parameters.

CRITICAL: Revenue synergies and cost synergies often have DIFFERENT phase-in schedules. Cost synergies typically realize FASTER than revenue synergies. Extract separate schedules for each.

Return a JSON object with the following structure:
{
  "acquirerName": "Acquirer Company Name",
  "targetName": "Target Company Name",
  "transactionDate": "YYYY-MM-DD",
  
  "acquirerRevenue": number (in millions),
  "acquirerRevenueGrowth": [y1, y2, y3, y4, y5] (as decimals),
  "acquirerEBITDAMargin": number (as decimal),
  "acquirerDAPercent": number (D&A as % of revenue, as decimal),
  "acquirerInterestExpense": number (in millions),
  "acquirerTaxRate": number (as decimal),
  "acquirerSharesOutstanding": number (in millions),
  "acquirerStockPrice": number ($ per share),
  
  "targetRevenue": number (in millions),
  "targetRevenueGrowth": [y1, y2, y3, y4, y5] (as decimals),
  "targetEBITDAMargin": number (as decimal),
  "targetDAPercent": number (as decimal),
  "targetInterestExpense": number (in millions),
  "targetTaxRate": number (as decimal),
  "targetNetDebt": number (in millions),
  
  "purchasePrice": number (equity value in millions),
  "cashPercent": number (as decimal, e.g., 0.5 for 50% cash),
  "stockPercent": number (as decimal, e.g., 0.5 for 50% stock),
  "premium": number (as decimal, e.g., 0.30 for 30% premium),
  "transactionFeePercent": number (as decimal, e.g., 0.01 for 1% of EV),
  
  "cashFromBalance": number (in millions),
  "newDebtAmount": number (in millions),
  "newDebtRate": number (as decimal),
  
  "revenueSynergies": number (annual run-rate in millions),
  "revenueSynergyRealizationY1": number (as decimal, SLOWER realization typical),
  "revenueSynergyRealizationY2": number (as decimal),
  "revenueSynergyRealizationY3": number (as decimal),
  "revenueSynergyRealizationY4": number (as decimal),
  "revenueSynergyRealizationY5": number (as decimal, typically 1.0),
  
  "costSynergies": number (annual EBITDA improvement in millions),
  "costSynergyRealizationY1": number (as decimal, FASTER realization typical),
  "costSynergyRealizationY2": number (as decimal),
  "costSynergyRealizationY3": number (as decimal),
  "costSynergyRealizationY4": number (as decimal),
  "costSynergyRealizationY5": number (as decimal, typically 1.0),
  
  "integrationCostsY1": number (in millions),
  "integrationCostsY2": number (in millions),
  "integrationCostsY3": number (in millions),
  
  "intangibleAssets": number (in millions),
  "intangibleAmortYears": number
}

DEFAULTS (use if not explicitly stated):
- Revenue growth: 3-8% annually
- EBITDA margins: 15-25%
- D&A: 3-5% of revenue
- Tax rate: 25%
- Cash/stock mix: 50/50 if not specified
- Premium: 20-40% for public targets
- Transaction fees: 1-2% of EV (use 0.02 if not specified)
- REVENUE synergy phase-in (SLOWER): 10% Y1, 25% Y2, 50% Y3, 75% Y4, 100% Y5
- COST synergy phase-in (FASTER): 30% Y1, 60% Y2, 85% Y3, 95% Y4, 100% Y5
- Integration costs: 2-3x annual synergies spread over 3 years
- Intangible amortization: 10-15 years

Look for keywords like "faster", "quicker", "accelerated" for cost synergies vs "gradual", "slower" for revenue synergies. If percentages are explicitly given for each synergy type, use those exact values.

IMPORTANT: Return ONLY the JSON object, no markdown, no explanation.`;

export async function parseMADescription(
  description: string,
  provider: "zhi1" | "zhi2" | "zhi3" | "zhi4" | "zhi5",
  customInstructions?: string
): Promise<{ assumptions: MAAssumptions; providerUsed: string }> {
  const fullPrompt = customInstructions 
    ? `${MA_PARSING_PROMPT}\n\nAdditional Instructions: ${customInstructions}\n\nDescription:\n${description}`
    : `${MA_PARSING_PROMPT}\n\nDescription:\n${description}`;

  let responseText: string = "";
  let providerUsed: string = "";

  if (provider === "zhi1") {
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: fullPrompt }],
      temperature: 0.3,
    });
    responseText = response.choices[0]?.message?.content || "";
    providerUsed = "OpenAI GPT-4";
  } else if (provider === "zhi2") {
    const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    const response = await anthropic.messages.create({
      model: "claude-3-7-sonnet-20250219",
      max_tokens: 4096,
      messages: [{ role: "user", content: fullPrompt }],
    });
    responseText = response.content[0].type === "text" ? response.content[0].text : "";
    providerUsed = "Anthropic Claude";
  } else if (provider === "zhi3") {
    const deepseek = new OpenAI({
      baseURL: "https://api.deepseek.com",
      apiKey: process.env.DEEPSEEK_API_KEY,
    });
    const response = await deepseek.chat.completions.create({
      model: "deepseek-chat",
      messages: [{ role: "user", content: fullPrompt }],
      temperature: 0.3,
    });
    responseText = response.choices[0]?.message?.content || "";
    providerUsed = "DeepSeek";
  } else if (provider === "zhi4") {
    const response = await fetch("https://api.perplexity.ai/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.PERPLEXITY_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "sonar-pro",
        messages: [{ role: "user", content: fullPrompt }],
        temperature: 0.3,
      }),
    });
    const data = await response.json();
    responseText = data.choices?.[0]?.message?.content || "";
    providerUsed = "Perplexity";
  } else if (provider === "zhi5") {
    const grok = new OpenAI({
      baseURL: "https://api.x.ai/v1",
      apiKey: process.env.GROK_API_KEY,
    });
    const response = await grok.chat.completions.create({
      model: "grok-3",
      messages: [{ role: "user", content: fullPrompt }],
      temperature: 0.3,
    });
    responseText = response.choices[0]?.message?.content || "";
    providerUsed = "Grok";
  }

  // Parse JSON from response
  let jsonStr = responseText.trim();
  if (jsonStr.startsWith("```json")) {
    jsonStr = jsonStr.slice(7);
  }
  if (jsonStr.startsWith("```")) {
    jsonStr = jsonStr.slice(3);
  }
  if (jsonStr.endsWith("```")) {
    jsonStr = jsonStr.slice(0, -3);
  }
  jsonStr = jsonStr.trim();

  const assumptions: MAAssumptions = JSON.parse(jsonStr);
  return { assumptions, providerUsed };
}

export function calculateMAMetrics(assumptions: MAAssumptions) {
  const {
    acquirerRevenue,
    acquirerRevenueGrowth,
    acquirerEBITDAMargin,
    acquirerDAPercent,
    acquirerInterestExpense,
    acquirerTaxRate,
    acquirerSharesOutstanding,
    acquirerStockPrice,
    targetRevenue,
    targetRevenueGrowth,
    targetEBITDAMargin,
    targetDAPercent,
    targetInterestExpense,
    targetTaxRate,
    targetNetDebt,
    purchasePrice,
    cashPercent,
    stockPercent,
    transactionFeePercent,
    newDebtAmount,
    newDebtRate,
    revenueSynergies,
    revenueSynergyRealizationY1,
    revenueSynergyRealizationY2,
    revenueSynergyRealizationY3,
    revenueSynergyRealizationY4,
    revenueSynergyRealizationY5,
    costSynergies,
    costSynergyRealizationY1,
    costSynergyRealizationY2,
    costSynergyRealizationY3,
    costSynergyRealizationY4,
    costSynergyRealizationY5,
    integrationCostsY1,
    integrationCostsY2,
    integrationCostsY3,
    intangibleAssets,
    intangibleAmortYears,
  } = assumptions;

  const years = [0, 1, 2, 3, 4, 5];
  
  // Acquirer Standalone Projections
  const acquirerRev: number[] = [acquirerRevenue];
  const acquirerEBITDA: number[] = [acquirerRevenue * acquirerEBITDAMargin];
  const acquirerNetIncome: number[] = [];
  const acquirerEPS: number[] = [];
  
  for (let i = 1; i <= 5; i++) {
    const growth = acquirerRevenueGrowth[i - 1] || 0.05;
    acquirerRev.push(acquirerRev[i - 1] * (1 + growth));
  }
  
  for (let i = 0; i <= 5; i++) {
    const ebitda = acquirerRev[i] * acquirerEBITDAMargin;
    if (i > 0) acquirerEBITDA.push(ebitda);
    const da = acquirerRev[i] * acquirerDAPercent;
    const ebit = ebitda - da;
    const ebt = ebit - acquirerInterestExpense;
    const taxes = Math.max(0, ebt * acquirerTaxRate);
    const netIncome = ebt - taxes;
    acquirerNetIncome.push(netIncome);
    acquirerEPS.push(netIncome / acquirerSharesOutstanding);
  }

  // Target Standalone Projections
  const targetRev: number[] = [targetRevenue];
  const targetEBITDA: number[] = [targetRevenue * targetEBITDAMargin];
  const targetNetIncome: number[] = [];
  
  for (let i = 1; i <= 5; i++) {
    const growth = targetRevenueGrowth[i - 1] || 0.05;
    targetRev.push(targetRev[i - 1] * (1 + growth));
  }
  
  for (let i = 0; i <= 5; i++) {
    const ebitda = targetRev[i] * targetEBITDAMargin;
    if (i > 0) targetEBITDA.push(ebitda);
    const da = targetRev[i] * targetDAPercent;
    const ebit = ebitda - da;
    const ebt = ebit - targetInterestExpense;
    const taxes = Math.max(0, ebt * targetTaxRate);
    targetNetIncome.push(ebt - taxes);
  }

  // Transaction Metrics
  const cashConsideration = purchasePrice * cashPercent;
  const stockConsideration = purchasePrice * stockPercent;
  const newSharesIssued = stockConsideration / acquirerStockPrice;
  const proFormaShares = acquirerSharesOutstanding + newSharesIssued;
  const enterpriseValue = purchasePrice + targetNetDebt;
  const evEbitdaMultiple = enterpriseValue / (targetRevenue * targetEBITDAMargin);
  
  // Goodwill calculation
  const targetBookValue = targetRevenue * 0.3; // Assumed book value
  const excessPurchasePrice = purchasePrice - targetBookValue;
  const goodwill = Math.max(0, excessPurchasePrice - intangibleAssets);
  const annualIntangibleAmort = intangibleAssets / intangibleAmortYears;

  // Synergies by year - SEPARATE schedules for revenue vs cost synergies
  // VALIDATED DEFAULTS from Test #2 (CloudCore/DataVault): Rev 10/25/50/75/100, Cost 30/60/85/95/100
  const revenueSynergyRealization = [
    0, 
    revenueSynergyRealizationY1 || 0.10, 
    revenueSynergyRealizationY2 || 0.25, 
    revenueSynergyRealizationY3 || 0.50, 
    revenueSynergyRealizationY4 || 0.75, 
    revenueSynergyRealizationY5 || 1.0
  ];
  const costSynergyRealization = [
    0, 
    costSynergyRealizationY1 || 0.30, 
    costSynergyRealizationY2 || 0.60, 
    costSynergyRealizationY3 || 0.85, 
    costSynergyRealizationY4 || 0.95, 
    costSynergyRealizationY5 || 1.0
  ];
  const revSynergiesByYear = revenueSynergyRealization.map(r => revenueSynergies * r);
  const costSynergiesByYear = costSynergyRealization.map(r => costSynergies * r);
  const integrationCosts = [0, integrationCostsY1, integrationCostsY2, integrationCostsY3, 0, 0];

  // Pro Forma Combined Projections
  const proFormaRevenue: number[] = [];
  const proFormaEBITDA: number[] = [];
  const proFormaNetIncome: number[] = [];
  const proFormaEPS: number[] = [];
  const accretionDilution: number[] = [];
  const accretionDilutionPercent: number[] = [];

  for (let i = 0; i <= 5; i++) {
    // Revenue
    const combinedRev = acquirerRev[i] + targetRev[i] + revSynergiesByYear[i];
    proFormaRevenue.push(combinedRev);
    
    // EBITDA
    const combinedEBITDA = acquirerEBITDA[i] + targetEBITDA[i] + costSynergiesByYear[i];
    proFormaEBITDA.push(combinedEBITDA);
    
    // D&A (combined + intangible amortization)
    const combinedDA = (acquirerRev[i] * acquirerDAPercent) + (targetRev[i] * targetDAPercent) + (i > 0 ? annualIntangibleAmort : 0);
    
    // EBIT
    const ebit = combinedEBITDA - combinedDA;
    
    // Interest (existing + new debt)
    const combinedInterest = acquirerInterestExpense + targetInterestExpense + (newDebtAmount * newDebtRate);
    
    // EBT
    const ebt = ebit - combinedInterest;
    
    // Taxes
    const taxes = Math.max(0, ebt * acquirerTaxRate);
    
    // Net Income (less integration costs after-tax)
    const integrationAfterTax = integrationCosts[i] * (1 - acquirerTaxRate);
    const netIncome = ebt - taxes - integrationAfterTax;
    proFormaNetIncome.push(netIncome);
    
    // EPS
    const eps = netIncome / proFormaShares;
    proFormaEPS.push(eps);
    
    // Accretion/Dilution
    const epsImpact = eps - acquirerEPS[i];
    accretionDilution.push(epsImpact);
    accretionDilutionPercent.push(acquirerEPS[i] !== 0 ? (eps / acquirerEPS[i]) - 1 : 0);
  }

  // Sources and Uses
  const feePercent = transactionFeePercent || 0.02;
  const transactionFees = enterpriseValue * feePercent;
  
  const sources = {
    cashFromBalance: assumptions.cashFromBalance,
    newDebt: newDebtAmount,
    stockConsideration,
    total: assumptions.cashFromBalance + newDebtAmount + stockConsideration,
  };
  
  const uses = {
    equityValue: purchasePrice,
    netDebtAssumed: targetNetDebt,
    transactionFees,
    total: purchasePrice + targetNetDebt + transactionFees,
  };

  return {
    assumptions,
    acquirerProjections: {
      years,
      revenue: acquirerRev,
      ebitda: acquirerEBITDA,
      netIncome: acquirerNetIncome,
      eps: acquirerEPS,
    },
    targetProjections: {
      years,
      revenue: targetRev,
      ebitda: targetEBITDA,
      netIncome: targetNetIncome,
    },
    transactionMetrics: {
      purchasePrice,
      enterpriseValue,
      evEbitdaMultiple,
      cashConsideration,
      stockConsideration,
      newSharesIssued,
      proFormaShares,
      goodwill,
      intangibleAssets,
    },
    synergies: {
      revenueSynergies,
      costSynergies,
      totalSynergies: revenueSynergies + costSynergies,
      revSynergiesByYear,
      costSynergiesByYear,
      revenueSynergyRealization,
      costSynergyRealization,
      integrationCosts,
    },
    sourcesAndUses: {
      sources,
      uses,
    },
    proFormaProjections: {
      years,
      revenue: proFormaRevenue,
      ebitda: proFormaEBITDA,
      netIncome: proFormaNetIncome,
      eps: proFormaEPS,
    },
    accretionDilution: {
      epsImpact: accretionDilution,
      percentImpact: accretionDilutionPercent,
      isAccretiveY1: accretionDilutionPercent[1] > 0,
      isAccretiveY2: accretionDilutionPercent[2] > 0,
      isAccretiveY3: accretionDilutionPercent[3] > 0,
    },
  };
}

export async function generateMAExcel(assumptions: MAAssumptions): Promise<Buffer> {
  const workbook = new ExcelJS.Workbook();
  workbook.creator = "Finance Panel";
  workbook.created = new Date();

  const results = calculateMAMetrics(assumptions);
  const { acquirerProjections, targetProjections, transactionMetrics, synergies, sourcesAndUses, proFormaProjections, accretionDilution } = results;

  const currencyFormat = '"$"#,##0';
  const percentFormat = "0.0%";
  const multipleFormat = "0.0x";
  const epsFormat = '"$"0.00';

  // ============ EXECUTIVE SUMMARY ============
  const summarySheet = workbook.addWorksheet("Executive Summary");
  summarySheet.columns = [{ width: 35 }, { width: 20 }, { width: 20 }];

  summarySheet.getCell("A1").value = `${assumptions.acquirerName} Acquisition of ${assumptions.targetName}`;
  summarySheet.getCell("A1").font = { bold: true, size: 16 };
  summarySheet.getCell("A2").value = `Transaction Date: ${assumptions.transactionDate || new Date().toLocaleDateString()}`;

  summarySheet.getCell("A4").value = "TRANSACTION OVERVIEW";
  summarySheet.getCell("A4").font = { bold: true, size: 14 };

  summarySheet.getCell("A5").value = "Purchase Price (Equity):";
  summarySheet.getCell("B5").value = transactionMetrics.purchasePrice;
  summarySheet.getCell("B5").numFmt = currencyFormat;

  summarySheet.getCell("A6").value = "Enterprise Value:";
  summarySheet.getCell("B6").value = transactionMetrics.enterpriseValue;
  summarySheet.getCell("B6").numFmt = currencyFormat;

  summarySheet.getCell("A7").value = "EV/EBITDA Multiple:";
  summarySheet.getCell("B7").value = transactionMetrics.evEbitdaMultiple;
  summarySheet.getCell("B7").numFmt = multipleFormat;

  summarySheet.getCell("A8").value = "Premium Paid:";
  summarySheet.getCell("B8").value = assumptions.premium;
  summarySheet.getCell("B8").numFmt = percentFormat;

  summarySheet.getCell("A10").value = "CONSIDERATION MIX";
  summarySheet.getCell("A10").font = { bold: true, size: 14 };

  summarySheet.getCell("A11").value = "Cash:";
  summarySheet.getCell("B11").value = transactionMetrics.cashConsideration;
  summarySheet.getCell("B11").numFmt = currencyFormat;
  summarySheet.getCell("C11").value = assumptions.cashPercent;
  summarySheet.getCell("C11").numFmt = percentFormat;

  summarySheet.getCell("A12").value = "Stock:";
  summarySheet.getCell("B12").value = transactionMetrics.stockConsideration;
  summarySheet.getCell("B12").numFmt = currencyFormat;
  summarySheet.getCell("C12").value = assumptions.stockPercent;
  summarySheet.getCell("C12").numFmt = percentFormat;

  summarySheet.getCell("A14").value = "SYNERGIES";
  summarySheet.getCell("A14").font = { bold: true, size: 14 };

  summarySheet.getCell("A15").value = "Revenue Synergies (Run-Rate):";
  summarySheet.getCell("B15").value = synergies.revenueSynergies;
  summarySheet.getCell("B15").numFmt = currencyFormat;

  summarySheet.getCell("A16").value = "Cost Synergies (Run-Rate):";
  summarySheet.getCell("B16").value = synergies.costSynergies;
  summarySheet.getCell("B16").numFmt = currencyFormat;

  summarySheet.getCell("A17").value = "Total Synergies:";
  summarySheet.getCell("B17").value = synergies.totalSynergies;
  summarySheet.getCell("B17").numFmt = currencyFormat;
  summarySheet.getRow(17).font = { bold: true };

  summarySheet.getCell("A19").value = "ACCRETION/DILUTION";
  summarySheet.getCell("A19").font = { bold: true, size: 14 };

  summarySheet.getCell("A20").value = "Year 1 EPS Impact:";
  summarySheet.getCell("B20").value = accretionDilution.percentImpact[1];
  summarySheet.getCell("B20").numFmt = percentFormat;
  summarySheet.getCell("C20").value = accretionDilution.isAccretiveY1 ? "Accretive" : "Dilutive";
  summarySheet.getCell("C20").font = { color: { argb: accretionDilution.isAccretiveY1 ? "FF008000" : "FFFF0000" } };

  summarySheet.getCell("A21").value = "Year 2 EPS Impact:";
  summarySheet.getCell("B21").value = accretionDilution.percentImpact[2];
  summarySheet.getCell("B21").numFmt = percentFormat;
  summarySheet.getCell("C21").value = accretionDilution.isAccretiveY2 ? "Accretive" : "Dilutive";
  summarySheet.getCell("C21").font = { color: { argb: accretionDilution.isAccretiveY2 ? "FF008000" : "FFFF0000" } };

  summarySheet.getCell("A22").value = "Year 3 EPS Impact:";
  summarySheet.getCell("B22").value = accretionDilution.percentImpact[3];
  summarySheet.getCell("B22").numFmt = percentFormat;
  summarySheet.getCell("C22").value = accretionDilution.isAccretiveY3 ? "Accretive" : "Dilutive";
  summarySheet.getCell("C22").font = { color: { argb: accretionDilution.isAccretiveY3 ? "FF008000" : "FFFF0000" } };

  // ============ SOURCES & USES ============
  const suSheet = workbook.addWorksheet("Sources_Uses");
  suSheet.columns = [{ width: 35 }, { width: 18 }, { width: 18 }];

  suSheet.getCell("A1").value = "SOURCES & USES OF FUNDS";
  suSheet.getCell("A1").font = { bold: true, size: 14 };

  suSheet.getCell("A3").value = "SOURCES";
  suSheet.getCell("B3").value = "Amount ($M)";
  suSheet.getCell("C3").value = "% of Total";
  suSheet.getRow(3).font = { bold: true };

  suSheet.getCell("A4").value = "Cash from Balance Sheet";
  suSheet.getCell("B4").value = sourcesAndUses.sources.cashFromBalance;
  suSheet.getCell("B4").numFmt = currencyFormat;
  suSheet.getCell("C4").value = sourcesAndUses.sources.cashFromBalance / sourcesAndUses.sources.total;
  suSheet.getCell("C4").numFmt = percentFormat;

  suSheet.getCell("A5").value = "New Debt Issuance";
  suSheet.getCell("B5").value = sourcesAndUses.sources.newDebt;
  suSheet.getCell("B5").numFmt = currencyFormat;
  suSheet.getCell("C5").value = sourcesAndUses.sources.newDebt / sourcesAndUses.sources.total;
  suSheet.getCell("C5").numFmt = percentFormat;

  suSheet.getCell("A6").value = "Stock Consideration";
  suSheet.getCell("B6").value = sourcesAndUses.sources.stockConsideration;
  suSheet.getCell("B6").numFmt = currencyFormat;
  suSheet.getCell("C6").value = sourcesAndUses.sources.stockConsideration / sourcesAndUses.sources.total;
  suSheet.getCell("C6").numFmt = percentFormat;

  suSheet.getCell("A7").value = "TOTAL SOURCES";
  suSheet.getCell("B7").value = sourcesAndUses.sources.total;
  suSheet.getCell("B7").numFmt = currencyFormat;
  suSheet.getRow(7).font = { bold: true };

  suSheet.getCell("A9").value = "USES";
  suSheet.getRow(9).font = { bold: true };

  suSheet.getCell("A10").value = "Target Equity Value";
  suSheet.getCell("B10").value = sourcesAndUses.uses.equityValue;
  suSheet.getCell("B10").numFmt = currencyFormat;

  suSheet.getCell("A11").value = "Net Debt Assumed";
  suSheet.getCell("B11").value = sourcesAndUses.uses.netDebtAssumed;
  suSheet.getCell("B11").numFmt = currencyFormat;

  suSheet.getCell("A12").value = "Transaction Fees";
  suSheet.getCell("B12").value = sourcesAndUses.uses.transactionFees;
  suSheet.getCell("B12").numFmt = currencyFormat;

  suSheet.getCell("A13").value = "TOTAL USES";
  suSheet.getCell("B13").value = sourcesAndUses.uses.total;
  suSheet.getCell("B13").numFmt = currencyFormat;
  suSheet.getRow(13).font = { bold: true };

  // ============ ACQUIRER PROJECTIONS ============
  const acqSheet = workbook.addWorksheet("Acquirer_Standalone");
  acqSheet.columns = [
    { width: 25 },
    { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 },
  ];

  acqSheet.getCell("A1").value = `${assumptions.acquirerName} - Standalone Projections`;
  acqSheet.getCell("A1").font = { bold: true, size: 14 };

  acqSheet.addRow(["", "Year 0", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]);
  acqSheet.getRow(2).font = { bold: true };
  acqSheet.getRow(2).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFE0E0E0" } };

  acqSheet.addRow(["Revenue ($M)", ...acquirerProjections.revenue]);
  for (let i = 2; i <= 7; i++) acqSheet.getCell(3, i).numFmt = currencyFormat;

  acqSheet.addRow(["EBITDA ($M)", ...acquirerProjections.ebitda]);
  for (let i = 2; i <= 7; i++) acqSheet.getCell(4, i).numFmt = currencyFormat;

  acqSheet.addRow(["Net Income ($M)", ...acquirerProjections.netIncome]);
  for (let i = 2; i <= 7; i++) acqSheet.getCell(5, i).numFmt = currencyFormat;

  acqSheet.addRow(["EPS", ...acquirerProjections.eps]);
  for (let i = 2; i <= 7; i++) acqSheet.getCell(6, i).numFmt = epsFormat;

  // ============ TARGET PROJECTIONS ============
  const tgtSheet = workbook.addWorksheet("Target_Standalone");
  tgtSheet.columns = [
    { width: 25 },
    { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 },
  ];

  tgtSheet.getCell("A1").value = `${assumptions.targetName} - Standalone Projections`;
  tgtSheet.getCell("A1").font = { bold: true, size: 14 };

  tgtSheet.addRow(["", "Year 0", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]);
  tgtSheet.getRow(2).font = { bold: true };
  tgtSheet.getRow(2).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFE0E0E0" } };

  tgtSheet.addRow(["Revenue ($M)", ...targetProjections.revenue]);
  for (let i = 2; i <= 7; i++) tgtSheet.getCell(3, i).numFmt = currencyFormat;

  tgtSheet.addRow(["EBITDA ($M)", ...targetProjections.ebitda]);
  for (let i = 2; i <= 7; i++) tgtSheet.getCell(4, i).numFmt = currencyFormat;

  tgtSheet.addRow(["Net Income ($M)", ...targetProjections.netIncome]);
  for (let i = 2; i <= 7; i++) tgtSheet.getCell(5, i).numFmt = currencyFormat;

  // ============ SYNERGIES ============
  const synSheet = workbook.addWorksheet("Synergies");
  synSheet.columns = [
    { width: 30 },
    { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 },
  ];

  synSheet.getCell("A1").value = "SYNERGY SCHEDULE";
  synSheet.getCell("A1").font = { bold: true, size: 14 };

  synSheet.addRow(["", "Year 0", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]);
  synSheet.getRow(2).font = { bold: true };

  synSheet.addRow(["Revenue Synergies ($M)", ...synergies.revSynergiesByYear]);
  for (let i = 2; i <= 7; i++) synSheet.getCell(3, i).numFmt = currencyFormat;

  synSheet.addRow(["Cost Synergies ($M)", ...synergies.costSynergiesByYear]);
  for (let i = 2; i <= 7; i++) synSheet.getCell(4, i).numFmt = currencyFormat;

  synSheet.addRow(["Integration Costs ($M)", ...synergies.integrationCosts]);
  for (let i = 2; i <= 7; i++) synSheet.getCell(5, i).numFmt = currencyFormat;

  // ============ PRO FORMA ============
  const pfSheet = workbook.addWorksheet("Pro_Forma_Combined");
  pfSheet.columns = [
    { width: 25 },
    { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 },
  ];

  pfSheet.getCell("A1").value = "PRO FORMA COMBINED FINANCIALS";
  pfSheet.getCell("A1").font = { bold: true, size: 14 };

  pfSheet.addRow(["", "Year 0", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]);
  pfSheet.getRow(2).font = { bold: true };
  pfSheet.getRow(2).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFE0E0E0" } };

  pfSheet.addRow(["Revenue ($M)", ...proFormaProjections.revenue]);
  for (let i = 2; i <= 7; i++) pfSheet.getCell(3, i).numFmt = currencyFormat;

  pfSheet.addRow(["EBITDA ($M)", ...proFormaProjections.ebitda]);
  for (let i = 2; i <= 7; i++) pfSheet.getCell(4, i).numFmt = currencyFormat;

  pfSheet.addRow(["Net Income ($M)", ...proFormaProjections.netIncome]);
  for (let i = 2; i <= 7; i++) pfSheet.getCell(5, i).numFmt = currencyFormat;

  pfSheet.addRow(["Pro Forma EPS", ...proFormaProjections.eps]);
  for (let i = 2; i <= 7; i++) pfSheet.getCell(6, i).numFmt = epsFormat;

  // ============ ACCRETION/DILUTION ============
  const adSheet = workbook.addWorksheet("Accretion_Dilution");
  adSheet.columns = [
    { width: 30 },
    { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 },
  ];

  adSheet.getCell("A1").value = "ACCRETION / DILUTION ANALYSIS";
  adSheet.getCell("A1").font = { bold: true, size: 14 };

  adSheet.addRow(["", "Year 0", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]);
  adSheet.getRow(2).font = { bold: true };

  adSheet.addRow(["Acquirer Standalone EPS", ...acquirerProjections.eps]);
  for (let i = 2; i <= 7; i++) adSheet.getCell(3, i).numFmt = epsFormat;

  adSheet.addRow(["Pro Forma EPS", ...proFormaProjections.eps]);
  for (let i = 2; i <= 7; i++) adSheet.getCell(4, i).numFmt = epsFormat;

  adSheet.addRow(["EPS Impact ($)", ...accretionDilution.epsImpact]);
  for (let i = 2; i <= 7; i++) adSheet.getCell(5, i).numFmt = epsFormat;

  adSheet.addRow(["EPS Impact (%)", ...accretionDilution.percentImpact]);
  for (let i = 2; i <= 7; i++) {
    adSheet.getCell(6, i).numFmt = percentFormat;
    const val = accretionDilution.percentImpact[i - 2];
    if (val > 0) {
      adSheet.getCell(6, i).font = { color: { argb: "FF008000" } };
    } else if (val < 0) {
      adSheet.getCell(6, i).font = { color: { argb: "FFFF0000" } };
    }
  }

  const buffer = await workbook.xlsx.writeBuffer();
  return Buffer.from(buffer);
}
