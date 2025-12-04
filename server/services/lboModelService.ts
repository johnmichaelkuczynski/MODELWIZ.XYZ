import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import ExcelJS from "exceljs";

export interface LBOAssumptions {
  companyName: string;
  transactionDate: string;
  
  // Target Company Financials
  baseYearRevenue: number;
  revenueGrowthRates: number[];
  baseEBITDAMargin: number;
  targetEBITDAMargin: number;
  marginExpansionYears: number;
  daPercent: number;
  capexPercent: number;
  nwcPercent: number;
  taxRate: number;
  
  // Purchase Price
  purchasePrice: number;
  entryMultiple: number;
  transactionCosts: number;
  financingFees: number;
  managementRollover: number;
  
  // Financing Structure
  seniorDebtAmount: number;
  seniorDebtRate: number;
  seniorDebtAmortization: number;
  subDebtAmount: number;
  subDebtRate: number;
  subDebtPIK: number;
  revolverSize: number;
  revolverRate: number;
  sponsorEquity: number;
  cashFlowSweepPercent: number;
  
  // Exit Assumptions
  exitYear: number;
  exitMultiple: number;
  exitCosts: number;
  
  // Management Fee
  managementFeePercent: number;
}

const LBO_PARSING_PROMPT = `You are a financial analyst expert. Parse the following natural language description of an LBO (Leveraged Buyout) transaction and extract all relevant parameters.

Return a JSON object with the following structure:
{
  "companyName": "Target Company Name",
  "transactionDate": "YYYY-MM-DD",
  
  "baseYearRevenue": number (in millions),
  "revenueGrowthRates": [year1, year2, year3, year4, year5] (as decimals, e.g., 0.10 for 10%),
  "baseEBITDAMargin": number (as decimal, e.g., 0.20 for 20%),
  "targetEBITDAMargin": number (as decimal),
  "marginExpansionYears": number (years to reach target margin),
  "daPercent": number (D&A as % of revenue, as decimal),
  "capexPercent": number (CapEx as % of revenue, as decimal),
  "nwcPercent": number (NWC as % of revenue, as decimal),
  "taxRate": number (as decimal, e.g., 0.25 for 25%),
  
  "purchasePrice": number (in millions),
  "entryMultiple": number (e.g., 10.5 for 10.5x EBITDA),
  "transactionCosts": number (as decimal, e.g., 0.02 for 2% of purchase price),
  "financingFees": number (as decimal, e.g., 0.01 for 1% of total debt),
  "managementRollover": number (in millions),
  
  "seniorDebtAmount": number (in millions),
  "seniorDebtRate": number (as decimal, e.g., 0.06 for 6%),
  "seniorDebtAmortization": number (annual mandatory repayment as decimal of original),
  "subDebtAmount": number (in millions),
  "subDebtRate": number (as decimal),
  "subDebtPIK": number (PIK interest as decimal, 0 if cash pay),
  "revolverSize": number (in millions),
  "revolverRate": number (as decimal),
  "sponsorEquity": number (in millions - will be recalculated to balance sources=uses),
  "cashFlowSweepPercent": number (as decimal, e.g., 0.75 for 75% FCF sweep to debt),
  
  "exitYear": number (typically 5),
  "exitMultiple": number,
  "exitCosts": number (as decimal),
  
  "managementFeePercent": number (as decimal)
}

If any value is not explicitly stated, use reasonable LBO industry defaults:
- Revenue growth: 5-10% annually
- EBITDA margin: 15-25%
- Target margin expansion: 2-5% over 5 years
- D&A: 3-5% of revenue
- CapEx: 3-5% of revenue
- NWC: 10-15% of revenue
- Tax rate: 25%
- Entry multiple: 8-12x EBITDA
- Transaction costs: 2-3% of purchase price
- Financing fees: 1% of total debt (default 0.01)
- Senior debt: 4-5x EBITDA at 5-7%
- Subordinated debt: 1-2x EBITDA at 10-12%
- Cash flow sweep: 75% of FCF to debt paydown (default 0.75)
- Exit multiple: Similar to entry or slight expansion
- Exit year: 5 years
- Management fee: 1-2% of EBITDA

IMPORTANT: Return ONLY the JSON object, no markdown, no explanation.`;

export async function parseLBODescription(
  description: string,
  provider: "zhi1" | "zhi2" | "zhi3" | "zhi4" | "zhi5",
  customInstructions?: string
): Promise<{ assumptions: LBOAssumptions; providerUsed: string }> {
  const fullPrompt = customInstructions 
    ? `${LBO_PARSING_PROMPT}\n\nAdditional Instructions: ${customInstructions}\n\nDescription:\n${description}`
    : `${LBO_PARSING_PROMPT}\n\nDescription:\n${description}`;

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
    providerUsed = "ZHI 1";
  } else if (provider === "zhi2") {
    const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    const response = await anthropic.messages.create({
      model: "claude-3-7-sonnet-20250219",
      max_tokens: 4096,
      messages: [{ role: "user", content: fullPrompt }],
    });
    responseText = response.content[0].type === "text" ? response.content[0].text : "";
    providerUsed = "ZHI 2";
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
    providerUsed = "ZHI 3";
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
    providerUsed = "ZHI 4";
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
    providerUsed = "ZHI 5";
  }

  // Parse JSON from response - handle various response formats including conversational text
  let jsonStr = responseText.trim();
  
  // First try: extract JSON from markdown code blocks
  if (jsonStr.includes("```json")) {
    const match = jsonStr.match(/```json\s*([\s\S]*?)\s*```/);
    if (match) {
      jsonStr = match[1].trim();
    }
  } else if (jsonStr.includes("```")) {
    const match = jsonStr.match(/```\s*([\s\S]*?)\s*```/);
    if (match) {
      jsonStr = match[1].trim();
    }
  }
  
  // Second try: find JSON object by looking for opening/closing braces
  if (!jsonStr.startsWith("{")) {
    const startIdx = jsonStr.indexOf("{");
    const endIdx = jsonStr.lastIndexOf("}");
    if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
      jsonStr = jsonStr.slice(startIdx, endIdx + 1);
    }
  }
  
  jsonStr = jsonStr.trim();

  const assumptions: LBOAssumptions = JSON.parse(jsonStr);
  return { assumptions, providerUsed };
}

export function calculateLBOReturns(assumptions: LBOAssumptions) {
  const {
    baseYearRevenue,
    revenueGrowthRates,
    baseEBITDAMargin,
    targetEBITDAMargin,
    marginExpansionYears,
    daPercent,
    capexPercent,
    nwcPercent,
    taxRate,
    purchasePrice,
    entryMultiple,
    transactionCosts,
    financingFees = 0.01,
    managementRollover,
    seniorDebtAmount,
    seniorDebtRate,
    seniorDebtAmortization,
    subDebtAmount,
    subDebtRate,
    subDebtPIK,
    cashFlowSweepPercent = 0.75,
    exitYear,
    exitMultiple,
    exitCosts,
    managementFeePercent,
  } = assumptions;
  
  // ============ SOURCES & USES - PROPERLY BALANCED ============
  // Uses: Purchase Price + Transaction Costs + Financing Fees
  const transactionCostsAmount = purchasePrice * transactionCosts;
  const totalDebt = seniorDebtAmount + subDebtAmount;
  const financingFeesAmount = totalDebt * financingFees;
  const totalUses = purchasePrice + transactionCostsAmount + financingFeesAmount;
  
  // Sources: Debt + Management Rollover + Sponsor Equity (balancing item)
  // Sponsor Equity = Total Uses - Debt - Management Rollover
  const sponsorEquity = totalUses - seniorDebtAmount - subDebtAmount - managementRollover;
  const totalSources = seniorDebtAmount + subDebtAmount + sponsorEquity + managementRollover;
  
  console.log(`[LBO Model] Sources & Uses Balance Check:`);
  console.log(`  Uses: Purchase=${purchasePrice.toFixed(2)}M + TxnCosts=${transactionCostsAmount.toFixed(2)}M + FinFees=${financingFeesAmount.toFixed(2)}M = ${totalUses.toFixed(2)}M`);
  console.log(`  Sources: Senior=${seniorDebtAmount.toFixed(2)}M + Sub=${subDebtAmount.toFixed(2)}M + Sponsor=${sponsorEquity.toFixed(2)}M + Rollover=${managementRollover.toFixed(2)}M = ${totalSources.toFixed(2)}M`);

  // Calculate projections
  const years = [0, 1, 2, 3, 4, 5];
  const revenue: number[] = [baseYearRevenue];
  const ebitdaMargins: number[] = [baseEBITDAMargin];
  const ebitda: number[] = [baseYearRevenue * baseEBITDAMargin];
  
  for (let i = 1; i <= 5; i++) {
    const growthRate = revenueGrowthRates[i - 1] || 0.05;
    revenue.push(revenue[i - 1] * (1 + growthRate));
    
    const marginStep = (targetEBITDAMargin - baseEBITDAMargin) / marginExpansionYears;
    const margin = Math.min(baseEBITDAMargin + marginStep * i, targetEBITDAMargin);
    ebitdaMargins.push(margin);
    ebitda.push(revenue[i] * margin);
  }

  // ============ DEBT SCHEDULE - FCF SWEEP WITH CONFIGURABLE % ============
  // Logic: cashFlowSweepPercent (default 75%) of FCF goes to Senior Debt first, then Sub Debt
  // Sweep_Cash = FCF * sweepPercent
  // Senior_Paydown = min(Sweep_Cash, Senior_Balance)
  // Sub_Paydown = min(Sweep_Cash - Senior_Paydown, Sub_Balance + PIK)
  
  const seniorDebt: number[] = [seniorDebtAmount];
  const subDebt: number[] = [subDebtAmount];
  const seniorInterest: number[] = [0]; // Year 0 has no interest
  const subInterest: number[] = [0]; // Year 0 has no interest
  const seniorAmort: number[] = [0]; // Year 0 has no amortization
  const subAmort: number[] = [0]; // Track sub debt amortization
  const freeCashFlow: number[] = [0]; // Year 0 FCF
  const cashSweep: number[] = [0]; // Track cash sweep amounts
  const da: number[] = [baseYearRevenue * daPercent]; // D&A by year
  const capex: number[] = [baseYearRevenue * capexPercent]; // CapEx by year
  const nwc: number[] = [baseYearRevenue * nwcPercent]; // NWC balance by year
  const nwcChange: number[] = [0]; // NWC change by year
  const ebit: number[] = [ebitda[0] - da[0]]; // EBIT by year
  const taxes: number[] = [0]; // Taxes by year
  const netIncome: number[] = [0]; // Net income by year
  
  const sweepPercent = cashFlowSweepPercent;
  console.log(`[LBO Model] Initial debt structure: Senior=${seniorDebtAmount.toFixed(2)}M, Sub=${subDebtAmount.toFixed(2)}M, FCF Sweep=${(sweepPercent * 100).toFixed(0)}%`);
  
  for (let i = 1; i <= 5; i++) {
    // ========== STEP 1: Calculate operating metrics ==========
    da.push(revenue[i] * daPercent);
    ebit.push(ebitda[i] - da[i]);
    
    // Interest on BEGINNING balance
    const seniorInt = seniorDebt[i - 1] * seniorDebtRate;
    const subInt = subDebt[i - 1] * subDebtRate;
    seniorInterest.push(seniorInt);
    subInterest.push(subInt);
    
    const totalInterest = seniorInt + subInt;
    const mgmtFee = ebitda[i] * managementFeePercent;
    const ebt = ebit[i] - totalInterest - mgmtFee;
    const taxAmount = Math.max(0, ebt * taxRate);
    taxes.push(taxAmount);
    netIncome.push(ebt - taxAmount);
    
    // ========== STEP 2: Calculate Free Cash Flow ==========
    capex.push(revenue[i] * capexPercent);
    nwc.push(revenue[i] * nwcPercent);
    const nwcDelta = nwc[i] - nwc[i - 1];
    nwcChange.push(nwcDelta);
    
    // FCF = Net Income + D&A - CapEx - ΔNWC
    // (Interest is already deducted in Net Income)
    const ecf = netIncome[i] + da[i] - capex[i] - nwcDelta;
    freeCashFlow.push(ecf);
    
    // ========== STEP 3: Debt sweep - CONFIGURABLE % OF FCF ==========
    // Only sweepPercent (e.g., 75%) of positive FCF goes to debt paydown
    // Senior first, then Sub (with PIK accrual)
    
    const seniorBeginning = seniorDebt[i - 1];
    const subBeginning = subDebt[i - 1] + (subDebt[i - 1] * subDebtPIK); // Add PIK if any
    
    // Calculate sweep amount = sweepPercent of positive FCF
    const sweepCash = Math.max(0, ecf) * sweepPercent;
    
    // Senior paydown = min(Sweep Cash, Senior Beginning Balance)
    const seniorPaydown = Math.min(sweepCash, seniorBeginning);
    const seniorEnding = seniorBeginning - seniorPaydown;
    
    // Cash remaining after Senior paydown
    const cashAfterSenior = sweepCash - seniorPaydown;
    
    // Sub paydown = min(Remaining Sweep Cash, Sub Beginning Balance)
    const subPaydown = Math.min(cashAfterSenior, subBeginning);
    const subEnding = subBeginning - subPaydown;
    
    // Push results
    seniorAmort.push(seniorPaydown);
    seniorDebt.push(Math.max(0, seniorEnding));
    subAmort.push(subPaydown);
    subDebt.push(Math.max(0, subEnding));
    cashSweep.push(seniorPaydown + subPaydown);
    
    console.log(`[LBO Model] Year ${i}: FCF=${ecf.toFixed(2)}M, Sweep(${(sweepPercent*100).toFixed(0)}%)=${sweepCash.toFixed(2)}M | Senior: Beg=${seniorBeginning.toFixed(2)}M, Paydown=${seniorPaydown.toFixed(2)}M, End=${seniorEnding.toFixed(2)}M | Sub: Beg=${subBeginning.toFixed(2)}M, Paydown=${subPaydown.toFixed(2)}M, End=${subEnding.toFixed(2)}M`);
  }

  // Exit valuation
  const exitEBITDA = ebitda[exitYear];
  const exitEV = exitEBITDA * exitMultiple;
  const remainingDebt = seniorDebt[exitYear] + subDebt[exitYear];
  const exitTransactionCosts = exitEV * exitCosts;
  const exitEquityValue = exitEV - remainingDebt - exitTransactionCosts;

  // Returns calculation
  const sponsorOwnership = sponsorEquity / (sponsorEquity + managementRollover);
  const managementOwnership = 1 - sponsorOwnership;
  
  const sponsorExitProceeds = exitEquityValue * sponsorOwnership;
  const managementExitProceeds = exitEquityValue * managementOwnership;
  
  const sponsorMOIC = sponsorExitProceeds / sponsorEquity;
  const managementMOIC = managementRollover > 0 ? managementExitProceeds / managementRollover : 0;
  
  // IRR calculation (simplified - assumes single exit cash flow)
  const sponsorIRR = Math.pow(sponsorMOIC, 1 / exitYear) - 1;
  const managementIRR = managementRollover > 0 ? Math.pow(managementMOIC, 1 / exitYear) - 1 : 0;

  // Entry and Exit leverage
  const entryLeverage = (seniorDebtAmount + subDebtAmount) / ebitda[0];
  const exitLeverage = remainingDebt / exitEBITDA;
  const debtPaydown = (seniorDebtAmount + subDebtAmount) - remainingDebt;
  const debtPaydownPercent = debtPaydown / (seniorDebtAmount + subDebtAmount);

  return {
    assumptions,
    projections: {
      years,
      revenue,
      ebitda,
      ebitdaMargins,
      da,
      ebit,
      taxes,
      netIncome,
      capex,
      nwc,
      nwcChange,
      freeCashFlow,
      seniorDebt,
      subDebt,
      seniorInterest,
      subInterest,
      seniorAmort,
      subAmort,
      cashSweep,
    },
    sourcesAndUses: {
      sources: {
        seniorDebt: seniorDebtAmount,
        subDebt: subDebtAmount,
        sponsorEquity,
        managementRollover,
        total: totalSources,
      },
      uses: {
        purchasePrice,
        transactionCosts: transactionCostsAmount,
        financingFees: financingFeesAmount,
        total: totalUses,
      },
    },
    exitValuation: {
      exitYear,
      exitEBITDA,
      exitMultiple,
      exitEV,
      remainingDebt,
      exitTransactionCosts,
      exitEquityValue,
    },
    returns: {
      sponsor: {
        equity: sponsorEquity,
        ownership: sponsorOwnership,
        exitProceeds: sponsorExitProceeds,
        moic: sponsorMOIC,
        irr: sponsorIRR,
      },
      management: {
        equity: managementRollover,
        ownership: managementOwnership,
        exitProceeds: managementExitProceeds,
        moic: managementMOIC,
        irr: managementIRR,
      },
    },
    keyMetrics: {
      entryMultiple,
      exitMultiple,
      entryLeverage,
      exitLeverage,
      debtPaydown,
      debtPaydownPercent,
    },
  };
}

export async function generateLBOExcel(assumptions: LBOAssumptions): Promise<Buffer> {
  const workbook = new ExcelJS.Workbook();
  workbook.creator = "Finance Panel";
  workbook.created = new Date();

  const results = calculateLBOReturns(assumptions);
  const { projections, sourcesAndUses, exitValuation, returns, keyMetrics } = results;

  const blueFont: Partial<ExcelJS.Font> = { color: { argb: "FF0000FF" }, bold: true };
  const currencyFormat = '"$"#,##0';
  const percentFormat = "0.0%";
  const multipleFormat = "0.0x";

  // ============ EXECUTIVE SUMMARY ============
  const summarySheet = workbook.addWorksheet("Executive Summary");
  summarySheet.columns = [
    { width: 35 },
    { width: 20 },
    { width: 20 },
  ];

  summarySheet.getCell("A1").value = `${assumptions.companyName} - LBO Model`;
  summarySheet.getCell("A1").font = { bold: true, size: 16 };
  summarySheet.getCell("A2").value = `Transaction Date: ${assumptions.transactionDate || new Date().toLocaleDateString()}`;

  summarySheet.getCell("A4").value = "DEAL OVERVIEW";
  summarySheet.getCell("A4").font = { bold: true, size: 14 };

  summarySheet.getCell("A5").value = "Purchase Price:";
  summarySheet.getCell("B5").value = assumptions.purchasePrice;
  summarySheet.getCell("B5").numFmt = currencyFormat;

  summarySheet.getCell("A6").value = "Entry EBITDA Multiple:";
  summarySheet.getCell("B6").value = assumptions.entryMultiple;
  summarySheet.getCell("B6").numFmt = multipleFormat;

  summarySheet.getCell("A7").value = "Total Debt:";
  summarySheet.getCell("B7").value = assumptions.seniorDebtAmount + assumptions.subDebtAmount;
  summarySheet.getCell("B7").numFmt = currencyFormat;

  summarySheet.getCell("A8").value = "Sponsor Equity:";
  summarySheet.getCell("B8").value = sourcesAndUses.sources.sponsorEquity;
  summarySheet.getCell("B8").numFmt = currencyFormat;

  summarySheet.getCell("A10").value = "RETURNS SUMMARY";
  summarySheet.getCell("A10").font = { bold: true, size: 14 };

  summarySheet.getCell("A11").value = "Sponsor IRR:";
  summarySheet.getCell("B11").value = returns.sponsor.irr;
  summarySheet.getCell("B11").numFmt = percentFormat;
  summarySheet.getCell("B11").font = { bold: true };

  summarySheet.getCell("A12").value = "Sponsor MOIC:";
  summarySheet.getCell("B12").value = returns.sponsor.moic;
  summarySheet.getCell("B12").numFmt = multipleFormat;
  summarySheet.getCell("B12").font = { bold: true };

  summarySheet.getCell("A13").value = "Exit Equity Value:";
  summarySheet.getCell("B13").value = exitValuation.exitEquityValue;
  summarySheet.getCell("B13").numFmt = currencyFormat;

  summarySheet.getCell("A14").value = "Exit Year:";
  summarySheet.getCell("B14").value = exitValuation.exitYear;

  summarySheet.getCell("A15").value = "Exit Multiple:";
  summarySheet.getCell("B15").value = exitValuation.exitMultiple;
  summarySheet.getCell("B15").numFmt = multipleFormat;

  summarySheet.getCell("A17").value = "KEY METRICS";
  summarySheet.getCell("A17").font = { bold: true, size: 14 };

  summarySheet.getCell("A18").value = "Entry Leverage:";
  summarySheet.getCell("B18").value = keyMetrics.entryLeverage;
  summarySheet.getCell("B18").numFmt = multipleFormat;

  summarySheet.getCell("A19").value = "Exit Leverage:";
  summarySheet.getCell("B19").value = keyMetrics.exitLeverage;
  summarySheet.getCell("B19").numFmt = multipleFormat;

  summarySheet.getCell("A20").value = "Debt Paydown:";
  summarySheet.getCell("B20").value = keyMetrics.debtPaydown;
  summarySheet.getCell("B20").numFmt = currencyFormat;

  summarySheet.getCell("A21").value = "% Debt Repaid:";
  summarySheet.getCell("B21").value = keyMetrics.debtPaydownPercent;
  summarySheet.getCell("B21").numFmt = percentFormat;

  // ============ SOURCES & USES ============
  const suSheet = workbook.addWorksheet("Sources_Uses");
  suSheet.columns = [{ width: 35 }, { width: 18 }, { width: 18 }];

  suSheet.getCell("A1").value = "SOURCES & USES OF FUNDS";
  suSheet.getCell("A1").font = { bold: true, size: 14 };

  suSheet.getCell("A3").value = "SOURCES";
  suSheet.getCell("B3").value = "Amount ($M)";
  suSheet.getCell("C3").value = "% of Total";
  suSheet.getRow(3).font = { bold: true };

  suSheet.getCell("A4").value = "Senior Debt";
  suSheet.getCell("B4").value = sourcesAndUses.sources.seniorDebt;
  suSheet.getCell("B4").numFmt = currencyFormat;
  suSheet.getCell("C4").value = sourcesAndUses.sources.seniorDebt / sourcesAndUses.sources.total;
  suSheet.getCell("C4").numFmt = percentFormat;

  suSheet.getCell("A5").value = "Subordinated Debt";
  suSheet.getCell("B5").value = sourcesAndUses.sources.subDebt;
  suSheet.getCell("B5").numFmt = currencyFormat;
  suSheet.getCell("C5").value = sourcesAndUses.sources.subDebt / sourcesAndUses.sources.total;
  suSheet.getCell("C5").numFmt = percentFormat;

  suSheet.getCell("A6").value = "Sponsor Equity";
  suSheet.getCell("B6").value = sourcesAndUses.sources.sponsorEquity;
  suSheet.getCell("B6").numFmt = currencyFormat;
  suSheet.getCell("C6").value = sourcesAndUses.sources.sponsorEquity / sourcesAndUses.sources.total;
  suSheet.getCell("C6").numFmt = percentFormat;

  suSheet.getCell("A7").value = "Management Rollover";
  suSheet.getCell("B7").value = sourcesAndUses.sources.managementRollover;
  suSheet.getCell("B7").numFmt = currencyFormat;
  suSheet.getCell("C7").value = sourcesAndUses.sources.managementRollover / sourcesAndUses.sources.total;
  suSheet.getCell("C7").numFmt = percentFormat;

  suSheet.getCell("A8").value = "TOTAL SOURCES";
  suSheet.getCell("B8").value = sourcesAndUses.sources.total;
  suSheet.getCell("B8").numFmt = currencyFormat;
  suSheet.getRow(8).font = { bold: true };

  suSheet.getCell("A10").value = "USES";
  suSheet.getRow(10).font = { bold: true };

  suSheet.getCell("A11").value = "Purchase Price";
  suSheet.getCell("B11").value = sourcesAndUses.uses.purchasePrice;
  suSheet.getCell("B11").numFmt = currencyFormat;
  suSheet.getCell("C11").value = sourcesAndUses.uses.purchasePrice / sourcesAndUses.uses.total;
  suSheet.getCell("C11").numFmt = percentFormat;

  suSheet.getCell("A12").value = "Transaction Costs";
  suSheet.getCell("B12").value = sourcesAndUses.uses.transactionCosts;
  suSheet.getCell("B12").numFmt = currencyFormat;
  suSheet.getCell("C12").value = sourcesAndUses.uses.transactionCosts / sourcesAndUses.uses.total;
  suSheet.getCell("C12").numFmt = percentFormat;

  suSheet.getCell("A13").value = "Financing Fees";
  suSheet.getCell("B13").value = sourcesAndUses.uses.financingFees;
  suSheet.getCell("B13").numFmt = currencyFormat;
  suSheet.getCell("C13").value = sourcesAndUses.uses.financingFees / sourcesAndUses.uses.total;
  suSheet.getCell("C13").numFmt = percentFormat;

  suSheet.getCell("A14").value = "TOTAL USES";
  suSheet.getCell("B14").value = sourcesAndUses.uses.total;
  suSheet.getCell("B14").numFmt = currencyFormat;
  suSheet.getCell("C14").value = 1.0;
  suSheet.getCell("C14").numFmt = percentFormat;
  suSheet.getRow(14).font = { bold: true };
  
  // Verify Sources = Uses
  suSheet.getCell("A16").value = "Sources = Uses Check:";
  suSheet.getCell("B16").value = Math.abs(sourcesAndUses.sources.total - sourcesAndUses.uses.total) < 0.01 ? "BALANCED" : "MISMATCH";
  suSheet.getCell("B16").font = { bold: true, color: { argb: Math.abs(sourcesAndUses.sources.total - sourcesAndUses.uses.total) < 0.01 ? "FF008000" : "FFFF0000" } };

  // ============ OPERATING PROJECTIONS ============
  const projSheet = workbook.addWorksheet("Projections");
  projSheet.columns = [
    { width: 25 },
    { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 },
  ];

  projSheet.addRow(["", "Year 0", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]);
  projSheet.getRow(1).font = { bold: true };
  projSheet.getRow(1).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFE0E0E0" } };

  projSheet.addRow(["Revenue ($M)", ...projections.revenue]);
  for (let i = 2; i <= 7; i++) projSheet.getCell(2, i).numFmt = currencyFormat;

  projSheet.addRow(["EBITDA ($M)", ...projections.ebitda]);
  for (let i = 2; i <= 7; i++) projSheet.getCell(3, i).numFmt = currencyFormat;

  projSheet.addRow(["EBITDA Margin", ...projections.ebitdaMargins]);
  for (let i = 2; i <= 7; i++) projSheet.getCell(4, i).numFmt = percentFormat;

  projSheet.addRow(["Free Cash Flow ($M)", ...projections.freeCashFlow]);
  for (let i = 2; i <= 7; i++) projSheet.getCell(5, i).numFmt = currencyFormat;

  // Add more detail to projections
  projSheet.addRow([""]);
  projSheet.addRow(["D&A ($M)", ...projections.da]);
  for (let i = 2; i <= 7; i++) projSheet.getCell(7, i).numFmt = currencyFormat;

  projSheet.addRow(["EBIT ($M)", ...projections.ebit]);
  for (let i = 2; i <= 7; i++) projSheet.getCell(8, i).numFmt = currencyFormat;

  projSheet.addRow(["CapEx ($M)", ...projections.capex]);
  for (let i = 2; i <= 7; i++) projSheet.getCell(9, i).numFmt = currencyFormat;

  projSheet.addRow(["NWC Change ($M)", ...projections.nwcChange]);
  for (let i = 2; i <= 7; i++) projSheet.getCell(10, i).numFmt = currencyFormat;

  // ============ DEBT SCHEDULE ============
  const debtSheet = workbook.addWorksheet("Debt_Schedule");
  debtSheet.columns = [
    { width: 25 },
    { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 },
  ];

  let debtRow = 1;
  
  // SENIOR DEBT SECTION
  debtSheet.addRow(["SENIOR DEBT", "Year 0", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]);
  debtSheet.getRow(debtRow).font = { bold: true };
  debtSheet.getRow(debtRow).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFE0E0E0" } };
  debtRow++;

  debtSheet.addRow(["Beginning Balance ($M)", "-", ...projections.seniorDebt.slice(0, 5)]);
  for (let i = 2; i <= 7; i++) debtSheet.getCell(debtRow, i).numFmt = currencyFormat;
  debtRow++;

  debtSheet.addRow(["Interest Expense ($M)", ...projections.seniorInterest]);
  for (let i = 2; i <= 7; i++) debtSheet.getCell(debtRow, i).numFmt = currencyFormat;
  debtRow++;

  debtSheet.addRow(["Principal Paydown ($M)", ...projections.seniorAmort]);
  for (let i = 2; i <= 7; i++) debtSheet.getCell(debtRow, i).numFmt = currencyFormat;
  debtRow++;

  debtSheet.addRow(["Ending Balance ($M)", ...projections.seniorDebt]);
  for (let i = 2; i <= 7; i++) debtSheet.getCell(debtRow, i).numFmt = currencyFormat;
  debtRow++;

  debtSheet.addRow([""]);
  debtRow++;

  // SUBORDINATED DEBT SECTION
  debtSheet.addRow(["SUBORDINATED DEBT", "Year 0", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]);
  debtSheet.getRow(debtRow).font = { bold: true };
  debtSheet.getRow(debtRow).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFE0E0E0" } };
  debtRow++;

  debtSheet.addRow(["Beginning Balance ($M)", "-", ...projections.subDebt.slice(0, 5)]);
  for (let i = 2; i <= 7; i++) debtSheet.getCell(debtRow, i).numFmt = currencyFormat;
  debtRow++;

  debtSheet.addRow(["Interest Expense ($M)", ...projections.subInterest]);
  for (let i = 2; i <= 7; i++) debtSheet.getCell(debtRow, i).numFmt = currencyFormat;
  debtRow++;

  debtSheet.addRow(["Principal Paydown ($M)", ...projections.subAmort]);
  for (let i = 2; i <= 7; i++) debtSheet.getCell(debtRow, i).numFmt = currencyFormat;
  debtRow++;

  debtSheet.addRow(["Ending Balance ($M)", ...projections.subDebt]);
  for (let i = 2; i <= 7; i++) debtSheet.getCell(debtRow, i).numFmt = currencyFormat;
  debtRow++;

  debtSheet.addRow([""]);
  debtRow++;

  // TOTAL DEBT SECTION
  debtSheet.addRow(["TOTAL DEBT", "Year 0", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]);
  debtSheet.getRow(debtRow).font = { bold: true };
  debtSheet.getRow(debtRow).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFE0E0E0" } };
  debtRow++;

  const totalDebt = projections.seniorDebt.map((s, i) => s + projections.subDebt[i]);
  debtSheet.addRow(["Total Debt ($M)", ...totalDebt]);
  for (let i = 2; i <= 7; i++) debtSheet.getCell(debtRow, i).numFmt = currencyFormat;
  debtRow++;

  const totalInterest = projections.seniorInterest.map((s, i) => s + projections.subInterest[i]);
  debtSheet.addRow(["Total Interest ($M)", ...totalInterest]);
  for (let i = 2; i <= 7; i++) debtSheet.getCell(debtRow, i).numFmt = currencyFormat;
  debtRow++;

  // FREE CASH FLOW SUMMARY
  debtSheet.addRow([""]);
  debtRow++;
  
  debtSheet.addRow(["CASH FLOW SUMMARY", "Year 0", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]);
  debtSheet.getRow(debtRow).font = { bold: true };
  debtSheet.getRow(debtRow).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFFFCC00" } };
  debtRow++;
  
  debtSheet.addRow(["Cash Available for Debt ($M)", ...projections.freeCashFlow]);
  for (let i = 2; i <= 7; i++) debtSheet.getCell(debtRow, i).numFmt = currencyFormat;

  // ============ RETURNS ANALYSIS ============
  const returnsSheet = workbook.addWorksheet("Returns_Analysis");
  returnsSheet.columns = [{ width: 30 }, { width: 20 }, { width: 20 }];

  returnsSheet.getCell("A1").value = "EXIT VALUATION";
  returnsSheet.getCell("A1").font = { bold: true, size: 14 };

  returnsSheet.getCell("A3").value = "Exit Year EBITDA:";
  returnsSheet.getCell("B3").value = exitValuation.exitEBITDA;
  returnsSheet.getCell("B3").numFmt = currencyFormat;

  returnsSheet.getCell("A4").value = "Exit Multiple:";
  returnsSheet.getCell("B4").value = exitValuation.exitMultiple;
  returnsSheet.getCell("B4").numFmt = multipleFormat;

  returnsSheet.getCell("A5").value = "Exit Enterprise Value:";
  returnsSheet.getCell("B5").value = exitValuation.exitEV;
  returnsSheet.getCell("B5").numFmt = currencyFormat;

  returnsSheet.getCell("A6").value = "Less: Remaining Debt:";
  returnsSheet.getCell("B6").value = -exitValuation.remainingDebt;
  returnsSheet.getCell("B6").numFmt = currencyFormat;

  returnsSheet.getCell("A7").value = "Less: Exit Costs:";
  returnsSheet.getCell("B7").value = -exitValuation.exitTransactionCosts;
  returnsSheet.getCell("B7").numFmt = currencyFormat;

  returnsSheet.getCell("A8").value = "Exit Equity Value:";
  returnsSheet.getCell("B8").value = exitValuation.exitEquityValue;
  returnsSheet.getCell("B8").numFmt = currencyFormat;
  returnsSheet.getRow(8).font = { bold: true };

  returnsSheet.getCell("A10").value = "SPONSOR RETURNS";
  returnsSheet.getCell("A10").font = { bold: true, size: 14 };

  returnsSheet.getCell("A11").value = "Initial Investment:";
  returnsSheet.getCell("B11").value = returns.sponsor.equity;
  returnsSheet.getCell("B11").numFmt = currencyFormat;

  returnsSheet.getCell("A12").value = "Ownership %:";
  returnsSheet.getCell("B12").value = returns.sponsor.ownership;
  returnsSheet.getCell("B12").numFmt = percentFormat;

  returnsSheet.getCell("A13").value = "Exit Proceeds:";
  returnsSheet.getCell("B13").value = returns.sponsor.exitProceeds;
  returnsSheet.getCell("B13").numFmt = currencyFormat;

  returnsSheet.getCell("A14").value = "MOIC:";
  returnsSheet.getCell("B14").value = returns.sponsor.moic;
  returnsSheet.getCell("B14").numFmt = multipleFormat;
  returnsSheet.getCell("B14").font = { bold: true };
  returnsSheet.getCell("B14").fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFFFFF00" } };

  returnsSheet.getCell("A15").value = "IRR:";
  returnsSheet.getCell("B15").value = returns.sponsor.irr;
  returnsSheet.getCell("B15").numFmt = percentFormat;
  returnsSheet.getCell("B15").font = { bold: true };
  returnsSheet.getCell("B15").fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFFFFF00" } };

  if (returns.management.equity > 0) {
    returnsSheet.getCell("A17").value = "MANAGEMENT RETURNS";
    returnsSheet.getCell("A17").font = { bold: true, size: 14 };

    returnsSheet.getCell("A18").value = "Initial Investment (Rollover):";
    returnsSheet.getCell("B18").value = returns.management.equity;
    returnsSheet.getCell("B18").numFmt = currencyFormat;

    returnsSheet.getCell("A19").value = "Exit Proceeds:";
    returnsSheet.getCell("B19").value = returns.management.exitProceeds;
    returnsSheet.getCell("B19").numFmt = currencyFormat;

    returnsSheet.getCell("A20").value = "MOIC:";
    returnsSheet.getCell("B20").value = returns.management.moic;
    returnsSheet.getCell("B20").numFmt = multipleFormat;

    returnsSheet.getCell("A21").value = "IRR:";
    returnsSheet.getCell("B21").value = returns.management.irr;
    returnsSheet.getCell("B21").numFmt = percentFormat;
  }

  // ============ SENSITIVITY ANALYSIS ============
  const sensSheet = workbook.addWorksheet("Sensitivity_IRR");
  sensSheet.columns = [
    { width: 15 },
    { width: 12 }, { width: 12 }, { width: 12 }, { width: 12 }, { width: 12 },
  ];

  sensSheet.getCell("A1").value = "SPONSOR IRR SENSITIVITY";
  sensSheet.getCell("A1").font = { bold: true, size: 14 };
  sensSheet.getCell("A2").value = "Exit Multiple vs Revenue CAGR";

  sensSheet.getCell("B3").value = "Revenue CAGR";
  sensSheet.mergeCells("B3:F3");
  sensSheet.getCell("B3").alignment = { horizontal: "center" };
  sensSheet.getCell("B3").font = { bold: true };

  const baseGrowth = assumptions.revenueGrowthRates.reduce((a, b) => a + b, 0) / 5;
  const growthValues = [baseGrowth - 0.05, baseGrowth - 0.025, baseGrowth, baseGrowth + 0.025, baseGrowth + 0.05];
  const exitMultiples = [exitValuation.exitMultiple - 2, exitValuation.exitMultiple - 1, exitValuation.exitMultiple, exitValuation.exitMultiple + 1, exitValuation.exitMultiple + 2];

  sensSheet.addRow(["Exit Multiple", ...growthValues]);
  sensSheet.getRow(4).font = { bold: true };
  for (let i = 2; i <= 6; i++) sensSheet.getCell(4, i).numFmt = percentFormat;

  for (let m = 0; m < exitMultiples.length; m++) {
    const row: (number | string)[] = [exitMultiples[m]];
    for (let g = 0; g < growthValues.length; g++) {
      // Simplified IRR calculation for sensitivity
      const modifiedGrowth = growthValues[g];
      const exitEBITDA = assumptions.baseYearRevenue * Math.pow(1 + modifiedGrowth, 5) * assumptions.targetEBITDAMargin;
      const exitEV = exitEBITDA * exitMultiples[m];
      const remainingDebt = projections.seniorDebt[5] + projections.subDebt[5];
      const exitEquity = exitEV - remainingDebt - (exitEV * assumptions.exitCosts);
      const sponsorExit = exitEquity * returns.sponsor.ownership;
      const irr = Math.pow(sponsorExit / assumptions.sponsorEquity, 1 / 5) - 1;
      row.push(irr);
    }
    const excelRow = sensSheet.addRow(row);
    excelRow.getCell(1).numFmt = multipleFormat;
    for (let i = 2; i <= 6; i++) {
      excelRow.getCell(i).numFmt = percentFormat;
      if (m === 2 && i === 4) {
        excelRow.getCell(i).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFFFFF00" } };
      }
    }
  }

  const buffer = await workbook.xlsx.writeBuffer();
  return Buffer.from(buffer);
}
