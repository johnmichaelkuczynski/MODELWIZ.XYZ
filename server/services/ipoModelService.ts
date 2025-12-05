import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import ExcelJS from "exceljs";

export type FinanceLLMProvider = "zhi1" | "zhi2" | "zhi3" | "zhi4" | "zhi5";

// Company type classification for valuation methodology selection
export type CompanyType = "mature" | "high-growth" | "biotech" | "industrial";

// Shareholder structure for pre-IPO ownership
interface Shareholder {
  name: string;
  shares: number;
  type: "founder" | "investor" | "option_pool" | "management" | "other";
  votingMultiple?: number; // For dual-class shares (e.g., 10 for 10:1 voting)
  secondarySalePercent?: number; // Percentage of position being sold (e.g., 0.40 for selling 40%)
}

// Comparable company for valuation
interface ComparableCompany {
  name: string;
  evRevenue?: number;
  evEbitda?: number;
  pegRatio?: number;
  marketCap?: number;
}

// Pipeline asset for biotech companies
interface PipelineAsset {
  name: string;
  phase: string;
  probabilityOfSuccess: number;
  peakSales: number;
  launchYear: number;
  patentExpiry?: number;
}

export interface IPOAssumptions {
  companyName: string;
  companyType: CompanyType;
  
  // Pre-IPO Capital Structure
  preIPOShares: {
    total: number;
    breakdown: Shareholder[];
  };
  
  // Financial Data
  currentRevenue: number;
  revenueGrowthRates: number[];
  currentEBITDA: number;
  ebitdaMargin: number;
  targetEbitdaMargin: number;
  currentCash: number;
  currentDebt: number;
  interestRate: number;
  taxRate: number;
  wacc: number;
  terminalGrowthRate: number;
  
  // IPO Specifics
  primaryRaiseAmount: number;
  secondarySaleShares: number; // Can be calculated from shareholder secondarySalePercent or specified directly
  greenshoePercent: number;
  debtRepaymentAmount: number; // Amount of debt to pay down from proceeds
  
  // Valuation Comparables
  comparables: ComparableCompany[];
  
  // Special Cases
  dualClassShares: boolean;
  founderControlMinimum?: number;
  lockUpPeriodDays: number;
  pipelineAssets?: PipelineAsset[];
  
  // Customization
  ipoDiscountPercent: number;
  projectionYears: number;
}

export interface IPOPricingResult {
  companyName: string;
  companyType: CompanyType;
  valuationMethodology: string;
  
  // Core Valuation
  enterpriseValue: number;
  equityValue: number;
  fairValuePerShare: number;
  
  // Share Mechanics
  preIPOSharesTotal: number;
  primarySharesIssued: number;
  secondarySharesSold: number;
  greenshoeShares: number;
  postIPOSharesTotal: number;
  publicFloat: number;
  
  // Price Range
  priceRange: {
    low: number;
    mid: number;
    high: number;
  };
  recommendedOfferPrice: number;
  
  // Proceeds
  grossPrimaryProceeds: number;
  netPrimaryProceeds: number;
  secondaryProceeds: number;
  greenshoeProceeds: number;
  totalOfferingSize: number;
  
  // Post-IPO Metrics
  postIPOCash: number;
  postIPODebt: number;
  postIPOEnterpriseValue: number;
  marketCap: number;
  
  // Pre-IPO Valuation (implied)
  impliedPreIPOValuation: number;
  
  // Ownership & Dilution
  dilutionPercent: number;
  ownershipTable: {
    name: string;
    preIPOShares: number;
    preIPOPercent: number;
    postIPOShares: number;
    postIPOPercent: number;
    votingPercent: number;
  }[];
  
  // Valuation Multiples
  impliedEVRevenue: number;
  impliedEVEBITDA: number | null;
  comparableMedianEVRevenue: number;
  comparableMedianEVEBITDA: number | null;
  
  // Pricing Matrix (3 scenarios)
  pricingMatrix: {
    scenario: string;
    offerPrice: number;
    primaryShares: number;
    marketCap: number;
    enterpriseValue: number;
    evRevenue: number;
    evEbitda: number | null;
    dilution: number;
    founderOwnership: number;
    expectedDayOnePop: number;
  }[];
  
  // Warnings
  warnings: string[];
  
  // Recommendation
  recommendation: string;
  
  // Provider used
  providerUsed: string;
}

const IPO_PARSING_PROMPT = `You are an expert investment banker specializing in IPO pricing and valuation. Parse the following natural language description of a company preparing for IPO and extract all relevant parameters.

Return a JSON object with the following structure:
{
  "companyName": "Company Name",
  "companyType": "mature" | "high-growth" | "biotech" | "industrial",
  
  "preIPOShares": {
    "total": number (total fully diluted shares in millions),
    "breakdown": [
      {
        "name": "Founder/CEO",
        "shares": number (in millions),
        "type": "founder" | "investor" | "option_pool" | "management" | "other",
        "votingMultiple": number (optional, for dual-class shares, e.g., 10 for 10:1),
        "secondarySalePercent": number (optional, percentage of position being sold, e.g., 0.40 for selling 40% of their shares)
      }
    ]
  },
  
  "currentRevenue": number (in millions, LTM revenue),
  "revenueGrowthRates": [year1, year2, year3, year4, year5] (as decimals, e.g., 0.40 for 40%),
  "currentEBITDA": number (in millions, can be negative),
  "ebitdaMargin": number (as decimal, current),
  "targetEbitdaMargin": number (as decimal, at maturity),
  "currentCash": number (in millions),
  "currentDebt": number (in millions),
  "interestRate": number (as decimal, e.g., 0.06 for 6%),
  "taxRate": number (as decimal, e.g., 0.21 for 21%),
  "wacc": number (as decimal, e.g., 0.12 for 12%),
  "terminalGrowthRate": number (as decimal, e.g., 0.03 for 3%),
  
  "primaryRaiseAmount": number (in millions, new money raised by company),
  "secondarySaleShares": number (in millions, total shares sold by existing holders - set to 0 if using per-shareholder secondarySalePercent),
  "greenshoePercent": number (as decimal, typically 0.15 for 15%),
  "debtRepaymentAmount": number (in millions, amount of debt to pay down from IPO proceeds, default 0),
  
  "comparables": [
    {
      "name": "Comparable Company",
      "evRevenue": number (EV/Revenue multiple),
      "evEbitda": number (EV/EBITDA multiple, null if unprofitable),
      "marketCap": number (in billions, optional)
    }
  ],
  
  "dualClassShares": boolean,
  "founderControlMinimum": number (as decimal, e.g., 0.20 for 20% minimum ownership),
  "lockUpPeriodDays": number (typically 180),
  
  "pipelineAssets": [
    {
      "name": "Drug Name",
      "phase": "Phase 1/2/3/Approved",
      "probabilityOfSuccess": number (as decimal),
      "peakSales": number (in millions),
      "launchYear": number
    }
  ] (only for biotech companies, otherwise null),
  
  "ipoDiscountPercent": number (as decimal, e.g., 0.15 for 15% discount to fair value),
  "projectionYears": number (default 5)
}

SECONDARY SALE EXTRACTION:
- If the description says "Series A selling 40% of position" and Series A owns 14.2M shares, set secondarySalePercent: 0.40 for that shareholder
- Calculate total secondary shares from: sum of (shares × secondarySalePercent) for each selling shareholder
- If a flat number of secondary shares is given, use secondarySaleShares directly

DEBT REPAYMENT EXTRACTION:
- If the description says "pay down $150M of debt", set debtRepaymentAmount: 150
- This reduces post-IPO debt by this amount

COMPANY TYPE CLASSIFICATION:
- "mature": Profitable, stable growth, positive EBITDA, established business model
- "high-growth": High revenue growth (>25%), may have negative EBITDA, SaaS/tech companies
- "biotech": Pre-revenue or minimal revenue, pipeline-dependent, FDA approval risk
- "industrial": Asset-heavy, capital-intensive, manufacturing/infrastructure

DEFAULTS if not specified:
- revenueGrowthRates: Based on company type (mature: 5-10%, high-growth: 25-50%, biotech: varies, industrial: 3-8%)
- greenshoePercent: 0.15 (15%)
- lockUpPeriodDays: 180
- ipoDiscountPercent: 0.15 (15%)
- projectionYears: 5
- wacc: 0.10-0.15 depending on risk
- terminalGrowthRate: 0.025

CRITICAL RULES:
1. ALWAYS extract pre-IPO share count from the description - this is crucial for ownership calculations
2. If the description mentions specific shareholders (founders, VCs, etc.), include each in the breakdown
3. For biotech companies, extract any pipeline assets mentioned
4. If dual-class shares are mentioned, set dualClassShares to true and extract voting multiples
5. The company type determines which valuation methodology will be used

IMPORTANT: Return ONLY valid JSON, no markdown, no explanations.`;

export async function parseIPODescription(
  description: string,
  provider: FinanceLLMProvider,
  customInstructions?: string
): Promise<{ assumptions: IPOAssumptions; providerUsed: string }> {
  const fullPrompt = customInstructions
    ? `${IPO_PARSING_PROMPT}\n\nAdditional Instructions: ${customInstructions}\n\nDescription:\n${description}`
    : `${IPO_PARSING_PROMPT}\n\nDescription:\n${description}`;

  let responseText: string = "";
  let providerUsed: string = "";

  if (provider === "zhi1") {
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: fullPrompt }],
      temperature: 0.2,
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
      temperature: 0.2,
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
        temperature: 0.2,
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
      temperature: 0.2,
    });
    responseText = response.choices[0]?.message?.content || "";
    providerUsed = "ZHI 5";
  }

  // Parse JSON response
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

  console.log("[IPO Model] Raw AI response:", jsonStr.substring(0, 500));

  const assumptions: IPOAssumptions = JSON.parse(jsonStr);
  
  // Validate critical fields
  if (!assumptions.preIPOShares || assumptions.preIPOShares.total <= 0) {
    throw new Error("Pre-IPO share count is required and must be greater than 0");
  }
  
  return { assumptions, providerUsed };
}

export function calculateIPOPricing(assumptions: IPOAssumptions): Omit<IPOPricingResult, "providerUsed"> {
  const {
    companyName,
    companyType,
    preIPOShares,
    currentRevenue,
    revenueGrowthRates,
    currentEBITDA,
    ebitdaMargin,
    targetEbitdaMargin,
    currentCash,
    currentDebt,
    taxRate,
    wacc,
    terminalGrowthRate,
    primaryRaiseAmount,
    secondarySaleShares: directSecondarySaleShares,
    greenshoePercent,
    comparables,
    dualClassShares,
    founderControlMinimum,
    pipelineAssets,
    ipoDiscountPercent,
    projectionYears,
    debtRepaymentAmount = 0, // ERROR 3 FIX: Default to 0 if not specified
  } = assumptions;

  console.log(`[IPO Model] ============ IPO PRICING ANALYSIS ============`);
  console.log(`[IPO Model] Company: ${companyName}, Type: ${companyType}`);
  console.log(`[IPO Model] Pre-IPO Shares: ${preIPOShares.total.toFixed(2)}M`);
  console.log(`[IPO Model] Primary Raise Target: $${primaryRaiseAmount.toFixed(2)}M`);
  console.log(`[IPO Model] Debt Repayment Target: $${debtRepaymentAmount.toFixed(2)}M`);

  const warnings: string[] = [];
  let valuationMethodology = "";
  let enterpriseValue = 0;
  
  // ============ ERROR 1 FIX: CALCULATE SECONDARY SHARES FROM SHAREHOLDER PERCENTAGES ============
  // If shareholders have secondarySalePercent, calculate from that; otherwise use direct value
  let calculatedSecondaryShares = 0;
  const shareholderSecondaryDetails: { name: string; sharesSold: number }[] = [];
  
  for (const holder of preIPOShares.breakdown) {
    if (holder.secondarySalePercent && holder.secondarySalePercent > 0) {
      const sharesSold = holder.shares * holder.secondarySalePercent;
      calculatedSecondaryShares += sharesSold;
      shareholderSecondaryDetails.push({ name: holder.name, sharesSold });
      console.log(`[IPO Model] Secondary: ${holder.name} selling ${(holder.secondarySalePercent * 100).toFixed(0)}% of ${holder.shares.toFixed(2)}M = ${sharesSold.toFixed(2)}M shares`);
    }
  }
  
  // Use calculated secondary shares if any, otherwise fall back to direct value
  const secondarySaleShares = calculatedSecondaryShares > 0 ? calculatedSecondaryShares : directSecondarySaleShares;
  console.log(`[IPO Model] Total Secondary Shares: ${secondarySaleShares.toFixed(2)}M`);

  // ============ STEP 1: CALCULATE ENTERPRISE VALUE BASED ON COMPANY TYPE ============
  
  if (companyType === "mature" || companyType === "industrial") {
    // DCF-based valuation for mature/industrial companies
    valuationMethodology = companyType === "mature" 
      ? "Discounted Cash Flow (DCF) with Terminal Value"
      : "EBITDA Multiple with DCF Cross-Check";
    
    // Project FCF
    const projectedFCF: number[] = [];
    let projectedRevenue = currentRevenue;
    let projectedMargin = ebitdaMargin;
    const marginStep = (targetEbitdaMargin - ebitdaMargin) / projectionYears;
    
    for (let i = 0; i < projectionYears; i++) {
      const growthRate = revenueGrowthRates[i] || revenueGrowthRates[revenueGrowthRates.length - 1] || 0.05;
      projectedRevenue = projectedRevenue * (1 + growthRate);
      projectedMargin = Math.min(projectedMargin + marginStep, targetEbitdaMargin);
      
      const ebitda = projectedRevenue * projectedMargin;
      const da = projectedRevenue * 0.04;
      const capex = projectedRevenue * 0.05;
      const nwcChange = projectedRevenue * growthRate * 0.10;
      const fcf = (ebitda - da) * (1 - taxRate) + da - capex - nwcChange;
      projectedFCF.push(fcf);
    }
    
    // Terminal value
    const terminalFCF = projectedFCF[projectedFCF.length - 1] * (1 + terminalGrowthRate);
    const terminalValue = terminalFCF / (wacc - terminalGrowthRate);
    
    // DCF valuation
    let pvFCF = 0;
    for (let i = 0; i < projectedFCF.length; i++) {
      pvFCF += projectedFCF[i] / Math.pow(1 + wacc, i + 1);
    }
    const pvTerminal = terminalValue / Math.pow(1 + wacc, projectionYears);
    
    enterpriseValue = pvFCF + pvTerminal;
    
    // Cross-check with EBITDA multiple
    if (comparables.length > 0) {
      const medianEVEbitda = comparables
        .filter(c => c.evEbitda && c.evEbitda > 0)
        .map(c => c.evEbitda!)
        .sort((a, b) => a - b);
      
      if (medianEVEbitda.length > 0 && currentEBITDA > 0) {
        const midIdx = Math.floor(medianEVEbitda.length / 2);
        const medianMultiple = medianEVEbitda[midIdx];
        const comparableEV = currentEBITDA * medianMultiple;
        
        // Use more conservative of DCF or comparable
        if (companyType === "industrial") {
          enterpriseValue = Math.min(enterpriseValue, comparableEV);
        }
      }
    }
    
  } else if (companyType === "high-growth") {
    // Revenue multiple + DCF for high-growth companies
    valuationMethodology = "Revenue Multiple with DCF Cross-Check";
    
    // Get median EV/Revenue from comparables
    const evRevenueMultiples = comparables
      .filter(c => c.evRevenue && c.evRevenue > 0)
      .map(c => c.evRevenue!)
      .sort((a, b) => a - b);
    
    let medianEVRevenue = 10; // Default for high-growth
    if (evRevenueMultiples.length > 0) {
      const midIdx = Math.floor(evRevenueMultiples.length / 2);
      medianEVRevenue = evRevenueMultiples[midIdx];
    }
    
    // Apply comparable multiple to NTM revenue
    const ntmRevenue = currentRevenue * (1 + (revenueGrowthRates[0] || 0.30));
    const comparableEV = ntmRevenue * medianEVRevenue;
    
    // Also calculate DCF (even with negative EBITDA, project to profitability)
    const projectedFCF: number[] = [];
    let projectedRevenue = currentRevenue;
    let projectedMargin = ebitdaMargin;
    const marginStep = (targetEbitdaMargin - ebitdaMargin) / projectionYears;
    
    for (let i = 0; i < projectionYears; i++) {
      const growthRate = revenueGrowthRates[i] || revenueGrowthRates[revenueGrowthRates.length - 1] || 0.20;
      projectedRevenue = projectedRevenue * (1 + growthRate);
      projectedMargin = projectedMargin + marginStep;
      
      const ebitda = projectedRevenue * projectedMargin;
      const da = projectedRevenue * 0.03;
      const capex = projectedRevenue * 0.04;
      const nwcChange = projectedRevenue * growthRate * 0.08;
      const fcf = Math.max(0, (ebitda - da) * (1 - taxRate) + da - capex - nwcChange);
      projectedFCF.push(fcf);
    }
    
    const terminalFCF = projectedFCF[projectedFCF.length - 1] * (1 + terminalGrowthRate);
    const terminalValue = terminalFCF / (wacc - terminalGrowthRate);
    
    let pvFCF = 0;
    for (let i = 0; i < projectedFCF.length; i++) {
      pvFCF += projectedFCF[i] / Math.pow(1 + wacc, i + 1);
    }
    const pvTerminal = terminalValue / Math.pow(1 + wacc, projectionYears);
    const dcfEV = pvFCF + pvTerminal;
    
    // Use more conservative of the two
    enterpriseValue = Math.min(comparableEV, dcfEV);
    console.log(`[IPO Model] Comparable EV: $${comparableEV.toFixed(0)}M, DCF EV: $${dcfEV.toFixed(0)}M, Using: $${enterpriseValue.toFixed(0)}M`);
    
  } else if (companyType === "biotech") {
    // Risk-adjusted NPV (rNPV) for biotech
    valuationMethodology = "Risk-Adjusted NPV (rNPV) with Pipeline Analysis";
    
    if (pipelineAssets && pipelineAssets.length > 0) {
      let totalRNPV = 0;
      
      for (const asset of pipelineAssets) {
        const yearsToLaunch = asset.launchYear - new Date().getFullYear();
        const peakYear = yearsToLaunch + 3; // Peak sales ~3 years after launch
        
        // Model sales curve (ramp up, peak, decline)
        let assetNPV = 0;
        for (let year = yearsToLaunch; year <= yearsToLaunch + 10; year++) {
          let salesFactor = 0;
          const yearsPostLaunch = year - yearsToLaunch;
          
          if (yearsPostLaunch <= 3) {
            salesFactor = yearsPostLaunch / 3; // Ramp up
          } else if (yearsPostLaunch <= 7) {
            salesFactor = 1; // Peak
          } else {
            salesFactor = 1 - (yearsPostLaunch - 7) * 0.15; // Decline
          }
          
          const sales = asset.peakSales * Math.max(0, salesFactor);
          const margin = 0.30; // Typical biotech margin
          const fcf = sales * margin * (1 - taxRate);
          assetNPV += fcf / Math.pow(1 + wacc, year);
        }
        
        // Apply probability of success
        const riskAdjustedNPV = assetNPV * asset.probabilityOfSuccess;
        totalRNPV += riskAdjustedNPV;
        
        console.log(`[IPO Model] Pipeline: ${asset.name} (${asset.phase}), rNPV: $${riskAdjustedNPV.toFixed(0)}M`);
      }
      
      enterpriseValue = totalRNPV;
    } else {
      // No pipeline data - use revenue multiple with heavy discount
      const evRevenueMultiples = comparables
        .filter(c => c.evRevenue && c.evRevenue > 0)
        .map(c => c.evRevenue!);
      
      const medianMultiple = evRevenueMultiples.length > 0 
        ? evRevenueMultiples.sort((a, b) => a - b)[Math.floor(evRevenueMultiples.length / 2)]
        : 5;
      
      enterpriseValue = currentRevenue * medianMultiple * 0.5; // 50% discount for no pipeline data
      warnings.push("Limited pipeline data - valuation may be imprecise");
    }
  }

  // ============ STEP 2: CALCULATE EQUITY VALUE ============
  
  const equityValue = enterpriseValue - currentDebt + currentCash;
  const fairValuePerShare = equityValue / preIPOShares.total;
  
  console.log(`[IPO Model] Enterprise Value: $${enterpriseValue.toFixed(0)}M`);
  console.log(`[IPO Model] Equity Value: $${equityValue.toFixed(0)}M`);
  console.log(`[IPO Model] Fair Value/Share: $${fairValuePerShare.toFixed(2)}`);

  // ============ STEP 3: DETERMINE OFFER PRICE RANGE ============
  
  const discountedPrice = fairValuePerShare * (1 - ipoDiscountPercent);
  const midPrice = discountedPrice;
  const lowPrice = midPrice * 0.90; // 10% below mid
  const highPrice = midPrice * 1.10; // 10% above mid
  
  console.log(`[IPO Model] Price Range: $${lowPrice.toFixed(2)} - $${midPrice.toFixed(2)} - $${highPrice.toFixed(2)}`);

  // ============ STEP 4: CALCULATE SHARE MECHANICS ============
  
  // Primary shares needed at mid price
  const primarySharesAtMid = primaryRaiseAmount / midPrice;
  const primarySharesAtLow = primaryRaiseAmount / lowPrice;
  const primarySharesAtHigh = primaryRaiseAmount / highPrice;
  
  // Total offering (primary + secondary)
  const totalOfferingShares = primarySharesAtMid + secondarySaleShares;
  
  // Greenshoe
  const greenshoeShares = totalOfferingShares * greenshoePercent;
  
  // Post-IPO shares (CRITICAL: preserve pre-IPO shares)
  const postIPOShares = preIPOShares.total + primarySharesAtMid + greenshoeShares;
  
  // Public float
  const publicFloat = primarySharesAtMid + secondarySaleShares + greenshoeShares;
  
  console.log(`[IPO Model] Share Mechanics:`);
  console.log(`  Pre-IPO: ${preIPOShares.total.toFixed(2)}M`);
  console.log(`  Primary New: ${primarySharesAtMid.toFixed(2)}M`);
  console.log(`  Secondary: ${secondarySaleShares.toFixed(2)}M`);
  console.log(`  Greenshoe: ${greenshoeShares.toFixed(2)}M`);
  console.log(`  Post-IPO Total: ${postIPOShares.toFixed(2)}M`);

  // ============ STEP 5: CALCULATE PROCEEDS ============
  
  // Underwriting discount typically 7% for IPO
  const underwritingDiscount = 0.07;
  
  const grossPrimaryProceeds = primarySharesAtMid * midPrice;
  const netPrimaryProceeds = grossPrimaryProceeds * (1 - underwritingDiscount);
  const secondaryProceeds = secondarySaleShares * midPrice * (1 - underwritingDiscount);
  const greenshoeProceeds = greenshoeShares * midPrice * (1 - underwritingDiscount);
  const totalOfferingSize = (primarySharesAtMid + secondarySaleShares + greenshoeShares) * midPrice;
  
  console.log(`[IPO Model] Proceeds (at $${midPrice.toFixed(2)}):`);
  console.log(`  Gross Primary: $${grossPrimaryProceeds.toFixed(0)}M`);
  console.log(`  Net Primary (to Company): $${netPrimaryProceeds.toFixed(0)}M`);
  console.log(`  Secondary (to Sellers): $${secondaryProceeds.toFixed(0)}M`);

  // ============ STEP 6: POST-IPO FINANCIAL POSITION ============
  
  // ERROR 3 FIX: Implement debt repayment from proceeds
  // Company gets primary proceeds + greenshoe (NOT secondary - that goes to sellers)
  // Then pays down debt as specified
  const postIPOCash = currentCash + netPrimaryProceeds + greenshoeProceeds - debtRepaymentAmount;
  const postIPODebt = Math.max(0, currentDebt - debtRepaymentAmount); // Reduce debt by repayment amount
  const marketCap = postIPOShares * midPrice;
  const postIPOEnterpriseValue = marketCap + postIPODebt - postIPOCash;
  
  console.log(`[IPO Model] Post-IPO Debt: $${postIPODebt.toFixed(0)}M (after $${debtRepaymentAmount.toFixed(0)}M repayment)`);

  // ERROR 6 FIX: Calculate implied pre-IPO valuation
  // Pre-IPO Valuation = Market Cap at offer - Primary Proceeds (new money)
  // OR: Offer Price × Pre-IPO Shares
  const impliedPreIPOValuation = midPrice * preIPOShares.total;
  console.log(`[IPO Model] Implied Pre-IPO Valuation: $${impliedPreIPOValuation.toFixed(0)}M`);

  // ============ STEP 7: OWNERSHIP TABLE (with ERROR 1 & ERROR 4 fixes) ============
  
  // ERROR 4 FIX: Correct voting power calculation for dual-class shares
  // For dual-class: Voting Power = (Class B shares × voting multiple) ÷ (Total Class B × multiple + Total Class A × 1)
  // Calculate total voting power denominator
  let totalVotingShares = 0;
  for (const holder of preIPOShares.breakdown) {
    const votingWeight = dualClassShares && holder.votingMultiple ? holder.votingMultiple : 1;
    // ERROR 1 FIX: Subtract secondary sale shares from post-IPO holdings
    const holdersSecondarySale = holder.secondarySalePercent ? holder.shares * holder.secondarySalePercent : 0;
    const postIPOHolderShares = holder.shares - holdersSecondarySale;
    totalVotingShares += postIPOHolderShares * votingWeight;
  }
  // Public shareholders have 1 vote per share (Class A)
  totalVotingShares += publicFloat * 1;
  
  const ownershipTable = preIPOShares.breakdown.map(holder => {
    // ERROR 1 FIX: Subtract secondary sale shares from seller's post-IPO ownership
    const holdersSecondarySale = holder.secondarySalePercent ? holder.shares * holder.secondarySalePercent : 0;
    const postIPOHolderShares = holder.shares - holdersSecondarySale;
    const postIPOPercent = (postIPOHolderShares / postIPOShares) * 100;
    
    // ERROR 4 FIX: Correct voting power for dual-class
    const votingWeight = dualClassShares && holder.votingMultiple ? holder.votingMultiple : 1;
    const holderVotingPower = postIPOHolderShares * votingWeight;
    const votingPercent = (holderVotingPower / totalVotingShares) * 100;
    
    return {
      name: holder.name,
      preIPOShares: holder.shares,
      preIPOPercent: (holder.shares / preIPOShares.total) * 100,
      postIPOShares: postIPOHolderShares, // Reduced by secondary sale
      postIPOPercent,
      votingPercent,
    };
  });
  
  // Add public shareholders
  const publicVotingPercent = (publicFloat * 1 / totalVotingShares) * 100;
  ownershipTable.push({
    name: "Public Shareholders",
    preIPOShares: 0,
    preIPOPercent: 0,
    postIPOShares: publicFloat,
    postIPOPercent: (publicFloat / postIPOShares) * 100,
    votingPercent: publicVotingPercent,
  });
  
  // ERROR 2 FIX: Correct dilution formula
  // Dilution % = New Shares Issued ÷ (Pre-IPO Shares + New Shares Issued)
  // NOT: New Shares ÷ Pre-IPO Shares (which gives >100% incorrectly)
  const newSharesIssued = primarySharesAtMid + greenshoeShares;
  const dilutionPercent = (newSharesIssued / (preIPOShares.total + newSharesIssued)) * 100;
  console.log(`[IPO Model] Dilution: ${newSharesIssued.toFixed(2)}M / (${preIPOShares.total.toFixed(2)}M + ${newSharesIssued.toFixed(2)}M) = ${dilutionPercent.toFixed(1)}%`);
  
  // Check founder control (using corrected post-IPO shares after secondary sales)
  const founderHolders = preIPOShares.breakdown.filter(h => h.type === "founder");
  let totalFounderPostIPOShares = 0;
  let totalFounderVotingPower = 0;
  for (const holder of founderHolders) {
    const secondarySale = holder.secondarySalePercent ? holder.shares * holder.secondarySalePercent : 0;
    const postShares = holder.shares - secondarySale;
    totalFounderPostIPOShares += postShares;
    const votingWeight = dualClassShares && holder.votingMultiple ? holder.votingMultiple : 1;
    totalFounderVotingPower += postShares * votingWeight;
  }
  const founderPostIPOPercent = (totalFounderPostIPOShares / postIPOShares) * 100;
  const founderVotingPercent = (totalFounderVotingPower / totalVotingShares) * 100;
  
  console.log(`[IPO Model] Founder Post-IPO Ownership: ${founderPostIPOPercent.toFixed(1)}%`);
  console.log(`[IPO Model] Founder Voting Power: ${founderVotingPercent.toFixed(1)}%`);
  
  if (founderControlMinimum && founderPostIPOPercent < founderControlMinimum * 100) {
    warnings.push(`Founder ownership (${founderPostIPOPercent.toFixed(1)}%) falls below required minimum (${(founderControlMinimum * 100).toFixed(1)}%)`);
  }
  
  // VALIDATION: Sum of ownership should equal 100%
  const totalOwnership = ownershipTable.reduce((sum, h) => sum + h.postIPOPercent, 0);
  if (Math.abs(totalOwnership - 100) > 0.5) {
    warnings.push(`Ownership percentages sum to ${totalOwnership.toFixed(1)}% (should be 100%)`);
  }
  
  // VALIDATION: Dilution should always be < 100%
  if (dilutionPercent >= 100) {
    warnings.push(`Dilution of ${dilutionPercent.toFixed(1)}% is invalid - check share calculations`);
  }
  
  // ERROR 5 VALIDATION: Offer price should be lower than fair value (discount applied)
  if (midPrice >= fairValuePerShare) {
    warnings.push(`Offer price ($${midPrice.toFixed(2)}) should be below fair value ($${fairValuePerShare.toFixed(2)})`);
  }

  // ============ STEP 8: VALUATION MULTIPLES ============
  
  const impliedEVRevenue = postIPOEnterpriseValue / currentRevenue;
  const impliedEVEBITDA = currentEBITDA > 0 ? postIPOEnterpriseValue / currentEBITDA : null;
  
  const compRevMultiples = comparables.filter(c => c.evRevenue).map(c => c.evRevenue!).sort((a, b) => a - b);
  const comparableMedianEVRevenue = compRevMultiples.length > 0 
    ? compRevMultiples[Math.floor(compRevMultiples.length / 2)]
    : impliedEVRevenue;
  
  const compEbitdaMultiples = comparables.filter(c => c.evEbitda).map(c => c.evEbitda!).sort((a, b) => a - b);
  const comparableMedianEVEBITDA = compEbitdaMultiples.length > 0
    ? compEbitdaMultiples[Math.floor(compEbitdaMultiples.length / 2)]
    : null;

  // ============ STEP 9: PRICING MATRIX ============
  
  const createScenario = (scenarioName: string, price: number, primaryShares: number) => {
    const scenarioGreenshoe = (primaryShares + secondarySaleShares) * greenshoePercent;
    const scenarioNewShares = primaryShares + scenarioGreenshoe;
    const postShares = preIPOShares.total + scenarioNewShares;
    const mktCap = postShares * price;
    const netProceeds = primaryShares * price * (1 - underwritingDiscount);
    const scenarioGreenshoeProceeds = scenarioGreenshoe * price * (1 - underwritingDiscount);
    const postCash = currentCash + netProceeds + scenarioGreenshoeProceeds - debtRepaymentAmount;
    const postDebt = Math.max(0, currentDebt - debtRepaymentAmount);
    const ev = mktCap + postDebt - postCash;
    // ERROR 2 FIX: Correct dilution formula
    const dilution = (scenarioNewShares / (preIPOShares.total + scenarioNewShares)) * 100;
    const founderOwn = (totalFounderPostIPOShares / postShares) * 100;
    const dayOnePop = ((fairValuePerShare - price) / price) * 100;
    
    return {
      scenario: scenarioName,
      offerPrice: price,
      primaryShares,
      marketCap: mktCap,
      enterpriseValue: ev,
      evRevenue: ev / currentRevenue,
      evEbitda: currentEBITDA > 0 ? ev / currentEBITDA : null,
      dilution,
      founderOwnership: founderOwn,
      expectedDayOnePop: Math.max(0, dayOnePop),
    };
  };
  
  const pricingMatrix = [
    createScenario("Low", lowPrice, primarySharesAtLow),
    createScenario("Mid", midPrice, primarySharesAtMid),
    createScenario("High", highPrice, primarySharesAtHigh),
  ];

  // ============ STEP 10: RECOMMENDATION ============
  
  let recommendation = "";
  const expectedPop = ((fairValuePerShare - midPrice) / midPrice) * 100;
  
  if (expectedPop > 30) {
    recommendation = `PRICE HIGHER: Expected day-one pop of ${expectedPop.toFixed(0)}% suggests significant money left on the table. Consider pricing at the high end ($${highPrice.toFixed(2)}) or above.`;
  } else if (expectedPop < 10) {
    recommendation = `PRICE CAREFULLY: Expected day-one pop of ${expectedPop.toFixed(0)}% is below typical range. Consider pricing at the low end ($${lowPrice.toFixed(2)}) to ensure successful offering.`;
  } else {
    recommendation = `PRICE AT MID-POINT: Expected day-one pop of ${expectedPop.toFixed(0)}% is healthy. Recommended offer price of $${midPrice.toFixed(2)} balances proceeds maximization with successful execution.`;
  }
  
  if (dilutionPercent > 30) {
    warnings.push(`High dilution (${dilutionPercent.toFixed(1)}%) may concern existing shareholders`);
  }
  
  if (currentEBITDA < 0) {
    warnings.push("Company is not yet profitable - higher execution risk");
  }

  return {
    companyName,
    companyType,
    valuationMethodology,
    enterpriseValue,
    equityValue,
    fairValuePerShare,
    preIPOSharesTotal: preIPOShares.total,
    primarySharesIssued: primarySharesAtMid,
    secondarySharesSold: secondarySaleShares,
    greenshoeShares,
    postIPOSharesTotal: postIPOShares,
    publicFloat,
    priceRange: { low: lowPrice, mid: midPrice, high: highPrice },
    recommendedOfferPrice: midPrice,
    grossPrimaryProceeds,
    netPrimaryProceeds,
    secondaryProceeds,
    greenshoeProceeds,
    totalOfferingSize,
    postIPOCash,
    postIPODebt,
    postIPOEnterpriseValue,
    marketCap,
    impliedPreIPOValuation, // ERROR 6 FIX: Include implied pre-IPO valuation
    dilutionPercent,
    ownershipTable,
    impliedEVRevenue,
    impliedEVEBITDA,
    comparableMedianEVRevenue,
    comparableMedianEVEBITDA,
    pricingMatrix,
    warnings,
    recommendation,
  };
}

export async function generateIPOExcel(result: IPOPricingResult): Promise<Buffer> {
  const workbook = new ExcelJS.Workbook();
  workbook.creator = "IPO Pricing Analysis Tool";
  workbook.created = new Date();
  
  // ============ SUMMARY TAB ============
  const summarySheet = workbook.addWorksheet("Summary");
  summarySheet.columns = [
    { header: "", key: "label", width: 35 },
    { header: "", key: "value", width: 25 },
    { header: "", key: "notes", width: 40 },
  ];
  
  // Title
  summarySheet.mergeCells("A1:C1");
  summarySheet.getCell("A1").value = `${result.companyName} - IPO Pricing Analysis`;
  summarySheet.getCell("A1").font = { bold: true, size: 16 };
  summarySheet.getCell("A1").alignment = { horizontal: "center" };
  
  summarySheet.getCell("A2").value = `Generated: ${new Date().toLocaleDateString()}`;
  summarySheet.getCell("A3").value = `Valuation Methodology: ${result.valuationMethodology}`;
  summarySheet.getCell("A4").value = `Company Type: ${result.companyType.charAt(0).toUpperCase() + result.companyType.slice(1)}`;
  
  // Warnings
  if (result.warnings.length > 0) {
    summarySheet.getCell("A6").value = "WARNINGS";
    summarySheet.getCell("A6").font = { bold: true, color: { argb: "FFFF0000" } };
    result.warnings.forEach((warning, idx) => {
      summarySheet.getCell(`A${7 + idx}`).value = `• ${warning}`;
      summarySheet.getCell(`A${7 + idx}`).font = { color: { argb: "FFFF0000" } };
    });
  }
  
  const startRow = 8 + result.warnings.length;
  
  // Recommended Price
  summarySheet.getCell(`A${startRow}`).value = "RECOMMENDED OFFER PRICE";
  summarySheet.getCell(`A${startRow}`).font = { bold: true, size: 14 };
  summarySheet.getCell(`B${startRow}`).value = result.recommendedOfferPrice;
  summarySheet.getCell(`B${startRow}`).numFmt = "$#,##0.00";
  summarySheet.getCell(`B${startRow}`).font = { bold: true, size: 14 };
  
  summarySheet.getCell(`A${startRow + 1}`).value = "Price Range";
  summarySheet.getCell(`B${startRow + 1}`).value = `$${result.priceRange.low.toFixed(2)} - $${result.priceRange.high.toFixed(2)}`;
  
  // Key Metrics
  const metricsStart = startRow + 3;
  summarySheet.getCell(`A${metricsStart}`).value = "KEY METRICS";
  summarySheet.getCell(`A${metricsStart}`).font = { bold: true };
  
  const metrics = [
    ["Fair Value per Share", result.fairValuePerShare, "$#,##0.00"],
    ["Enterprise Value", result.enterpriseValue, "$#,##0.0,,\"M\""],
    ["Equity Value", result.equityValue, "$#,##0.0,,\"M\""],
    ["Implied Pre-IPO Valuation", result.impliedPreIPOValuation, "$#,##0.0,,\"M\""],
    ["Market Cap (at offer)", result.marketCap, "$#,##0.0,,\"M\""],
    ["Post-IPO Enterprise Value", result.postIPOEnterpriseValue, "$#,##0.0,,\"M\""],
    ["Post-IPO Debt", result.postIPODebt, "$#,##0.0,,\"M\""],
    ["Total Offering Size", result.totalOfferingSize, "$#,##0.0,,\"M\""],
    ["Implied EV/Revenue", result.impliedEVRevenue, "0.0x"],
    ["Implied EV/EBITDA", result.impliedEVEBITDA || "N/A", result.impliedEVEBITDA ? "0.0x" : "@"],
    ["Dilution", result.dilutionPercent / 100, "0.0%"],
  ];
  
  metrics.forEach((metric, idx) => {
    const row = metricsStart + 1 + idx;
    summarySheet.getCell(`A${row}`).value = metric[0] as string;
    summarySheet.getCell(`B${row}`).value = metric[1];
    if (typeof metric[1] === "number") {
      summarySheet.getCell(`B${row}`).numFmt = metric[2] as string;
    }
  });
  
  // Recommendation
  const recRow = metricsStart + metrics.length + 2;
  summarySheet.getCell(`A${recRow}`).value = "RECOMMENDATION";
  summarySheet.getCell(`A${recRow}`).font = { bold: true };
  summarySheet.mergeCells(`A${recRow + 1}:C${recRow + 1}`);
  summarySheet.getCell(`A${recRow + 1}`).value = result.recommendation;
  summarySheet.getCell(`A${recRow + 1}`).alignment = { wrapText: true };
  summarySheet.getRow(recRow + 1).height = 50;

  // ============ SHARE ISSUANCE TAB ============
  const sharesSheet = workbook.addWorksheet("Share Issuance");
  sharesSheet.columns = [
    { header: "", key: "label", width: 30 },
    { header: "Shares (M)", key: "shares", width: 15 },
    { header: "Notes", key: "notes", width: 40 },
  ];
  
  sharesSheet.getCell("A1").value = "SHARE ISSUANCE DETAIL";
  sharesSheet.getCell("A1").font = { bold: true, size: 14 };
  
  const shareRows = [
    ["Pre-IPO Fully Diluted Shares", result.preIPOSharesTotal, "All existing shares before IPO"],
    ["New Primary Shares Issued", result.primarySharesIssued, "Shares sold by company for new capital"],
    ["Secondary Shares Sold", result.secondarySharesSold, "Shares sold by existing shareholders"],
    ["Greenshoe Shares", result.greenshoeShares, `${(result.greenshoeShares / (result.primarySharesIssued + result.secondarySharesSold) * 100).toFixed(0)}% of offering`],
    ["Post-IPO Fully Diluted Shares", result.postIPOSharesTotal, "Total shares after IPO"],
    ["Public Float", result.publicFloat, "Shares available for trading"],
  ];
  
  shareRows.forEach((row, idx) => {
    sharesSheet.getCell(`A${idx + 3}`).value = row[0];
    sharesSheet.getCell(`B${idx + 3}`).value = row[1];
    sharesSheet.getCell(`B${idx + 3}`).numFmt = "#,##0.00";
    sharesSheet.getCell(`C${idx + 3}`).value = row[2];
  });

  // ============ OWNERSHIP TAB ============
  const ownershipSheet = workbook.addWorksheet("Ownership");
  ownershipSheet.columns = [
    { header: "Shareholder", key: "name", width: 25 },
    { header: "Pre-IPO Shares (M)", key: "preShares", width: 18 },
    { header: "Pre-IPO %", key: "prePct", width: 12 },
    { header: "Post-IPO Shares (M)", key: "postShares", width: 18 },
    { header: "Post-IPO %", key: "postPct", width: 12 },
    { header: "Voting %", key: "votingPct", width: 12 },
  ];
  
  ownershipSheet.getCell("A1").value = "POST-IPO OWNERSHIP TABLE";
  ownershipSheet.getCell("A1").font = { bold: true, size: 14 };
  
  // Headers
  const ownerHeaders = ["Shareholder", "Pre-IPO Shares (M)", "Pre-IPO %", "Post-IPO Shares (M)", "Post-IPO %", "Voting %"];
  ownerHeaders.forEach((header, idx) => {
    const cell = ownershipSheet.getCell(3, idx + 1);
    cell.value = header;
    cell.font = { bold: true };
    cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFE0E0E0" } };
  });
  
  result.ownershipTable.forEach((holder, idx) => {
    const row = ownershipSheet.getRow(4 + idx);
    row.getCell(1).value = holder.name;
    row.getCell(2).value = holder.preIPOShares;
    row.getCell(2).numFmt = "#,##0.00";
    row.getCell(3).value = holder.preIPOPercent / 100;
    row.getCell(3).numFmt = "0.0%";
    row.getCell(4).value = holder.postIPOShares;
    row.getCell(4).numFmt = "#,##0.00";
    row.getCell(5).value = holder.postIPOPercent / 100;
    row.getCell(5).numFmt = "0.0%";
    row.getCell(6).value = holder.votingPercent / 100;
    row.getCell(6).numFmt = "0.0%";
  });

  // ============ PROCEEDS TAB ============
  const proceedsSheet = workbook.addWorksheet("Proceeds");
  proceedsSheet.columns = [
    { header: "", key: "label", width: 35 },
    { header: "Amount ($M)", key: "amount", width: 18 },
    { header: "Recipient", key: "recipient", width: 25 },
  ];
  
  proceedsSheet.getCell("A1").value = "PROCEEDS CALCULATION";
  proceedsSheet.getCell("A1").font = { bold: true, size: 14 };
  
  const proceedRows = [
    ["Gross Primary Proceeds", result.grossPrimaryProceeds, "Company (before fees)"],
    ["Less: Underwriting Discount (7%)", result.grossPrimaryProceeds * 0.07, "Underwriters"],
    ["Net Primary Proceeds", result.netPrimaryProceeds, "Company"],
    ["", "", ""],
    ["Secondary Sale Proceeds", result.secondaryProceeds, "Selling Shareholders"],
    ["Greenshoe Proceeds", result.greenshoeProceeds, "Company"],
    ["", "", ""],
    ["Total Offering Size", result.totalOfferingSize, ""],
    ["", "", ""],
    ["Post-IPO Cash", result.postIPOCash, "Company Balance Sheet"],
    ["Post-IPO Debt", result.postIPODebt, ""],
    ["Post-IPO Enterprise Value", result.postIPOEnterpriseValue, ""],
  ];
  
  proceedRows.forEach((row, idx) => {
    proceedsSheet.getCell(`A${idx + 3}`).value = row[0];
    if (typeof row[1] === "number") {
      proceedsSheet.getCell(`B${idx + 3}`).value = row[1];
      proceedsSheet.getCell(`B${idx + 3}`).numFmt = "$#,##0.0";
    }
    proceedsSheet.getCell(`C${idx + 3}`).value = row[2];
  });

  // ============ PRICING MATRIX TAB ============
  const matrixSheet = workbook.addWorksheet("Pricing Matrix");
  
  matrixSheet.getCell("A1").value = "PRICING MATRIX - SCENARIO COMPARISON";
  matrixSheet.getCell("A1").font = { bold: true, size: 14 };
  
  const matrixHeaders = ["Metric", "Low", "Mid", "High"];
  matrixHeaders.forEach((header, idx) => {
    const cell = matrixSheet.getCell(3, idx + 1);
    cell.value = header;
    cell.font = { bold: true };
    cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFE0E0E0" } };
  });
  
  const matrixMetrics: { label: string; values: (number | string)[]; format: string }[] = [
    { label: "Offer Price", values: result.pricingMatrix.map(s => s.offerPrice), format: "$#,##0.00" },
    { label: "Primary Shares (M)", values: result.pricingMatrix.map(s => s.primaryShares), format: "#,##0.00" },
    { label: "Market Cap ($M)", values: result.pricingMatrix.map(s => s.marketCap), format: "$#,##0.0" },
    { label: "Enterprise Value ($M)", values: result.pricingMatrix.map(s => s.enterpriseValue), format: "$#,##0.0" },
    { label: "EV/Revenue", values: result.pricingMatrix.map(s => s.evRevenue), format: "0.0x" },
    { label: "EV/EBITDA", values: result.pricingMatrix.map(s => s.evEbitda || "N/A"), format: "0.0x" },
    { label: "Dilution", values: result.pricingMatrix.map(s => s.dilution / 100), format: "0.0%" },
    { label: "Founder Ownership", values: result.pricingMatrix.map(s => s.founderOwnership / 100), format: "0.0%" },
    { label: "Expected Day-1 Pop", values: result.pricingMatrix.map(s => s.expectedDayOnePop / 100), format: "0.0%" },
  ];
  
  matrixMetrics.forEach((metric, rowIdx) => {
    matrixSheet.getCell(`A${4 + rowIdx}`).value = metric.label;
    metric.values.forEach((val, colIdx) => {
      const cell = matrixSheet.getCell(4 + rowIdx, 2 + colIdx);
      cell.value = val;
      if (typeof val === "number") {
        cell.numFmt = metric.format;
      }
    });
  });

  // ============ VALUATION TAB ============
  const valuationSheet = workbook.addWorksheet("Valuation");
  valuationSheet.columns = [
    { header: "", key: "label", width: 35 },
    { header: "", key: "value", width: 20 },
  ];
  
  valuationSheet.getCell("A1").value = "VALUATION MECHANICS";
  valuationSheet.getCell("A1").font = { bold: true, size: 14 };
  
  valuationSheet.getCell("A3").value = "Methodology";
  valuationSheet.getCell("B3").value = result.valuationMethodology;
  
  valuationSheet.getCell("A4").value = "Company Type";
  valuationSheet.getCell("B4").value = result.companyType.charAt(0).toUpperCase() + result.companyType.slice(1);
  
  valuationSheet.getCell("A6").value = "VALUATION METRICS";
  valuationSheet.getCell("A6").font = { bold: true };
  
  const valMetrics = [
    ["Implied EV/Revenue", result.impliedEVRevenue, "0.0x"],
    ["Comparable Median EV/Revenue", result.comparableMedianEVRevenue, "0.0x"],
    ["Implied EV/EBITDA", result.impliedEVEBITDA || "N/A", "0.0x"],
    ["Comparable Median EV/EBITDA", result.comparableMedianEVEBITDA || "N/A", "0.0x"],
  ];
  
  valMetrics.forEach((metric, idx) => {
    valuationSheet.getCell(`A${7 + idx}`).value = metric[0];
    valuationSheet.getCell(`B${7 + idx}`).value = metric[1];
    if (typeof metric[1] === "number") {
      valuationSheet.getCell(`B${7 + idx}`).numFmt = metric[2] as string;
    }
  });

  // Generate buffer
  const buffer = await workbook.xlsx.writeBuffer();
  return Buffer.from(buffer);
}
