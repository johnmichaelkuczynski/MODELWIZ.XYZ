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
  acquirerExplicitEPS?: number; // BUG #1 FIX: If user provides EPS directly, use this
  acquirerCash?: number; // Acquirer's existing cash
  acquirerExistingDebt?: number; // Acquirer's existing debt
  
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
  transactionFees?: number; // Explicit transaction fees if provided
  
  // Financing
  cashFromBalance: number;
  newDebtAmount: number;
  newDebtRate: number;
  debtAmortizationRate?: number; // Annual amortization rate (e.g., 0.05 for 5%)
  debtMaturityYears?: number; // Years to maturity
  
  // Synergies - Revenue (typically slower realization)
  revenueSynergies: number;
  revenueSynergyRealizationY1: number;
  revenueSynergyRealizationY2: number;
  revenueSynergyRealizationY3: number;
  revenueSynergyRealizationY4: number;
  revenueSynergyRealizationY5: number;
  revenueSynergyMargin?: number; // BUG #5 FIX: Flow-through margin on revenue synergies (e.g., 0.50)
  
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
  
  // Purchase Price Allocation - BUG #3 FIX: Proper PPA breakdown
  targetBookValueNetAssets?: number; // Book value of net assets
  targetFairValueNetAssets?: number; // Fair value of net assets
  customerRelationships?: number; // Identified intangible: Customer relationships
  customerRelationshipsLife?: number; // Amortization years
  developedTechnology?: number; // Identified intangible: Developed technology  
  developedTechnologyLife?: number; // Amortization years
  otherIntangibles?: number; // Other identified intangibles
  otherIntangiblesLife?: number; // Amortization years
  intangibleAssets: number; // Legacy: total intangibles if not broken down
  intangibleAmortYears: number; // Legacy: average amortization period
}

const MA_PARSING_PROMPT = `You are a financial analyst expert in M&A transactions. Parse the following natural language description of a merger or acquisition and extract ALL relevant parameters.

CRITICAL RULES:
1. If the user explicitly states an EPS value (e.g., "earns $3.20 per share"), extract it as acquirerExplicitEPS. DO NOT IGNORE THIS.
2. Revenue synergies and cost synergies have DIFFERENT phase-in schedules. Extract both separately.
3. If a "flow-through margin" or "margin on revenue synergies" is mentioned, extract it as revenueSynergyMargin.
4. For Purchase Price Allocation, extract ALL components separately if provided.
5. Extract explicit transaction fees if mentioned (e.g., "forty-five million in transaction costs").

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
  "acquirerExplicitEPS": number or null (if user explicitly states EPS, use this EXACTLY - DO NOT RECALCULATE),
  "acquirerCash": number (in millions, acquirer's existing cash on balance sheet),
  "acquirerExistingDebt": number (in millions, acquirer's existing debt),
  
  "targetRevenue": number (in millions),
  "targetRevenueGrowth": [y1, y2, y3, y4, y5] (as decimals),
  "targetEBITDAMargin": number (as decimal),
  "targetDAPercent": number (as decimal),
  "targetInterestExpense": number (in millions),
  "targetTaxRate": number (as decimal),
  "targetNetDebt": number (in millions),
  
  "purchasePrice": number (equity value in millions),
  "cashPercent": number (as decimal, e.g., 0.6 for 60% cash),
  "stockPercent": number (as decimal, e.g., 0.4 for 40% stock),
  "premium": number (as decimal, e.g., 0.30 for 30% premium),
  "transactionFeePercent": number (as decimal for % of EV, or null if explicit fees given),
  "transactionFees": number or null (explicit transaction fees in millions if mentioned),
  
  "cashFromBalance": number (in millions - cash used from acquirer's balance sheet),
  "newDebtAmount": number (in millions - new debt raised for the transaction),
  "newDebtRate": number (as decimal - total interest rate e.g., 0.075 for 7.5%),
  "debtAmortizationRate": number (as decimal - annual amortization e.g., 0.05 for 5%),
  "debtMaturityYears": number (years until bullet maturity),
  
  "revenueSynergies": number (annual run-rate in millions),
  "revenueSynergyRealizationY1": number (as decimal),
  "revenueSynergyRealizationY2": number (as decimal),
  "revenueSynergyRealizationY3": number (as decimal),
  "revenueSynergyRealizationY4": number (as decimal),
  "revenueSynergyRealizationY5": number (as decimal),
  "revenueSynergyMargin": number or null (flow-through margin as decimal, e.g., 0.50 for 50%),
  
  "costSynergies": number (annual EBITDA improvement in millions),
  "costSynergyRealizationY1": number (as decimal),
  "costSynergyRealizationY2": number (as decimal),
  "costSynergyRealizationY3": number (as decimal),
  "costSynergyRealizationY4": number (as decimal),
  "costSynergyRealizationY5": number (as decimal),
  
  "integrationCostsY1": number (in millions),
  "integrationCostsY2": number (in millions),
  "integrationCostsY3": number (in millions),
  
  "targetBookValueNetAssets": number or null (book value of target's net assets in millions),
  "targetFairValueNetAssets": number or null (fair value of target's net assets in millions),
  "customerRelationships": number or null (customer relationships intangible value in millions),
  "customerRelationshipsLife": number or null (amortization period in years),
  "developedTechnology": number or null (developed technology intangible value in millions),
  "developedTechnologyLife": number or null (amortization period in years),
  "otherIntangibles": number or null (other identified intangibles in millions),
  "otherIntangiblesLife": number or null (amortization period in years),
  "intangibleAssets": number (total intangibles - sum of above or standalone if not broken down),
  "intangibleAmortYears": number (weighted average amortization period)
}

DEFAULTS (use if not explicitly stated):
- Revenue growth: 3-8% annually
- EBITDA margins: 15-25%
- D&A: 3-5% of revenue
- Tax rate: 21-25%
- Cash/stock mix: 50/50 if not specified
- Premium: 20-40% for public targets
- Transaction fees: 2.5% of EV if not specified
- REVENUE synergy phase-in: Extract EXACT percentages if given, else use 0% Y1, 50% Y2, 100% Y3+
- COST synergy phase-in: Extract EXACT percentages if given, else use 20% Y1, 60% Y2, 100% Y3+
- Revenue synergy margin: 1.0 (100%) if not specified - meaning full EBITDA pass-through
- Integration costs: as specified or 2-3x annual synergies spread over 2-3 years
- Debt amortization: 5% annually if not specified
- Debt maturity: 5 years if not specified

CRITICAL EXTRACTION RULES:
1. "earns X dollars per share" → acquirerExplicitEPS = X
2. "X percent flow-through margin on revenue synergies" → revenueSynergyMargin = X/100
3. "transaction costs/fees of X million" → transactionFees = X
4. "book value of X" for target assets → targetBookValueNetAssets = X
5. "fair value of X" for target assets → targetFairValueNetAssets = X
6. "customer relationships valued at X with Y-year amortization" → customerRelationships = X, customerRelationshipsLife = Y
7. "developed technology at X with Y-year amortization" → developedTechnology = X, developedTechnologyLife = Y

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
    acquirerExplicitEPS, // BUG #1 FIX
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
    transactionFees: explicitTransactionFees, // BUG #2 FIX
    cashFromBalance,
    newDebtAmount,
    newDebtRate,
    debtAmortizationRate,
    debtMaturityYears,
    revenueSynergies,
    revenueSynergyRealizationY1,
    revenueSynergyRealizationY2,
    revenueSynergyRealizationY3,
    revenueSynergyRealizationY4,
    revenueSynergyRealizationY5,
    revenueSynergyMargin, // BUG #5 FIX
    costSynergies,
    costSynergyRealizationY1,
    costSynergyRealizationY2,
    costSynergyRealizationY3,
    costSynergyRealizationY4,
    costSynergyRealizationY5,
    integrationCostsY1,
    integrationCostsY2,
    integrationCostsY3,
    // BUG #3 FIX: Proper PPA components
    targetBookValueNetAssets,
    targetFairValueNetAssets,
    customerRelationships,
    customerRelationshipsLife,
    developedTechnology,
    developedTechnologyLife,
    otherIntangibles,
    otherIntangiblesLife,
    intangibleAssets,
    intangibleAmortYears,
  } = assumptions;

  const years = [0, 1, 2, 3, 4, 5];
  
  // ============ ACQUIRER STANDALONE PROJECTIONS ============
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
    
    // BUG #1 FIX: Use explicit EPS if provided, otherwise calculate
    if (acquirerExplicitEPS !== undefined && acquirerExplicitEPS !== null && i === 0) {
      // Year 0: Use the explicit EPS the user provided
      acquirerEPS.push(acquirerExplicitEPS);
    } else if (acquirerExplicitEPS !== undefined && acquirerExplicitEPS !== null && i > 0) {
      // For projections, grow from the explicit base EPS proportionally
      const baseGrowth = acquirerNetIncome[i] / acquirerNetIncome[0];
      acquirerEPS.push(acquirerExplicitEPS * baseGrowth);
    } else {
      // No explicit EPS: calculate from Net Income / Shares
      acquirerEPS.push(netIncome / acquirerSharesOutstanding);
    }
  }

  // ============ TARGET STANDALONE PROJECTIONS ============
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

  // ============ TRANSACTION METRICS ============
  const cashConsideration = purchasePrice * cashPercent;
  const stockConsideration = purchasePrice * stockPercent;
  const newSharesIssued = stockConsideration / acquirerStockPrice;
  const proFormaShares = acquirerSharesOutstanding + newSharesIssued;
  const enterpriseValue = purchasePrice + targetNetDebt;
  const evEbitdaMultiple = enterpriseValue / (targetRevenue * targetEBITDAMargin);
  
  // BUG #2 FIX: Calculate transaction fees - use explicit if provided, otherwise calculate from EV
  const transactionFees = explicitTransactionFees !== undefined && explicitTransactionFees !== null 
    ? explicitTransactionFees 
    : enterpriseValue * (transactionFeePercent || 0.025);

  // BUG #3 FIX: Proper Goodwill and PPA calculation
  // Calculate identified intangibles from breakdown if available
  const identifiedCustomerRel = customerRelationships || 0;
  const identifiedDevTech = developedTechnology || 0;
  const identifiedOther = otherIntangibles || 0;
  const totalIdentifiedIntangibles = identifiedCustomerRel + identifiedDevTech + identifiedOther || intangibleAssets;
  
  // Use fair value of net assets if provided, otherwise estimate
  const fairValueNetAssets = targetFairValueNetAssets !== undefined && targetFairValueNetAssets !== null
    ? targetFairValueNetAssets
    : (targetBookValueNetAssets || targetRevenue * 0.3); // Fallback to book value or estimate
  
  // CORRECT Goodwill formula: Purchase Price - Fair Value Net Assets - Identified Intangibles
  const goodwill = Math.max(0, purchasePrice - fairValueNetAssets - totalIdentifiedIntangibles);
  
  // Calculate annual intangible amortization from breakdown if available
  const customerRelAmort = identifiedCustomerRel / (customerRelationshipsLife || 10);
  const devTechAmort = identifiedDevTech / (developedTechnologyLife || 5);
  const otherIntangAmort = identifiedOther / (otherIntangiblesLife || 10);
  const totalAnnualIntangibleAmort = customerRelAmort + devTechAmort + otherIntangAmort || (intangibleAssets / intangibleAmortYears);

  // ============ SYNERGIES ============
  // BUG #5 FIX: Apply revenue synergy flow-through margin
  const revSynergyFlowThrough = revenueSynergyMargin !== undefined && revenueSynergyMargin !== null 
    ? revenueSynergyMargin 
    : 1.0; // Default 100% if not specified
  
  const revenueSynergyRealization = [
    0, 
    revenueSynergyRealizationY1 || 0, 
    revenueSynergyRealizationY2 || 0.50, 
    revenueSynergyRealizationY3 || 1.0, 
    revenueSynergyRealizationY4 || 1.0, 
    revenueSynergyRealizationY5 || 1.0
  ];
  const costSynergyRealization = [
    0, 
    costSynergyRealizationY1 || 0.20, 
    costSynergyRealizationY2 || 0.60, 
    costSynergyRealizationY3 || 1.0, 
    costSynergyRealizationY4 || 1.0, 
    costSynergyRealizationY5 || 1.0
  ];
  
  // Revenue synergies by year (top-line)
  const revSynergiesByYear = revenueSynergyRealization.map(r => revenueSynergies * r);
  // Revenue synergy EBITDA impact (after applying margin) - BUG #5 FIX
  const revSynergyEBITDAByYear = revSynergiesByYear.map(r => r * revSynergyFlowThrough);
  // Cost synergies by year (direct EBITDA impact)
  const costSynergiesByYear = costSynergyRealization.map(r => costSynergies * r);
  // Total EBITDA synergies = Revenue Synergy EBITDA + Cost Synergies
  const totalEBITDASynergiesByYear = revSynergyEBITDAByYear.map((r, i) => r + costSynergiesByYear[i]);
  
  const integrationCosts = [0, integrationCostsY1 || 0, integrationCostsY2 || 0, integrationCostsY3 || 0, 0, 0];

  // ============ DEBT SCHEDULE (BUG #4 & #6 FIX) ============
  const debtAmortRate = debtAmortizationRate || 0.05;
  const maturityYears = debtMaturityYears || 5;
  const debtSchedule = {
    beginningBalance: [0, newDebtAmount, 0, 0, 0, 0, 0],
    mandatoryAmort: [0, 0, 0, 0, 0, 0],
    optionalPrepay: [0, 0, 0, 0, 0, 0],
    endingBalance: [newDebtAmount, 0, 0, 0, 0, 0],
    interestExpense: [0, 0, 0, 0, 0, 0],
  };
  
  for (let i = 1; i <= 5; i++) {
    debtSchedule.beginningBalance[i] = debtSchedule.endingBalance[i - 1];
    // Fixed: Amortize based on beginning balance, not original principal
    debtSchedule.mandatoryAmort[i] = Math.min(
      debtSchedule.beginningBalance[i] * debtAmortRate,
      debtSchedule.beginningBalance[i] // Can't amortize more than remaining balance
    );
    debtSchedule.endingBalance[i] = Math.max(0, debtSchedule.beginningBalance[i] - debtSchedule.mandatoryAmort[i]);
    // BUG #6 FIX: Calculate interest expense on average balance
    debtSchedule.interestExpense[i] = ((debtSchedule.beginningBalance[i] + debtSchedule.endingBalance[i]) / 2) * newDebtRate;
  }

  // ============ PRO FORMA COMBINED PROJECTIONS ============
  const proFormaRevenue: number[] = [];
  const proFormaEBITDA: number[] = [];
  const proFormaNetIncome: number[] = [];
  const proFormaEPS: number[] = [];
  const accretionDilution: number[] = [];
  const accretionDilutionPercent: number[] = [];

  for (let i = 0; i <= 5; i++) {
    // Revenue (includes revenue synergies as top-line)
    const combinedRev = acquirerRev[i] + targetRev[i] + revSynergiesByYear[i];
    proFormaRevenue.push(combinedRev);
    
    // EBITDA - BUG #5 FIX: Use EBITDA synergies (not raw revenue synergies)
    const combinedEBITDA = acquirerEBITDA[i] + targetEBITDA[i] + totalEBITDASynergiesByYear[i];
    proFormaEBITDA.push(combinedEBITDA);
    
    // D&A (combined + PPA intangible amortization)
    const combinedDA = (acquirerRev[i] * acquirerDAPercent) + (targetRev[i] * targetDAPercent) + (i > 0 ? totalAnnualIntangibleAmort : 0);
    
    // EBIT
    const ebit = combinedEBITDA - combinedDA;
    
    // BUG #6 FIX: Interest (existing + NEW debt interest from schedule)
    const newDebtInterest = i > 0 ? debtSchedule.interestExpense[i] : 0;
    const combinedInterest = acquirerInterestExpense + targetInterestExpense + newDebtInterest;
    
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

  // ============ SOURCES AND USES (BUG #2 FIX - FINAL CORRECTED) ============
  // Standard M&A Sources & Uses: Sources MUST equal Uses exactly
  // 
  // USES = Equity Value + Debt Payoff (if any) + Transaction Fees
  // SOURCES = Stock + Net Cash from Target + Cash from BS + New Debt
  //
  // Logic: Fill sources in priority order until uses are fully funded
  // 1. Stock consideration (fixed %, but capped at total uses)
  // 2. Net cash from target (if any, capped at remaining)
  // 3. Cash from balance sheet (user input, capped at remaining)
  // 4. New debt (whatever is left to balance)
  
  // Calculate what needs to be funded (positive debt means payoff needed)
  const debtPayoffAmount = Math.max(0, targetNetDebt);
  const rawNetCashFromTarget = targetNetDebt < 0 ? Math.abs(targetNetDebt) : 0;
  
  // Total Uses - this is the fixed target
  const grossUsesAmount = purchasePrice + debtPayoffAmount + transactionFees;
  
  // Step 1: Stock consideration (can't exceed total uses)
  const finalStockConsideration = Math.min(stockConsideration, grossUsesAmount);
  let remainingToFund = grossUsesAmount - finalStockConsideration;
  
  // Step 2: Net cash from target (can't exceed remaining)
  const netCashApplied = Math.min(rawNetCashFromTarget, remainingToFund);
  remainingToFund -= netCashApplied;
  
  // Step 3: Cash from balance sheet (user input, can't exceed remaining)
  const userCashFromBS = Math.max(0, cashFromBalance || 0);
  const finalCashFromBS = Math.min(userCashFromBS, remainingToFund);
  remainingToFund -= finalCashFromBS;
  
  // Step 4: New debt covers whatever is left
  const finalNewDebt = Math.max(0, remainingToFund);
  
  // Sources and Uses now GUARANTEED to balance
  const sourcesTotal = finalStockConsideration + netCashApplied + finalCashFromBS + finalNewDebt;
  const usesTotal = grossUsesAmount;
  
  // Final Sources
  const sources = {
    cashFromBalance: finalCashFromBS,
    newDebtRaised: finalNewDebt,
    stockConsideration: finalStockConsideration,
    netCashFromTarget: netCashApplied,
    total: sourcesTotal,
  };
  
  // Final Uses
  const uses = {
    equityValue: purchasePrice,
    debtPayoff: debtPayoffAmount,
    transactionFees,
    total: usesTotal,
  };
  
  // Validation: Gap should ALWAYS be 0 now
  const sourcesUsesGap = sources.total - uses.total;

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
      totalIdentifiedIntangibles,
      fairValueNetAssets,
      customerRelationships: identifiedCustomerRel,
      customerRelationshipsLife: customerRelationshipsLife || 10,
      developedTechnology: identifiedDevTech,
      developedTechnologyLife: developedTechnologyLife || 5,
    },
    synergies: {
      revenueSynergies,
      costSynergies,
      totalSynergies: revenueSynergies + costSynergies,
      revSynergiesByYear,
      revSynergyEBITDAByYear,
      revenueSynergyMargin: revSynergyFlowThrough,
      costSynergiesByYear,
      totalEBITDASynergiesByYear,
      revenueSynergyRealization,
      costSynergyRealization,
      integrationCosts,
    },
    sourcesAndUses: {
      sources,
      uses,
      isBalanced: Math.abs(sourcesUsesGap) < 0.01,
      gap: sourcesUsesGap,
    },
    debtSchedule: {
      principal: newDebtAmount,
      interestRate: newDebtRate,
      amortizationRate: debtAmortRate,
      maturityYears: maturityYears,
      schedule: debtSchedule,
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
    ppa: {
      purchasePrice,
      fairValueNetAssets,
      customerRelationships: identifiedCustomerRel,
      customerRelationshipsLife: customerRelationshipsLife || 10,
      customerRelAmortization: customerRelAmort,
      developedTechnology: identifiedDevTech,
      developedTechnologyLife: developedTechnologyLife || 5,
      devTechAmortization: devTechAmort,
      otherIntangibles: identifiedOther,
      totalIdentifiedIntangibles,
      goodwill,
      totalAnnualAmortization: totalAnnualIntangibleAmort,
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

  suSheet.getCell("A5").value = "New Debt Raised";
  suSheet.getCell("B5").value = sourcesAndUses.sources.newDebtRaised;
  suSheet.getCell("B5").numFmt = currencyFormat;
  suSheet.getCell("C5").value = sourcesAndUses.sources.newDebtRaised / sourcesAndUses.sources.total;
  suSheet.getCell("C5").numFmt = percentFormat;

  suSheet.getCell("A6").value = "Stock Consideration";
  suSheet.getCell("B6").value = sourcesAndUses.sources.stockConsideration;
  suSheet.getCell("B6").numFmt = currencyFormat;
  suSheet.getCell("C6").value = sourcesAndUses.sources.total > 0 ? sourcesAndUses.sources.stockConsideration / sourcesAndUses.sources.total : 0;
  suSheet.getCell("C6").numFmt = percentFormat;

  let sourceRowOffset = 7;
  // Include net cash from target if applicable (net cash acquisition)
  if (sourcesAndUses.sources.netCashFromTarget && sourcesAndUses.sources.netCashFromTarget > 0) {
    suSheet.getCell(`A${sourceRowOffset}`).value = "Net Cash from Target";
    suSheet.getCell(`B${sourceRowOffset}`).value = sourcesAndUses.sources.netCashFromTarget;
    suSheet.getCell(`B${sourceRowOffset}`).numFmt = currencyFormat;
    suSheet.getCell(`C${sourceRowOffset}`).value = sourcesAndUses.sources.netCashFromTarget / sourcesAndUses.sources.total;
    suSheet.getCell(`C${sourceRowOffset}`).numFmt = percentFormat;
    sourceRowOffset++;
  }

  suSheet.getCell(`A${sourceRowOffset}`).value = "TOTAL SOURCES";
  suSheet.getCell(`B${sourceRowOffset}`).value = sourcesAndUses.sources.total;
  suSheet.getCell(`B${sourceRowOffset}`).numFmt = currencyFormat;
  suSheet.getRow(sourceRowOffset).font = { bold: true };

  const usesStartRow = sourceRowOffset + 2;
  suSheet.getCell(`A${usesStartRow}`).value = "USES";
  suSheet.getRow(usesStartRow).font = { bold: true };

  let usesRow = usesStartRow + 1;
  suSheet.getCell(`A${usesRow}`).value = "Target Equity Value";
  suSheet.getCell(`B${usesRow}`).value = sourcesAndUses.uses.equityValue;
  suSheet.getCell(`B${usesRow}`).numFmt = currencyFormat;
  usesRow++;

  // Only show debt payoff if positive
  if (sourcesAndUses.uses.debtPayoff > 0) {
    suSheet.getCell(`A${usesRow}`).value = "Target Net Debt Payoff";
    suSheet.getCell(`B${usesRow}`).value = sourcesAndUses.uses.debtPayoff;
    suSheet.getCell(`B${usesRow}`).numFmt = currencyFormat;
    usesRow++;
  }

  suSheet.getCell(`A${usesRow}`).value = "Transaction Fees";
  suSheet.getCell(`B${usesRow}`).value = sourcesAndUses.uses.transactionFees;
  suSheet.getCell(`B${usesRow}`).numFmt = currencyFormat;
  usesRow++;

  suSheet.getCell(`A${usesRow}`).value = "TOTAL USES";
  suSheet.getCell(`B${usesRow}`).value = sourcesAndUses.uses.total;
  suSheet.getCell(`B${usesRow}`).numFmt = currencyFormat;
  suSheet.getRow(usesRow).font = { bold: true };
  const balanceCheckRow = usesRow + 2;

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

  synSheet.addRow(["Revenue Synergies (Top-Line)", ...synergies.revSynergiesByYear]);
  for (let i = 2; i <= 7; i++) synSheet.getCell(3, i).numFmt = currencyFormat;

  synSheet.getCell("A4").value = `Revenue Synergy EBITDA (${(synergies.revenueSynergyMargin * 100).toFixed(0)}% margin)`;
  synSheet.getCell("B4").value = synergies.revSynergyEBITDAByYear[0];
  synSheet.getCell("C4").value = synergies.revSynergyEBITDAByYear[1];
  synSheet.getCell("D4").value = synergies.revSynergyEBITDAByYear[2];
  synSheet.getCell("E4").value = synergies.revSynergyEBITDAByYear[3];
  synSheet.getCell("F4").value = synergies.revSynergyEBITDAByYear[4];
  synSheet.getCell("G4").value = synergies.revSynergyEBITDAByYear[5];
  for (let i = 2; i <= 7; i++) synSheet.getCell(4, i).numFmt = currencyFormat;

  synSheet.addRow(["Cost Synergies (Direct EBITDA)", ...synergies.costSynergiesByYear]);
  for (let i = 2; i <= 7; i++) synSheet.getCell(5, i).numFmt = currencyFormat;

  synSheet.addRow(["Total EBITDA Synergies", ...synergies.totalEBITDASynergiesByYear]);
  for (let i = 2; i <= 7; i++) synSheet.getCell(6, i).numFmt = currencyFormat;
  synSheet.getRow(6).font = { bold: true };

  synSheet.addRow([]);
  synSheet.addRow(["Integration Costs ($M)", ...synergies.integrationCosts]);
  for (let i = 2; i <= 7; i++) synSheet.getCell(8, i).numFmt = currencyFormat;

  synSheet.addRow([]);
  synSheet.getCell("A10").value = "PHASE-IN SCHEDULES";
  synSheet.getCell("A10").font = { bold: true };

  synSheet.addRow(["Revenue Synergy Phase-In (%)", ...synergies.revenueSynergyRealization.map((r: number) => r * 100)]);
  synSheet.getRow(11).numFmt = "0%";

  synSheet.addRow(["Cost Synergy Phase-In (%)", ...synergies.costSynergyRealization.map((r: number) => r * 100)]);
  synSheet.getRow(12).numFmt = "0%";

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

  // ============ PURCHASE PRICE ALLOCATION (BUG #4 FIX) ============
  const ppaSheet = workbook.addWorksheet("Purchase_Price_Allocation");
  ppaSheet.columns = [{ width: 35 }, { width: 18 }, { width: 18 }, { width: 18 }];

  ppaSheet.getCell("A1").value = "PURCHASE PRICE ALLOCATION";
  ppaSheet.getCell("A1").font = { bold: true, size: 14 };

  ppaSheet.getCell("A3").value = "Purchase Price (Equity Value)";
  ppaSheet.getCell("B3").value = results.ppa.purchasePrice;
  ppaSheet.getCell("B3").numFmt = currencyFormat;
  ppaSheet.getCell("B3").font = { color: { argb: "FF0000FF" } }; // Blue for inputs

  ppaSheet.getCell("A5").value = "Less: Fair Value of Identifiable Net Assets";
  ppaSheet.getCell("B5").value = -results.ppa.fairValueNetAssets;
  ppaSheet.getCell("B5").numFmt = currencyFormat;

  ppaSheet.getCell("A7").value = "Identified Intangible Assets:";
  ppaSheet.getCell("A7").font = { bold: true };

  ppaSheet.getCell("A8").value = "  Customer Relationships";
  ppaSheet.getCell("B8").value = -results.ppa.customerRelationships;
  ppaSheet.getCell("B8").numFmt = currencyFormat;
  ppaSheet.getCell("C8").value = `${results.ppa.customerRelationshipsLife} year life`;

  ppaSheet.getCell("A9").value = "  Developed Technology";
  ppaSheet.getCell("B9").value = -results.ppa.developedTechnology;
  ppaSheet.getCell("B9").numFmt = currencyFormat;
  ppaSheet.getCell("C9").value = `${results.ppa.developedTechnologyLife} year life`;

  ppaSheet.getCell("A10").value = "  Total Identified Intangibles";
  ppaSheet.getCell("B10").value = -results.ppa.totalIdentifiedIntangibles;
  ppaSheet.getCell("B10").numFmt = currencyFormat;
  ppaSheet.getRow(10).font = { bold: true };

  ppaSheet.getCell("A12").value = "Goodwill (Residual)";
  ppaSheet.getCell("B12").value = results.ppa.goodwill;
  ppaSheet.getCell("B12").numFmt = currencyFormat;
  ppaSheet.getRow(12).font = { bold: true };
  ppaSheet.getRow(12).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFFFF0C0" } };

  // Amortization Schedule
  ppaSheet.getCell("A15").value = "INTANGIBLE AMORTIZATION SCHEDULE";
  ppaSheet.getCell("A15").font = { bold: true, size: 12 };

  ppaSheet.addRow(["", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]);
  ppaSheet.getRow(16).font = { bold: true };

  const custRelAmort = results.ppa.customerRelAmortization;
  const devTechAmort = results.ppa.devTechAmortization;
  
  ppaSheet.addRow([
    "Customer Relationships",
    custRelAmort, custRelAmort, custRelAmort, custRelAmort, custRelAmort
  ]);
  for (let i = 2; i <= 6; i++) ppaSheet.getCell(17, i).numFmt = currencyFormat;

  ppaSheet.addRow([
    "Developed Technology",
    devTechAmort, devTechAmort, devTechAmort, devTechAmort, devTechAmort
  ]);
  for (let i = 2; i <= 6; i++) ppaSheet.getCell(18, i).numFmt = currencyFormat;

  const totalAmort = results.ppa.totalAnnualAmortization;
  ppaSheet.addRow([
    "Total PPA Amortization",
    totalAmort, totalAmort, totalAmort, totalAmort, totalAmort
  ]);
  for (let i = 2; i <= 6; i++) ppaSheet.getCell(19, i).numFmt = currencyFormat;
  ppaSheet.getRow(19).font = { bold: true };

  // ============ DEBT SCHEDULE (BUG #4 & #6 FIX) ============
  const debtSheet = workbook.addWorksheet("Debt_Schedule");
  debtSheet.columns = [{ width: 25 }, { width: 18 }, { width: 18 }, { width: 18 }, { width: 18 }, { width: 18 }];

  debtSheet.getCell("A1").value = "DEBT SCHEDULE";
  debtSheet.getCell("A1").font = { bold: true, size: 14 };

  debtSheet.getCell("A3").value = "New Debt Principal";
  debtSheet.getCell("B3").value = results.debtSchedule.principal;
  debtSheet.getCell("B3").numFmt = currencyFormat;
  debtSheet.getCell("B3").font = { color: { argb: "FF0000FF" } };

  debtSheet.getCell("A4").value = "Interest Rate";
  debtSheet.getCell("B4").value = results.debtSchedule.interestRate;
  debtSheet.getCell("B4").numFmt = percentFormat;
  debtSheet.getCell("B4").font = { color: { argb: "FF0000FF" } };

  debtSheet.getCell("A5").value = "Annual Amortization Rate";
  debtSheet.getCell("B5").value = results.debtSchedule.amortizationRate;
  debtSheet.getCell("B5").numFmt = percentFormat;
  debtSheet.getCell("B5").font = { color: { argb: "FF0000FF" } };

  debtSheet.getCell("A6").value = "Maturity (Years)";
  debtSheet.getCell("B6").value = results.debtSchedule.maturityYears;
  debtSheet.getCell("B6").font = { color: { argb: "FF0000FF" } };

  debtSheet.addRow([]);
  debtSheet.addRow(["YEAR", "Beginning Balance", "Mandatory Amort", "Ending Balance", "Interest Expense"]);
  debtSheet.getRow(9).font = { bold: true };
  debtSheet.getRow(9).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFE0E0E0" } };

  for (let year = 0; year <= 5; year++) {
    const sched = results.debtSchedule.schedule;
    debtSheet.addRow([
      year,
      sched.beginningBalance[year],
      sched.mandatoryAmort[year],
      sched.endingBalance[year],
      sched.interestExpense[year]
    ]);
    const row = 10 + year;
    for (let col = 2; col <= 5; col++) {
      debtSheet.getCell(row, col).numFmt = currencyFormat;
    }
  }

  // ============ BALANCE CHECK (BUG #2 FIX) ============
  suSheet.getCell(`A${balanceCheckRow}`).value = "BALANCE CHECK";
  suSheet.getCell(`A${balanceCheckRow}`).font = { bold: true };
  suSheet.getCell(`A${balanceCheckRow + 1}`).value = "Sources - Uses:";
  suSheet.getCell(`B${balanceCheckRow + 1}`).value = sourcesAndUses.gap || 0;
  suSheet.getCell(`B${balanceCheckRow + 1}`).numFmt = currencyFormat;
  suSheet.getCell(`C${balanceCheckRow + 1}`).value = sourcesAndUses.isBalanced ? "BALANCED" : "NOT BALANCED";
  suSheet.getCell(`C${balanceCheckRow + 1}`).font = { 
    bold: true,
    color: { argb: sourcesAndUses.isBalanced ? "FF008000" : "FFFF0000" } 
  };

  // ============ SENSITIVITY ANALYSIS (BUG #4 FIX) ============
  const sensSheet = workbook.addWorksheet("Sensitivity_Analysis");
  sensSheet.columns = [
    { width: 25 }, 
    { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }
  ];

  sensSheet.getCell("A1").value = "SENSITIVITY ANALYSIS - EPS ACCRETION/DILUTION";
  sensSheet.getCell("A1").font = { bold: true, size: 14 };

  // Table 1: EPS Impact vs. Synergy Realization
  sensSheet.getCell("A3").value = "Year 1 EPS Impact vs. Synergy Realization";
  sensSheet.getCell("A3").font = { bold: true, size: 12 };

  sensSheet.getCell("A4").value = "Synergy %";
  sensSheet.getCell("B4").value = "50%";
  sensSheet.getCell("C4").value = "75%";
  sensSheet.getCell("D4").value = "100%";
  sensSheet.getCell("E4").value = "125%";
  sensSheet.getRow(4).font = { bold: true };
  sensSheet.getRow(4).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFE0E0E0" } };

  // Calculate sensitivity for different synergy levels
  const baseSynergy = synergies.totalSynergies;
  const synergyMultipliers = [0.5, 0.75, 1.0, 1.25];
  const y1EpsBase = proFormaProjections.eps[1];
  const y1AcquirerEps = acquirerProjections.eps[1];
  
  sensSheet.getCell("A5").value = "Pro Forma EPS";
  synergyMultipliers.forEach((mult, idx) => {
    // Rough estimate: EPS scales with synergy changes
    const synergyDelta = baseSynergy * (mult - 1) * 0.3 / proFormaShares; // 30% flows to EPS after tax
    const adjustedEps = y1EpsBase + synergyDelta;
    sensSheet.getCell(5, idx + 2).value = adjustedEps;
    sensSheet.getCell(5, idx + 2).numFmt = epsFormat;
  });

  sensSheet.getCell("A6").value = "Accretion/(Dilution)";
  synergyMultipliers.forEach((mult, idx) => {
    const synergyDelta = baseSynergy * (mult - 1) * 0.3 / proFormaShares;
    const adjustedEps = y1EpsBase + synergyDelta;
    const accDil = (adjustedEps / y1AcquirerEps) - 1;
    sensSheet.getCell(6, idx + 2).value = accDil;
    sensSheet.getCell(6, idx + 2).numFmt = percentFormat;
    if (accDil > 0) {
      sensSheet.getCell(6, idx + 2).font = { color: { argb: "FF008000" } };
    } else if (accDil < 0) {
      sensSheet.getCell(6, idx + 2).font = { color: { argb: "FFFF0000" } };
    }
  });

  // Table 2: EPS Impact vs. Transaction Multiple
  sensSheet.getCell("A9").value = "Year 1 EPS Impact vs. EV/EBITDA Multiple";
  sensSheet.getCell("A9").font = { bold: true, size: 12 };

  const baseMultiple = transactionMetrics.evEbitdaMultiple;
  const multipleVariants = [
    baseMultiple - 2,
    baseMultiple - 1,
    baseMultiple,
    baseMultiple + 1,
    baseMultiple + 2
  ];

  sensSheet.getCell("A10").value = "EV/EBITDA";
  multipleVariants.forEach((m, idx) => {
    sensSheet.getCell(10, idx + 2).value = m;
    sensSheet.getCell(10, idx + 2).numFmt = multipleFormat;
  });
  sensSheet.getRow(10).font = { bold: true };
  sensSheet.getRow(10).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFE0E0E0" } };

  sensSheet.getCell("A11").value = "Implied Purchase Price";
  const targetEBITDA = results.targetProjections.ebitda[0];
  multipleVariants.forEach((m, idx) => {
    const impliedEV = m * targetEBITDA;
    const impliedPP = impliedEV - targetNetDebt;
    sensSheet.getCell(11, idx + 2).value = impliedPP;
    sensSheet.getCell(11, idx + 2).numFmt = currencyFormat;
  });

  sensSheet.getCell("A12").value = "Est. EPS Impact";
  multipleVariants.forEach((m, idx) => {
    // Higher multiple = higher price = more dilution
    const priceDelta = (m - baseMultiple) * targetEBITDA;
    // Extra stock issuance from higher price
    const extraShares = priceDelta / assumptions.acquirerStockPrice;
    const newProFormaShares = proFormaShares + extraShares;
    const adjustedEps = proFormaProjections.netIncome[1] / newProFormaShares;
    const accDil = (adjustedEps / y1AcquirerEps) - 1;
    sensSheet.getCell(12, idx + 2).value = accDil;
    sensSheet.getCell(12, idx + 2).numFmt = percentFormat;
    if (accDil > 0) {
      sensSheet.getCell(12, idx + 2).font = { color: { argb: "FF008000" } };
    } else if (accDil < 0) {
      sensSheet.getCell(12, idx + 2).font = { color: { argb: "FFFF0000" } };
    }
  });

  // Table 3: Integration Cost Sensitivity
  sensSheet.getCell("A15").value = "Year 1 EPS Impact vs. Integration Cost Overrun";
  sensSheet.getCell("A15").font = { bold: true, size: 12 };

  const costMultipliers = [0.75, 1.0, 1.25, 1.5, 2.0];
  
  sensSheet.getCell("A16").value = "Cost Multiplier";
  costMultipliers.forEach((m, idx) => {
    sensSheet.getCell(16, idx + 2).value = `${(m * 100).toFixed(0)}%`;
  });
  sensSheet.getRow(16).font = { bold: true };
  sensSheet.getRow(16).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFE0E0E0" } };

  const baseIntegrationY1 = synergies.integrationCosts[1] || 0;
  sensSheet.getCell("A17").value = "Integration Cost Y1";
  costMultipliers.forEach((m, idx) => {
    sensSheet.getCell(17, idx + 2).value = baseIntegrationY1 * m;
    sensSheet.getCell(17, idx + 2).numFmt = currencyFormat;
  });

  sensSheet.getCell("A18").value = "Est. EPS Impact";
  costMultipliers.forEach((m, idx) => {
    // Extra integration cost reduces net income
    const extraCost = baseIntegrationY1 * (m - 1) * (1 - assumptions.acquirerTaxRate);
    const adjustedNI = proFormaProjections.netIncome[1] - extraCost;
    const adjustedEps = adjustedNI / proFormaShares;
    const accDil = (adjustedEps / y1AcquirerEps) - 1;
    sensSheet.getCell(18, idx + 2).value = accDil;
    sensSheet.getCell(18, idx + 2).numFmt = percentFormat;
    if (accDil > 0) {
      sensSheet.getCell(18, idx + 2).font = { color: { argb: "FF008000" } };
    } else if (accDil < 0) {
      sensSheet.getCell(18, idx + 2).font = { color: { argb: "FFFF0000" } };
    }
  });

  // Key Assumptions Summary
  sensSheet.getCell("A21").value = "KEY ASSUMPTIONS (Modify in Excel)";
  sensSheet.getCell("A21").font = { bold: true, size: 12 };

  sensSheet.getCell("A22").value = "Base Synergies (Run-Rate)";
  sensSheet.getCell("B22").value = baseSynergy;
  sensSheet.getCell("B22").numFmt = currencyFormat;
  sensSheet.getCell("B22").font = { color: { argb: "FF0000FF" } };

  sensSheet.getCell("A23").value = "Base EV/EBITDA Multiple";
  sensSheet.getCell("B23").value = baseMultiple;
  sensSheet.getCell("B23").numFmt = multipleFormat;
  sensSheet.getCell("B23").font = { color: { argb: "FF0000FF" } };

  sensSheet.getCell("A24").value = "Y1 Integration Costs";
  sensSheet.getCell("B24").value = baseIntegrationY1;
  sensSheet.getCell("B24").numFmt = currencyFormat;
  sensSheet.getCell("B24").font = { color: { argb: "FF0000FF" } };

  sensSheet.getCell("A25").value = "Acquirer Stock Price";
  sensSheet.getCell("B25").value = assumptions.acquirerStockPrice;
  sensSheet.getCell("B25").numFmt = currencyFormat;
  sensSheet.getCell("B25").font = { color: { argb: "FF0000FF" } };

  // ============ CHARTS DATA TAB (BUG #4 FIX) ============
  // ExcelJS has limited native chart support, so we provide formatted data tables
  // that users can easily create charts from in Excel
  const chartSheet = workbook.addWorksheet("Charts_Data");
  chartSheet.columns = [
    { width: 20 },
    { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }, { width: 15 }
  ];

  chartSheet.getCell("A1").value = "CHART DATA - PRO FORMA FINANCIAL TRENDS";
  chartSheet.getCell("A1").font = { bold: true, size: 14 };

  // Chart 1: Revenue Trends
  chartSheet.getCell("A3").value = "Revenue ($M)";
  chartSheet.getCell("A3").font = { bold: true };
  chartSheet.addRow(["Year", 0, 1, 2, 3, 4, 5]);
  chartSheet.getRow(4).font = { bold: true };
  chartSheet.addRow(["Acquirer", ...acquirerProjections.revenue]);
  chartSheet.addRow(["Target", ...results.targetProjections.revenue]);
  chartSheet.addRow(["Pro Forma", ...proFormaProjections.revenue]);
  for (let row = 5; row <= 7; row++) {
    for (let col = 2; col <= 7; col++) {
      chartSheet.getCell(row, col).numFmt = currencyFormat;
    }
  }

  // Chart 2: EBITDA Trends
  chartSheet.getCell("A10").value = "EBITDA ($M)";
  chartSheet.getCell("A10").font = { bold: true };
  chartSheet.addRow(["Year", 0, 1, 2, 3, 4, 5]);
  chartSheet.getRow(11).font = { bold: true };
  chartSheet.addRow(["Acquirer", ...acquirerProjections.ebitda]);
  chartSheet.addRow(["Target", ...results.targetProjections.ebitda]);
  chartSheet.addRow(["Pro Forma", ...proFormaProjections.ebitda]);
  for (let row = 12; row <= 14; row++) {
    for (let col = 2; col <= 7; col++) {
      chartSheet.getCell(row, col).numFmt = currencyFormat;
    }
  }

  // Chart 3: EPS Comparison
  chartSheet.getCell("A17").value = "EPS ($)";
  chartSheet.getCell("A17").font = { bold: true };
  chartSheet.addRow(["Year", 0, 1, 2, 3, 4, 5]);
  chartSheet.getRow(18).font = { bold: true };
  chartSheet.addRow(["Acquirer Standalone", ...acquirerProjections.eps]);
  chartSheet.addRow(["Pro Forma Combined", ...proFormaProjections.eps]);
  for (let row = 19; row <= 20; row++) {
    for (let col = 2; col <= 7; col++) {
      chartSheet.getCell(row, col).numFmt = epsFormat;
    }
  }

  // Chart 4: Accretion/Dilution Bar Chart Data
  chartSheet.getCell("A23").value = "Accretion/Dilution (%)";
  chartSheet.getCell("A23").font = { bold: true };
  chartSheet.addRow(["Year", 0, 1, 2, 3, 4, 5]);
  chartSheet.getRow(24).font = { bold: true };
  chartSheet.addRow(["EPS Impact %", ...accretionDilution.percentImpact.map((v: number) => v * 100)]);
  for (let col = 2; col <= 7; col++) {
    const val = accretionDilution.percentImpact[col - 2] * 100;
    chartSheet.getCell(25, col).numFmt = "0.0%";
    if (val > 0) {
      chartSheet.getCell(25, col).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFC0FFC0" } };
    } else if (val < 0) {
      chartSheet.getCell(25, col).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFFFC0C0" } };
    }
  }

  // Chart 5: Synergy Contribution
  chartSheet.getCell("A28").value = "Synergy Contribution to EBITDA ($M)";
  chartSheet.getCell("A28").font = { bold: true };
  chartSheet.addRow(["Year", 0, 1, 2, 3, 4, 5]);
  chartSheet.getRow(29).font = { bold: true };
  chartSheet.addRow(["Revenue Synergy EBITDA", ...synergies.revSynergyEBITDAByYear]);
  chartSheet.addRow(["Cost Synergies", ...synergies.costSynergiesByYear]);
  chartSheet.addRow(["Total EBITDA Synergies", ...synergies.totalEBITDASynergiesByYear]);
  for (let row = 30; row <= 32; row++) {
    for (let col = 2; col <= 7; col++) {
      chartSheet.getCell(row, col).numFmt = currencyFormat;
    }
  }
  chartSheet.getRow(32).font = { bold: true };

  // Instructions for users
  chartSheet.getCell("A35").value = "HOW TO CREATE CHARTS:";
  chartSheet.getCell("A35").font = { bold: true, size: 12 };
  chartSheet.getCell("A36").value = "1. Select the data range (e.g., A4:G7 for Revenue)";
  chartSheet.getCell("A37").value = "2. Insert > Chart > Select chart type (Line, Bar, etc.)";
  chartSheet.getCell("A38").value = "3. Customize as needed with titles and formatting";

  const buffer = await workbook.xlsx.writeBuffer();
  return Buffer.from(buffer);
}
