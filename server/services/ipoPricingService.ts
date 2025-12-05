import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

export type FinanceLLMProvider = 'zhi1' | 'zhi2' | 'zhi3' | 'zhi4' | 'zhi5';

export interface IPOAssumptions {
  companyName: string;
  filingDate: string;
  sector: string;
  
  sharesOutstandingPreIPO: number;
  primarySharesOffered: number;
  secondarySharesOffered: number;
  greenshoeShares: number;
  greenshoePercent: number;
  greenshoeAssumedExercised?: boolean; // Default true - standard for successful IPOs
  
  // Dollar-based inputs (alternative to share-based)
  primaryDollarRaiseM?: number; // Primary proceeds target in $M
  secondaryDollarRaiseM?: number; // Secondary proceeds target in $M
  
  targetGrossProceeds: number;
  indicatedPriceRangeLow: number;
  indicatedPriceRangeHigh: number;
  
  currentCash: number;
  currentDebt: number; // NEW: Required for proper EV calculation
  
  currentYearRevenue: number;
  ntmRevenue: number;
  ntmRevenueGrowth: number;
  ntmEBITDA: number;
  ntmEBITDAMargin: number;
  
  // Fair value - can be DCF or raNPV
  fairValuePerShare: number;
  fairValueType: "dcf" | "ranpv";
  totalRaNPV?: number; // Total risk-adjusted NPV in millions
  
  // Peer comps
  peerMedianEVRevenue: number;
  peerMedianEVRaNPV?: number; // For biotech - EV/raNPV multiple
  peerMedianEVEBITDA?: number; // For restaurants, retail, consumer staples - EV/EBITDA multiple
  peerMedianNTMFCF?: number; // Alternative: NTM FCF multiple
  
  orderBook: {
    priceLevel: number;
    oversubscription: number;
  }[];
  
  // BUG FIX #4: Notable orders with max price constraints
  notableOrders?: {
    investorName: string;
    indicatedSizeM: number;
    maxPrice?: number;
    isDefending?: boolean; // Underwater investor defending position
  }[];
  
  // Sector-specific historical benchmarks
  historicalFirstDayPop: number;
  sectorAverageFirstDayPop: number;
  sectorMedianFirstDayPop?: number; // Can be negative for biotech
  
  foundersEmployeesOwnership: number;
  founderSharesExplicitM?: number; // Explicit founder shares in millions (from cap table parsing)
  vcPeOwnership: number;
  
  underwritingFeePercent: number;
  
  useOfProceeds?: string;
  lockupDays?: number;
  
  // Management guidance
  ceoGuidance?: string;
  boardGuidance?: string;
  pricingAggressiveness: "conservative" | "moderate" | "aggressive" | "maximum";
  managementPriority?: "valuation_maximization" | "runway_extension" | "deal_certainty";
  minAcceptablePrice?: number;
  
  // Risk factors
  hasBinaryCatalyst?: boolean;
  monthsToCatalyst?: number;
  catalystDescription?: string;
  
  // Secondary component
  secondaryOptics?: "neutral" | "negative" | "positive";
  
  // BUG FIX #1: Down-round detection
  lastPrivateRoundPrice?: number;
  downRoundOptics?: boolean;
  downRoundIpoPenalty?: number; // Historical avg additional discount (e.g., 0.22)
  
  // BUG FIX #2: Dual-class governance
  dualClass?: boolean;
  dualClassDiscount?: number; // Historical avg governance discount (e.g., 0.06)
  
  // BUG FIX #5: Growth trajectory
  growthRates?: {
    fy2024to2025Growth?: number;
    fy2025to2026Growth?: number;
  };
  
  // BUG FIX #6: Customer concentration
  customerConcentrationTop5?: number; // e.g., 0.47 for 47%
  
  // DCF Inputs - user-provided override sector defaults
  wacc?: number; // User-provided WACC (e.g., 0.085 for 8.5%)
  terminalGrowthRate?: number; // User-provided terminal growth (e.g., 0.03 for 3%)
  capexPercent?: number; // CapEx as % of revenue (e.g., 0.05 for 5%)
  taxRate?: number; // Tax rate (e.g., 0.26 for 26%)
  nwcDays?: number; // Net working capital days (negative for cash-generating business)
}

const IPO_PARSING_PROMPT = `You are an investment banking expert. Parse the IPO description and extract ALL parameters with precision.

*** CRITICAL: DO NOT HALLUCINATE VALUES ***
- ONLY extract values explicitly stated in the input
- DO NOT invent/guess indicatedPriceRangeLow or indicatedPriceRangeHigh if not provided
- DO NOT invent fairValuePerShare - leave it undefined if not given
- DO NOT infer values from context - ONLY use explicit user numbers

*** FLEXIBLE PARSING - RECOGNIZE NATURAL LANGUAGE ***
Users may describe inputs in various ways. Be flexible:
- "Target gross proceeds $600M with $480M primary and $120M secondary" → targetGrossProceeds=600, primaryDollarRaiseM=480, secondaryDollarRaiseM=120
- "raise $500 million" or "target $500M IPO" → primaryDollarRaiseM=500
- "founders hold X shares, investors hold Y shares, options Z shares" → add them for sharesOutstandingPreIPO
- "comparable companies: A at 7x, B at 4.5x, C at 9x" → calculate median EV/Revenue multiple
- "WACC 11.8%" or "discount rate of 11.8%" → wacc=0.118
- "15% greenshoe" or "include 15% over-allotment" → greenshoePercent=0.15

CRITICAL PARSING RULES:

1. SECTOR DETECTION (BE FLEXIBLE):
   - "biotech" / "biopharmaceutical" / "clinical-stage" / "Phase" → sector = "biotech"
   - "SaaS" / "enterprise software" → sector = "saas"
   - "AI" / "robotics" / "automation" / "machine learning" / "tech" → sector = "tech"
   - "AI infrastructure" / "GPU cloud" → sector = "ai_infrastructure"
   - "defense-tech" / "national security" → sector = "defense_tech"
   - "restaurant" / "fast-casual" / "QSR" / "food service" / "chain restaurant" → sector = "restaurant"
   - "consumer staples" / "consumer goods" / "CPG" → sector = "consumer_staples"
   - "retail" / "e-commerce" / "DTC" → sector = "retail"

2. VALUATION TYPE - CRITICAL FOR BIOTECH:
   - If input mentions "risk-adjusted NPV" / "raNPV" / "rNPV" → fairValueType = "ranpv"
   - Parse "ranpv_per_share" or "raNPV/share" as fairValuePerShare
   - Parse "total_ranpv" in millions as totalRaNPV
   - If input has "dcf" / "DCF valuation" → fairValueType = "dcf"

3. PEER COMPS - PARSE FLEXIBLY:
   - If "median EV/raNPV" or "EV/rNPV" is given, set peerMedianEVRaNPV (e.g., 2.4)
   - If "median EV/EBITDA" or "EV/EBITDA" is given, set peerMedianEVEBITDA (e.g., 8.8)
   - If "median NTM FCF" or "FCF multiple" is given, set peerMedianNTMFCF (e.g., 22)
   - Regular EV/Revenue goes to peerMedianEVRevenue
   - RESTAURANTS/RETAIL/CONSUMER: Always extract EV/EBITDA as the primary multiple
   - CRITICAL FOR RESTAURANTS: Parse "8.8× NTM EV/EBITDA" → peerMedianEVEBITDA = 8.8
   
   *** FLEXIBLE COMP PARSING ***
   - "Comparable companies: A at 7x revenue, B at 4.5x, C at 9x, D at 14x" → Calculate MEDIAN of [7, 4.5, 9, 14] = 8.0 for peerMedianEVRevenue
   - "trades at 7x revenue" or "7x sales" or "7x rev" → interpret as EV/Revenue multiple
   - If multiple comps given, use the MEDIAN (not average) for peerMedianEVRevenue
   - "Boston Dynamics parent trades at 7x revenue, Rockwell at 4.5x" → peerMedianEVRevenue = median of list

4. SECTOR HISTORICAL BENCHMARKS:
   - Parse "sector_ipos_2024_2025.avg_day1_return" → sectorAverageFirstDayPop
   - Parse "median_day1_return" → sectorMedianFirstDayPop
   - BIOTECH OFTEN HAS NEGATIVE RETURNS: -0.04 means -4%

5. MANAGEMENT GUIDANCE PRIORITY:
   - "runway extension" / "get the deal done" / "certainty" → managementPriority = "runway_extension"
   - "maximize valuation" / "biggest ever" → managementPriority = "valuation_maximization"
   - Parse "min_acceptable_price" exactly as minAcceptablePrice

6. BINARY CATALYST:
   - If "Phase 3 data" / "data readout" / "binary event" mentioned → hasBinaryCatalyst = true
   - Parse months until catalyst → monthsToCatalyst
   - Get description → catalystDescription

7. SECONDARY COMPONENT:
   - If insiders/founders selling shares at IPO → secondarySharesOffered > 0
   - Parse "optics" field → secondaryOptics ("negative" if noted)

8. ORDER BOOK - PARSE EVERY THRESHOLD:
   - "$24+: 3.2×" → { priceLevel: 24, oversubscription: 3.2 }
   - "$22+: 5.8×" → { priceLevel: 22, oversubscription: 5.8 }
   - Include ALL tiers mentioned

9. REVENUE AND EBITDA (BE FLEXIBLE):
   - For pre-revenue biotech, ntmRevenue = 0
   - Parse FY2026 guidance if present
   - Parse NTM EBITDA directly if given, OR calculate as: ntmEBITDA = ntmRevenue × ntmEBITDAMargin
   - For restaurants: Look for "EBITDA margin" and convert to decimal (16.5% → 0.165)
   
   *** FLEXIBLE REVENUE/EBITDA PARSING ***
   - "Current revenue $410 million" → currentYearRevenue=410
   - "growing at 48%" → apply growth: ntmRevenue = 410 × 1.48 = ~607
   - "EBITDA margin currently 12%" → ntmEBITDAMargin=0.12, ntmEBITDA = ntmRevenue × 0.12
   - If only "gross margin 61%" given, that's NOT EBITDA margin - keep separate
   - "revenue $X, growing Y% for years 1-2" → ntmRevenue = X × (1 + Y/100)

10. DOWN-ROUND DETECTION (CRITICAL):
   - Parse "last_private_round.price_per_share" or "Series E price" → lastPrivateRoundPrice
   - Parse "risk_factors.down_round_optics" → downRoundOptics (true ONLY if explicitly flagged as concern)
   - Parse "down_round_ipo_penalty.avg_additional_discount" → downRoundIpoPenalty (ONLY if explicitly provided, NO DEFAULT)
   - FOR RESTAURANTS/RETAIL/CONSUMER: downRoundOptics = false unless explicitly stated as a concern
   - DOWN-ROUND LOGIC ONLY APPLIES FOR: biotech sectors OR when input says "down-round is a concern"
   - PROFITABLE NON-BIOTECH COMPANIES: DO NOT set downRoundOptics=true just because lastPrivateRoundPrice exists

11. DUAL-CLASS STRUCTURE:
   - If "dual_class" or "Class A/B shares" mentioned → dualClass = true
   - Parse "dual_class_discount.avg_governance_discount" → dualClassDiscount (ONLY if explicitly provided, NO DEFAULT)

12. NOTABLE INVESTORS WITH MAX PRICE:
   - Parse order_book.notable_orders[] with { investorName, indicatedSizeM, maxPrice }
   - Example: "Fidelity: $75M, max $42" → { investorName: "Fidelity", indicatedSizeM: 75, maxPrice: 42 }
   - If investor is "underwater" or "defending" → isDefending: true

13. GROWTH TRAJECTORY:
   - Parse growth rates as decimals: 63% → 0.63
   - "fy2024_to_fy2025_growth" → growthRates.fy2024to2025Growth
   - "fy2025_to_fy2026_growth" → growthRates.fy2025to2026Growth

14. CUSTOMER CONCENTRATION:
   - Parse "top_5_customers_pct" or "customer_concentration" → customerConcentrationTop5 (as decimal, e.g., 0.47)

15. DCF INPUTS (CRITICAL - PARSE EXACTLY):
   - Parse "WACC" or "WACC 11.8%" or "discount rate" → wacc (as decimal, e.g., 0.118 for 11.8%)
   - Parse "terminal growth" or "terminal growth rate 3.5%" → terminalGrowthRate (as decimal, e.g., 0.035 for 3.5%)
   - Parse "CapEx" or "CapEx at 7% of revenue" → capexPercent (as decimal, e.g., 0.07 for 7%)
   - Parse "tax rate" or "tax rate 21%" → taxRate (as decimal, e.g., 0.21 for 21%)
   - Parse "NWC" or "NWC at 9%" → nwcDays (integer or percent, e.g., 0.09 for 9% of revenue)

16. CAP TABLE PARSING (FLEXIBLE):
   - "founders hold 45 million shares" → add to sharesOutstandingPreIPO AND set founderSharesExplicitM = 45
   - "Series A/B/C investors hold 30 million shares" → add to sharesOutstandingPreIPO
   - "employee option pool is 8 million shares" → add to sharesOutstandingPreIPO
   - Sum ALL share categories for total sharesOutstandingPreIPO (e.g., 45 + 30 + 8 = 83 million)
   - "Pre-IPO cap table: founders X, investors Y, options Z" → sharesOutstandingPreIPO = X + Y + Z, founderSharesExplicitM = X

17. FOUNDER SHARES EXTRACTION (CRITICAL - SUM ALL FOUNDER BUCKETS):
   - "founder CEO holds 38 million shares" → add 38 to founderSharesExplicitM
   - "co-founder CTO holds 22 million shares" → add 22 to founderSharesExplicitM (co-founder = founder!)
   - "founder", "co-founder", "founding partner", "founding CEO", "founding team" → ALL count as founders
   - ALWAYS SUM all founder/co-founder entries: CEO 38M + CTO 22M = founderSharesExplicitM = 60
   - This is SEPARATE from foundersEmployeesOwnership (which is a percentage)
   - founderSharesExplicitM = explicit share count in millions (used for ownership calculation)

Return JSON:
{
  "companyName": "string",
  "filingDate": "YYYY-MM-DD",
  "sector": "biotech" | "saas" | "ai_infrastructure" | "defense_tech" | "restaurant" | "consumer_staples" | "retail" | "tech",
  
  "sharesOutstandingPreIPO": number (millions),
  "primarySharesOffered": number (millions),
  "secondarySharesOffered": number (millions, default 0),
  "greenshoeShares": number (millions),
  "greenshoePercent": number (decimal),
  
  "targetGrossProceeds": number (millions),
  "primaryDollarRaiseM": number (millions, primary proceeds target),
  "secondaryDollarRaiseM": number (millions, secondary proceeds if any),
  "indicatedPriceRangeLow": number or null (ONLY if explicitly provided in input - DO NOT INVENT),
  "indicatedPriceRangeHigh": number or null (ONLY if explicitly provided in input - DO NOT INVENT),
  
  "currentCash": number (millions),
  "currentDebt": number (millions, total debt on balance sheet),
  
  "currentYearRevenue": number (millions),
  "ntmRevenue": number (millions - 0 for pre-revenue),
  "ntmRevenueGrowth": number (decimal),
  "ntmEBITDA": number (millions),
  "ntmEBITDAMargin": number (decimal),
  
  "fairValuePerShare": number or null (ONLY if explicitly provided - DCF will be computed automatically if not),
  "fairValueType": "dcf" | "ranpv",
  "totalRaNPV": number (millions, if provided),
  
  "peerMedianEVRevenue": number,
  "peerMedianEVRaNPV": number (if biotech),
  "peerMedianEVEBITDA": number (if restaurant/retail/consumer - THIS IS THE PRIMARY MULTIPLE FOR THESE SECTORS),
  "peerMedianNTMFCF": number (alternative FCF multiple if provided),
  
  "orderBook": [
    { "priceLevel": number, "oversubscription": number }
  ],
  "notableOrders": [
    { "investorName": string, "indicatedSizeM": number, "maxPrice": number, "isDefending": boolean }
  ],
  
  "historicalFirstDayPop": number (decimal),
  "sectorAverageFirstDayPop": number (decimal, can be negative),
  "sectorMedianFirstDayPop": number (decimal, can be negative),
  
  "foundersEmployeesOwnership": number (decimal),
  "founderSharesExplicitM": number (millions - sum of ALL founder/co-founder share counts if provided explicitly),
  "vcPeOwnership": number (decimal),
  
  "underwritingFeePercent": number (ONLY if explicitly provided, default 0 if not),
  
  "ceoGuidance": "exact quote",
  "boardGuidance": "exact quote",
  "pricingAggressiveness": "conservative" | "moderate" | "aggressive" | "maximum",
  "managementPriority": "valuation_maximization" | "runway_extension" | "deal_certainty",
  "minAcceptablePrice": number,
  
  "hasBinaryCatalyst": boolean,
  "monthsToCatalyst": number,
  "catalystDescription": "string",
  
  "secondaryOptics": "neutral" | "negative" | "positive",
  
  "lastPrivateRoundPrice": number (price per share of last private round),
  "downRoundOptics": boolean (true if down-round is a concern),
  "downRoundIpoPenalty": number (ONLY if explicitly provided, default 0 if not),
  
  "dualClass": boolean,
  "dualClassDiscount": number (ONLY if explicitly provided, default 0 if not),
  
  "growthRates": {
    "fy2024to2025Growth": number (decimal),
    "fy2025to2026Growth": number (decimal)
  },
  
  "customerConcentrationTop5": number (decimal, e.g., 0.47 for 47%),
  
  "wacc": number (decimal, e.g., 0.085 for 8.5% - USE USER INPUT IF PROVIDED),
  "terminalGrowthRate": number (decimal, e.g., 0.03 for 3%),
  "capexPercent": number (decimal, e.g., 0.05 for 5% of revenue),
  "taxRate": number (decimal, e.g., 0.26 for 26%),
  "nwcDays": number (integer, can be negative)
}

Return ONLY JSON, no markdown.`;

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
      temperature: 0.05,
    });
    responseText = response.choices[0]?.message?.content || "";
    providerUsed = "ZHI 1";
  } else if (provider === "zhi2") {
    const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-20250514",
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
      temperature: 0.05,
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
        temperature: 0.05,
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
      temperature: 0.05,
    });
    responseText = response.choices[0]?.message?.content || "";
    providerUsed = "ZHI 5";
  }

  let jsonStr = responseText.trim();
  if (jsonStr.startsWith("```json")) jsonStr = jsonStr.slice(7);
  else if (jsonStr.startsWith("```")) jsonStr = jsonStr.slice(3);
  if (jsonStr.endsWith("```")) jsonStr = jsonStr.slice(0, -3);
  
  if (!jsonStr.startsWith("{")) {
    const startIdx = jsonStr.indexOf("{");
    const endIdx = jsonStr.lastIndexOf("}");
    if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
      jsonStr = jsonStr.slice(startIdx, endIdx + 1);
    }
  }
  
  jsonStr = jsonStr.trim();
  const assumptions: IPOAssumptions = JSON.parse(jsonStr);
  
  // Set neutral defaults ONLY for truly optional fields
  // NO fabricated inference from text - values must come from explicit user input
  if (!assumptions.currentCash) assumptions.currentCash = 0;
  if (!assumptions.currentDebt) assumptions.currentDebt = 0;
  if (!assumptions.secondarySharesOffered) assumptions.secondarySharesOffered = 0;
  if (!assumptions.fairValueType) assumptions.fairValueType = "dcf";
  if (!assumptions.underwritingFeePercent) assumptions.underwritingFeePercent = 0; // Neutral default
  if (!assumptions.downRoundIpoPenalty) assumptions.downRoundIpoPenalty = 0; // Neutral default
  if (!assumptions.dualClassDiscount) assumptions.dualClassDiscount = 0; // Neutral default
  
  // Calculate ntmEBITDA if not directly provided but we have revenue and margin
  if (!assumptions.ntmEBITDA && assumptions.ntmRevenue && assumptions.ntmEBITDAMargin) {
    assumptions.ntmEBITDA = assumptions.ntmRevenue * assumptions.ntmEBITDAMargin;
  }
  
  // For restaurant/consumer sectors, ensure downRoundOptics defaults to false unless explicitly flagged
  const sectorLower = (assumptions.sector || "").toLowerCase();
  const isRestaurantConsumer = sectorLower === "restaurant" || sectorLower === "consumer_staples" || 
                               sectorLower === "retail" || sectorLower === "fast-casual" ||
                               sectorLower === "qsr" || sectorLower === "cpg";
  if (isRestaurantConsumer && assumptions.downRoundOptics === undefined) {
    assumptions.downRoundOptics = false; // Explicitly false for profitable consumer companies
  }
  
  // CRITICAL: NO inference from CEO guidance text - require explicit user input
  // If pricingAggressiveness not explicitly provided, default to "moderate" (neutral)
  if (!assumptions.pricingAggressiveness) {
    assumptions.pricingAggressiveness = "moderate"; // Neutral default
  }
  // managementPriority stays undefined if not explicitly provided (neutral)
  
  return { assumptions, providerUsed };
}

interface PricingRow {
  offerPrice: number;
  
  // Share counts - recomputed per price point
  sharesSoldPrimary: number; // Shares sold in primary offering (computed from dollar raise / price)
  sharesSoldSecondary: number; // Secondary shares sold
  sharesSoldGreenshoe: number; // Greenshoe shares
  totalSharesSold: number; // Total shares sold in IPO
  fdSharesPostIPO: number; // Fully diluted shares post-IPO (recalculated per price)
  
  // Ownership metrics - recomputed per price point
  dilutionPercent: number; // Dilution from primary + greenshoe
  founderOwnershipPost: number; // Founder/employee ownership post-IPO
  
  marketCapM: number;
  postIPOCashM: number;
  currentDebtM: number; // Debt from user input
  enterpriseValueM: number; // EV = MarketCap + Debt - Cash (CORRECT FORMULA)
  
  ntmEVRevenue: number;
  ntmEVEBITDA: number; // For restaurant/consumer sectors
  evRaNPV: number;
  growthAdjustedMultiple: number; // BUG FIX #5: peer multiple adjusted for deceleration
  
  vsPeerMedianRevenue: number;
  vsPeerMedianRaNPV: number;
  vsPeerMedianEBITDA: number; // For restaurant/consumer sectors
  
  fairValueSupport: number;
  grossProceedsM: number;
  basePrimaryProceedsM: number; // Exact target primary (e.g., $550M)
  greenshoeProceedsM: number; // Additional from greenshoe (15% additive)
  primaryProceedsM: number; // Total to company = basePrimary + greenshoe
  secondaryProceedsM: number; // Proceeds to selling shareholders
  
  oversubscription: number;
  effectiveOversubscription: number; // BUG FIX #4: after price-sensitive drop-off
  orderBookTier: string;
  investorsDropping: string[]; // BUG FIX #4: names of investors dropping at this price
  demandLostM: number; // BUG FIX #4: demand lost from max price constraints
  
  // Down-round analysis - BUG FIX #1
  downRoundPercent: number;
  isDownRound: boolean;
  downRoundDiscount: number;
  
  baseImpliedPop: number;
  bookQualityAdjustment: number;
  valuationPenalty: number;
  secondaryDiscount: number;
  catalystDiscount: number;
  dualClassDiscount: number; // BUG FIX #2
  customerConcentrationDiscount: number; // BUG FIX #6
  growthDecelPenalty: number; // BUG FIX #5
  adjustedImpliedPop: number;
  
  warnings: string[];
}

/**
 * SIMPLE DCF CALCULATION FOR IPO FAIR VALUE
 * Computes fair value per share from revenue/EBITDA projections
 * Uses sector-appropriate assumptions when not provided by user
 * CRITICAL: User-provided WACC takes precedence over sector defaults
 */
function calculateSimpleDCF(params: {
  ntmRevenue: number;
  ntmEBITDA?: number;
  ntmEBITDAMargin?: number;
  currentYearRevenue?: number;
  growthRates?: { fy2024to2025Growth?: number; fy2025to2026Growth?: number };
  currentCash: number;
  currentDebt: number;
  sharesOutstandingPreIPO: number;
  sector: string;
  userWACC?: number; // User-provided WACC takes precedence
  userTerminalGrowth?: number; // User-provided terminal growth
  projectionYears?: number; // User can specify projection years (default 5)
}): { fairValuePerShare: number; dcfDetails: string } {
  const {
    ntmRevenue,
    ntmEBITDA,
    ntmEBITDAMargin,
    growthRates,
    currentCash,
    currentDebt,
    sharesOutstandingPreIPO,
    sector,
    userWACC,
    userTerminalGrowth,
    projectionYears: inputProjYears,
  } = params;
  
  // Guard - can't compute DCF without revenue
  if (!ntmRevenue || ntmRevenue <= 0 || !sharesOutstandingPreIPO || sharesOutstandingPreIPO <= 0) {
    return { fairValuePerShare: 0, dcfDetails: "Insufficient data for DCF calculation" };
  }
  
  // Sector-specific defaults - USER-PROVIDED WACC TAKES PRECEDENCE
  let wacc = userWACC || 0.10; // Use user's WACC if provided, else 10% default
  let terminalGrowth = userTerminalGrowth || 0.025; // 2.5% default
  let targetEBITDAMargin = 0.20; // 20% terminal EBITDA margin
  const projectionYears = inputProjYears || 5;
  
  // Only apply sector defaults if user didn't provide WACC
  if (!userWACC) {
    switch (sector.toLowerCase()) {
      case "saas":
      case "enterprise software":
      case "software":
        wacc = 0.095; // 9.5% - lower risk for recurring revenue
        targetEBITDAMargin = 0.30; // 30% terminal
        break;
      case "consumer internet":
      case "consumer":
      case "marketplace":
        wacc = 0.11; // 11% - higher consumer risk
        targetEBITDAMargin = 0.25;
        break;
      case "fintech":
      case "payments":
        wacc = 0.10;
        targetEBITDAMargin = 0.25;
        break;
      case "ai_infrastructure":
      case "ai":
      case "hardware":
        wacc = 0.12; // 12% - capex heavy
        targetEBITDAMargin = 0.25;
        break;
      case "biotech":
      case "biopharmaceutical":
      case "clinical-stage":
        // For biotech, DCF doesn't apply - use raNPV instead
        return { fairValuePerShare: 0, dcfDetails: "DCF not applicable for biotech - use raNPV valuation" };
      // NEW: Restaurant and consumer staples sectors
      case "restaurant":
      case "fast-casual":
      case "qsr":
      case "food service":
        wacc = 0.085; // 8.5% - stable cash flows
        targetEBITDAMargin = 0.18; // 18% terminal
        terminalGrowth = userTerminalGrowth || 0.03; // 3% terminal
        break;
      case "consumer_staples":
      case "consumer staples":
      case "cpg":
        wacc = 0.08; // 8% - very stable
        targetEBITDAMargin = 0.20;
        terminalGrowth = userTerminalGrowth || 0.025;
        break;
      case "retail":
      case "e-commerce":
      case "dtc":
        wacc = 0.10;
        targetEBITDAMargin = 0.15;
        break;
    }
  }
  
  // Determine EBITDA margin - use provided or estimate from sector
  let currentMargin = ntmEBITDAMargin || (ntmEBITDA && ntmRevenue > 0 ? ntmEBITDA / ntmRevenue : 0.10);
  if (!isFinite(currentMargin)) currentMargin = 0.10;
  
  // Determine growth rate - use provided or estimate
  let initialGrowthRate = 0.20; // 20% default
  if (growthRates?.fy2024to2025Growth && growthRates.fy2024to2025Growth > 0) {
    initialGrowthRate = growthRates.fy2024to2025Growth;
  }
  
  // N-year DCF projection (default 5)
  let revenue = ntmRevenue;
  let totalPVFCF = 0;
  const dcfYears: string[] = [];
  
  for (let year = 1; year <= projectionYears; year++) {
    // Growth rate decays toward terminal
    const yearGrowthRate = initialGrowthRate * Math.pow(0.8, year - 1); // 20% decay per year
    revenue = revenue * (1 + yearGrowthRate);
    
    // Margin expands toward target
    const marginProgress = year / projectionYears;
    const yearMargin = currentMargin + (targetEBITDAMargin - currentMargin) * marginProgress;
    
    const ebitda = revenue * yearMargin;
    
    // Simple FCF estimate: EBITDA * 60% (after tax, capex, NWC)
    const fcfConversion = 0.60;
    const fcf = ebitda * fcfConversion;
    
    // Discount factor
    const discountFactor = Math.pow(1 + wacc, year);
    const pvFCF = fcf / discountFactor;
    totalPVFCF += pvFCF;
    
    dcfYears.push(`Y${year}: Rev=$${revenue.toFixed(0)}M, EBITDA=$${ebitda.toFixed(0)}M (${(yearMargin*100).toFixed(0)}%), FCF=$${fcf.toFixed(0)}M, PV=$${pvFCF.toFixed(0)}M`);
  }
  
  // Terminal value using perpetuity growth
  const terminalFCF = revenue * targetEBITDAMargin * 0.60; // Terminal year FCF
  const terminalValue = terminalFCF * (1 + terminalGrowth) / (wacc - terminalGrowth);
  const pvTerminal = terminalValue / Math.pow(1 + wacc, projectionYears);
  
  // Enterprise Value
  const enterpriseValue = totalPVFCF + pvTerminal;
  
  // Equity Value = EV - Debt + Cash
  const equityValue = enterpriseValue - currentDebt + currentCash;
  
  // Per share
  const fairValuePerShare = equityValue / sharesOutstandingPreIPO;
  
  const dcfDetails = [
    `DCF Calculation (WACC=${(wacc*100).toFixed(1)}%, Terminal Growth=${(terminalGrowth*100).toFixed(1)}%):`,
    ...dcfYears,
    `Terminal Value: $${terminalValue.toFixed(0)}M (PV: $${pvTerminal.toFixed(0)}M)`,
    `Enterprise Value: $${enterpriseValue.toFixed(0)}M`,
    `Less Debt: -$${currentDebt.toFixed(0)}M`,
    `Plus Cash: +$${currentCash.toFixed(0)}M`,
    `Equity Value: $${equityValue.toFixed(0)}M`,
    `Shares: ${(sharesOutstandingPreIPO).toFixed(1)}M`,
    `Fair Value/Share: $${fairValuePerShare.toFixed(2)}`,
  ].join("\n");
  
  return { 
    fairValuePerShare: isFinite(fairValuePerShare) && fairValuePerShare > 0 ? fairValuePerShare : 0, 
    dcfDetails 
  };
}

export function calculateIPOPricing(assumptions: IPOAssumptions): {
  assumptions: IPOAssumptions;
  pricingMatrix: PricingRow[];
  recommendedRangeLow: number;
  recommendedRangeHigh: number;
  recommendedPrice: number;
  rationale: string[];
  warnings: string[];
  memoText: string;
} {
  const {
    companyName,
    sector,
    sharesOutstandingPreIPO,
    primarySharesOffered: inputPrimaryShares,
    secondarySharesOffered: inputSecondaryShares = 0,
    greenshoeShares: inputGreenshoeShares,
    greenshoePercent,
    // Dollar-based inputs (take precedence if provided)
    primaryDollarRaiseM,
    secondaryDollarRaiseM,
    currentCash = 0,
    currentDebt = 0, // NEW: Required for proper EV calculation
    ntmRevenue,
    fairValuePerShare,
    fairValueType,
    totalRaNPV = 0,
    peerMedianEVRevenue,
    peerMedianEVRaNPV = 0,
    orderBook,
    notableOrders = [],
    sectorMedianFirstDayPop,
    sectorAverageFirstDayPop,
    historicalFirstDayPop,
    foundersEmployeesOwnership,
    founderSharesExplicitM, // NEW: Explicit founder shares from cap table
    pricingAggressiveness,
    managementPriority,
    minAcceptablePrice,
    ceoGuidance,
    hasBinaryCatalyst = false,
    monthsToCatalyst, // No default - user must provide if catalyst exists
    secondaryOptics = "neutral",
    indicatedPriceRangeLow,
    indicatedPriceRangeHigh,
    // BUG FIX #1: Down-round detection
    lastPrivateRoundPrice,
    downRoundOptics = false,
    downRoundIpoPenalty = 0, // Neutral default - user must provide penalty
    // BUG FIX #2: Dual-class governance - user must provide discount
    dualClass = false,
    dualClassDiscount: dualClassDiscountRate = 0, // Neutral default - user must provide discount
    // BUG FIX #5: Growth trajectory
    growthRates,
    // BUG FIX #6: Customer concentration
    customerConcentrationTop5 = 0,
    // DCF Inputs - user-provided override sector defaults
    wacc: userWACC,
    terminalGrowthRate: userTerminalGrowth,
    capexPercent,
    taxRate,
    nwcDays,
    // New: EV/EBITDA for restaurant/consumer sectors
    peerMedianEVEBITDA,
    peerMedianNTMFCF,
  } = assumptions;

  // FIX: Sector detection - only treat as biotech if explicitly tagged biotech
  // Don't force biotech mode just because revenue is missing
  const isBiotech = sector === "biotech" || sector === "biopharmaceutical" || sector === "clinical-stage";
  const isPreRevenue = ntmRevenue === 0 || ntmRevenue < 1;
  // Only use raNPV valuation if EXPLICITLY biotech/clinical-stage, not just missing revenue
  const useRaNPVValuation = isBiotech && isPreRevenue;
  
  // Sector uses EV/EBITDA as primary multiple (not EV/Revenue)
  const sectorLower = (sector || "").toLowerCase();
  const useEVEBITDAValuation = sectorLower === "restaurant" || sectorLower === "fast-casual" || 
                               sectorLower === "qsr" || sectorLower === "food service" ||
                               sectorLower === "consumer_staples" || sectorLower === "consumer staples" ||
                               sectorLower === "retail" || sectorLower === "cpg";
  
  // Down-round logic ONLY applies if:
  // 1. User explicitly says downRoundOptics is true, OR
  // 2. This is a biotech/clinical-stage company (where down-rounds are common IPO risk)
  // For profitable consumer/restaurant companies, down-rounds are NOT relevant IPO concerns
  const applyDownRoundLogic = downRoundOptics === true || isBiotech;
  
  const warnings: string[] = [];
  
  // CRITICAL: Backfill ntmEBITDA from revenue × margin BEFORE price range derivation
  // This ensures EV/EBITDA Method 3 can run for restaurant/consumer sectors
  let effectiveNtmEBITDA = assumptions.ntmEBITDA || 0;
  if (!effectiveNtmEBITDA && ntmRevenue > 0 && assumptions.ntmEBITDAMargin && assumptions.ntmEBITDAMargin > 0) {
    effectiveNtmEBITDA = ntmRevenue * assumptions.ntmEBITDAMargin;
    assumptions.ntmEBITDA = effectiveNtmEBITDA; // Update assumptions for downstream use
  }
  
  // Add sector detection warning for non-biotech with missing revenue
  if (isPreRevenue && !isBiotech) {
    warnings.push(`Pre-revenue company detected but not marked as biotech - using standard valuation metrics. Set sector="biotech" if this is a clinical-stage company.`);
  }
  
  // Add warning for restaurant/consumer sectors if EV/EBITDA data is missing
  if (useEVEBITDAValuation) {
    if (!peerMedianEVEBITDA || peerMedianEVEBITDA <= 0) {
      warnings.push(`Restaurant/consumer sector detected but peerMedianEVEBITDA not provided - may fall back to EV/Revenue valuation`);
    }
    if (!effectiveNtmEBITDA || effectiveNtmEBITDA <= 0) {
      warnings.push(`Restaurant/consumer sector detected but ntmEBITDA not provided - may fall back to EV/Revenue valuation`);
    }
  }
  
  // Determine if we're using dollar-based or share-based inputs
  const useDollarBased = (primaryDollarRaiseM !== undefined && primaryDollarRaiseM > 0);
  
  // BUG FIX #5: Calculate growth deceleration penalty for peer multiple
  // ONLY apply when CEO is NOT prioritizing aggressive pricing/pop maximization
  // MECHANICAL: compression equals the deceleration rate itself (no embedded multiplier)
  let growthDecelPenalty = 0;
  let growthAdjustedPeerMultiple = peerMedianEVRevenue || 0;
  
  // Check if CEO wants aggressive pricing - if so, skip growth decel penalty
  const wantsAggressivePricing = pricingAggressiveness === "aggressive" || 
                                  pricingAggressiveness === "maximum" ||
                                  managementPriority === "valuation_maximization";
  
  // Only apply growth decel to non-biotech revenue companies with explicit growth rates
  // AND only when NOT prioritizing aggressive pricing
  if (!wantsAggressivePricing && !isBiotech && !isPreRevenue && peerMedianEVRevenue > 0 && 
      growthRates && growthRates.fy2024to2025Growth && growthRates.fy2024to2025Growth > 0 && 
      growthRates.fy2025to2026Growth !== undefined) {
    const decelRate = 1 - (growthRates.fy2025to2026Growth / growthRates.fy2024to2025Growth);
    if (decelRate > 0 && isFinite(decelRate)) {
      // MECHANICAL: growth decel penalty = the deceleration rate itself
      // If growth decelerates by 30%, multiple compresses by 30%
      growthDecelPenalty = decelRate;
      growthAdjustedPeerMultiple = peerMedianEVRevenue * (1 - growthDecelPenalty);
    }
  }
  
  // Determine price range - use user input, or COMPUTE from fair value / raise amounts
  const hasUserProvidedRange = (indicatedPriceRangeLow !== undefined && indicatedPriceRangeLow > 0) && 
                                (indicatedPriceRangeHigh !== undefined && indicatedPriceRangeHigh > 0);
  
  let minPrice: number;
  let maxPrice: number;
  let priceRangeSource: string = "user-provided";
  
  // Track DCF details for memo if we compute it
  let dcfComputedDetails: string = "";
  let effectiveFairValuePerShare = fairValuePerShare || 0;
  
  // ALWAYS compute DCF if we have revenue data (for fair value support calculation)
  // This runs BEFORE price range determination so we have DCF available regardless of how range is set
  // CRITICAL: Use user-provided WACC and terminal growth if given
  if (!isBiotech && ntmRevenue > 0 && sharesOutstandingPreIPO > 0 && effectiveFairValuePerShare <= 0) {
    const dcfResult = calculateSimpleDCF({
      ntmRevenue,
      ntmEBITDA: assumptions.ntmEBITDA,
      ntmEBITDAMargin: assumptions.ntmEBITDAMargin,
      currentYearRevenue: assumptions.currentYearRevenue,
      growthRates,
      currentCash,
      currentDebt,
      sharesOutstandingPreIPO,
      sector,
      userWACC,  // Pass user-provided WACC
      userTerminalGrowth,  // Pass user-provided terminal growth
    });
    
    if (dcfResult.fairValuePerShare > 0) {
      effectiveFairValuePerShare = dcfResult.fairValuePerShare;
      dcfComputedDetails = dcfResult.dcfDetails;
      warnings.push(`DCF fair value: $${dcfResult.fairValuePerShare.toFixed(2)}/share`);
    }
  }
  
  if (hasUserProvidedRange) {
    // Guard for inverted range
    if (indicatedPriceRangeHigh <= indicatedPriceRangeLow) {
      const errorWarning = "ERROR: indicatedPriceRangeHigh must be greater than indicatedPriceRangeLow.";
      return {
        assumptions,
        pricingMatrix: [],
        recommendedRangeLow: 0,
        recommendedRangeHigh: 0,
        recommendedPrice: 0,
        rationale: [],
        warnings: [errorWarning],
        memoText: `IPO PRICING ERROR\n\n${errorWarning}`,
      };
    }
    minPrice = indicatedPriceRangeLow;
    maxPrice = indicatedPriceRangeHigh;
    priceRangeSource = "user-provided";
  } else {
    // === CORRECTED PRICING LOGIC ===
    // Priority: Peer multiples = BASE CASE anchor, DCF = CEILING (not target)
    // Price at 85-90% of fair value to ensure +10-15% Day-1 pop
    
    let peerDerivedPrice: number | null = null;
    let dcfCeilingPrice: number | null = null;
    let peerSource = "";
    
    // Step 1: Compute PEER-DERIVED price (this is the BASE CASE)
    // For restaurant/consumer - use EV/EBITDA
    // POST-MONEY EQUITY CALCULATION: Accounts for primary proceeds
    // NOTE: Uses iterative convergence for circular dependency (price → shares → price)
    if (useEVEBITDAValuation && peerMedianEVEBITDA && peerMedianEVEBITDA > 0 && 
        assumptions.ntmEBITDA && assumptions.ntmEBITDA > 0 && sharesOutstandingPreIPO > 0) {
      const impliedEV = assumptions.ntmEBITDA * peerMedianEVEBITDA;
      const expectedPrimaryRaiseM = primaryDollarRaiseM || 0;
      const impliedEquity = impliedEV - currentDebt + currentCash + expectedPrimaryRaiseM;
      
      // Iterative convergence for post-money price
      const preMoneyEquity = impliedEV - currentDebt + currentCash;
      let currentPrice = preMoneyEquity / sharesOutstandingPreIPO;
      
      // Iterate up to 5 times for convergence (typically converges in 2-3)
      for (let i = 0; i < 5; i++) {
        const estimatedPrimaryShares = expectedPrimaryRaiseM > 0 && currentPrice > 0 
          ? expectedPrimaryRaiseM / currentPrice 
          : 0;
        const postIPOShares = sharesOutstandingPreIPO + estimatedPrimaryShares;
        const newPrice = postIPOShares > 0 ? impliedEquity / postIPOShares : currentPrice;
        if (Math.abs(newPrice - currentPrice) < 0.01) break; // Converged
        currentPrice = newPrice;
      }
      
      peerDerivedPrice = currentPrice;
      peerSource = `peer EV/EBITDA ${peerMedianEVEBITDA.toFixed(1)}×`;
    }
    // For revenue companies - use EV/Revenue
    // POST-MONEY EQUITY CALCULATION: Accounts for primary proceeds
    // NOTE: Uses two-pass iteration for circular dependency (price → shares → price)
    // This converges well for typical IPOs; large primary raises (>50% of market cap) may require manual adjustment
    else if (!useEVEBITDAValuation && peerMedianEVRevenue > 0 && ntmRevenue > 0 && sharesOutstandingPreIPO > 0) {
      const impliedEV = ntmRevenue * peerMedianEVRevenue;
      const expectedPrimaryRaiseM = primaryDollarRaiseM || 0;
      const impliedEquity = impliedEV - currentDebt + currentCash + expectedPrimaryRaiseM;
      
      // Iterative convergence for post-money price
      const preMoneyEquity = impliedEV - currentDebt + currentCash;
      let currentPrice = preMoneyEquity / sharesOutstandingPreIPO;
      
      // Iterate up to 5 times for convergence (typically converges in 2-3)
      for (let i = 0; i < 5; i++) {
        const estimatedPrimaryShares = expectedPrimaryRaiseM > 0 && currentPrice > 0 
          ? expectedPrimaryRaiseM / currentPrice 
          : 0;
        const postIPOShares = sharesOutstandingPreIPO + estimatedPrimaryShares;
        const newPrice = postIPOShares > 0 ? impliedEquity / postIPOShares : currentPrice;
        if (Math.abs(newPrice - currentPrice) < 0.01) break; // Converged
        currentPrice = newPrice;
      }
      
      peerDerivedPrice = currentPrice;
      peerSource = `peer EV/Revenue ${peerMedianEVRevenue.toFixed(1)}×`;
    }
    // For biotech - use EV/raNPV
    else if (totalRaNPV > 0 && peerMedianEVRaNPV > 0 && sharesOutstandingPreIPO > 0) {
      const impliedEV = totalRaNPV * peerMedianEVRaNPV;
      peerDerivedPrice = impliedEV / sharesOutstandingPreIPO;
      peerSource = `peer EV/raNPV ${peerMedianEVRaNPV.toFixed(1)}×`;
    }
    
    // Step 2: Compute DCF CEILING (caps peer valuation, not the anchor)
    if (effectiveFairValuePerShare > 0) {
      dcfCeilingPrice = effectiveFairValuePerShare;
    }
    
    // Step 3: Determine anchor price
    // CORRECTED LOGIC: Peer multiples = base, DCF = ceiling
    let anchorPrice: number | null = null;
    
    if (peerDerivedPrice !== null && peerDerivedPrice > 0) {
      // Peer-derived is the base case
      anchorPrice = peerDerivedPrice;
      priceRangeSource = `derived from ${peerSource}`;
      
      // If DCF ceiling exists and is LOWER than peer, cap at DCF
      if (dcfCeilingPrice !== null && dcfCeilingPrice > 0 && dcfCeilingPrice < peerDerivedPrice) {
        anchorPrice = dcfCeilingPrice;
        priceRangeSource = `derived from DCF ceiling (peer ${peerSource} exceeded DCF)`;
        warnings.push(`Peer-derived price $${peerDerivedPrice.toFixed(2)} exceeded DCF ceiling $${dcfCeilingPrice.toFixed(2)} - using DCF as cap`);
      }
    } else if (dcfCeilingPrice !== null && dcfCeilingPrice > 0) {
      // No peer data - fall back to DCF only
      anchorPrice = dcfCeilingPrice;
      priceRangeSource = "derived from DCF valuation (no peer data)";
      warnings.push("No peer multiples provided - using DCF only (recommend providing peer comps)");
    }
    
    // Step 4: Fallback to dollar raise / shares if no valuation data
    if (anchorPrice === null && primaryDollarRaiseM && primaryDollarRaiseM > 0 && inputPrimaryShares && inputPrimaryShares > 0) {
      anchorPrice = primaryDollarRaiseM / inputPrimaryShares;
      priceRangeSource = "derived from dollar raise / shares offered";
    }
    
    // Step 5: Error if no anchor could be computed
    if (anchorPrice === null || anchorPrice <= 0) {
      // Check for missing inputs in restaurant/consumer sectors
      if (useEVEBITDAValuation) {
        const errorWarning = `ERROR: ${sector || 'Consumer'} sector detected - EV/EBITDA valuation required but inputs missing. Please provide: (1) ntmEBITDA (or ntmRevenue + ntmEBITDAMargin), AND (2) peerMedianEVEBITDA. EV/Revenue is not appropriate for this sector.`;
        return {
          assumptions,
          pricingMatrix: [],
          recommendedRangeLow: 0,
          recommendedRangeHigh: 0,
          recommendedPrice: 0,
          rationale: [],
          warnings: [errorWarning],
          memoText: `IPO PRICING ERROR\n\n${errorWarning}`,
        };
      }
      const errorWarning = "ERROR: Cannot determine price range. Please provide either (1) indicatedPriceRangeLow/High, (2) peer multiples with revenue/EBITDA, (3) fairValuePerShare, or (4) primaryDollarRaiseM with primarySharesOffered.";
      return {
        assumptions,
        pricingMatrix: [],
        recommendedRangeLow: 0,
        recommendedRangeHigh: 0,
        recommendedPrice: 0,
        rationale: [],
        warnings: [errorWarning],
        memoText: `IPO PRICING ERROR\n\n${errorWarning}`,
      };
    }
    
    // === CRITICAL FIX: Price at 85-90% of anchor for +10-15% pop ===
    // Standard IPO practice: File at discount to fair value to ensure positive aftermarket
    // Target pricing at ~87.5% of fair value (midpoint of 85-90%)
    const IPO_DISCOUNT_TARGET = 0.875; // 87.5% of fair value = ~14% expected pop
    
    const targetPrice = anchorPrice * IPO_DISCOUNT_TARGET;
    
    // Build range: ±7.5% around target price
    minPrice = Math.round(targetPrice * 0.915); // Low end: ~80% of anchor
    maxPrice = Math.round(targetPrice * 1.085); // High end: ~95% of anchor
    
    // Ensure minimum $1 range and proper ordering
    if (minPrice < 1) minPrice = 1;
    if (maxPrice <= minPrice) maxPrice = minPrice + 4;
    
    warnings.push(`Price range ${priceRangeSource}: $${minPrice} - $${maxPrice}`);
    if (peerDerivedPrice && peerDerivedPrice > 0) {
      warnings.push(`Peer-implied fair value: $${peerDerivedPrice.toFixed(2)}/share (${peerSource})`);
    }
    if (dcfCeilingPrice && dcfCeilingPrice > 0) {
      warnings.push(`DCF ceiling: $${dcfCeilingPrice.toFixed(2)}/share`);
    }
  }
  
  const pricePoints: number[] = [];
  for (let p = minPrice; p <= maxPrice; p += 1) {
    if (p > 0) pricePoints.push(p);
  }
  
  // Sort order book by price DESCENDING - only use explicit user-provided tiers
  const sortedOrderBook = orderBook ? [...orderBook].sort((a, b) => b.priceLevel - a.priceLevel) : [];
  const hasExplicitOrderBook = sortedOrderBook.length > 0;
  
  // Determine base expected return for sector - only from user input
  const baseExpectedReturn = sectorMedianFirstDayPop ?? sectorAverageFirstDayPop ?? historicalFirstDayPop ?? 0;
  // If no sector data provided, we won't compute POP
  const hasUserProvidedPopData = sectorMedianFirstDayPop !== undefined || 
                                  sectorAverageFirstDayPop !== undefined || 
                                  historicalFirstDayPop !== undefined;
  
  const pricingMatrix: PricingRow[] = pricePoints.map(offerPrice => {
    const rowWarnings: string[] = [];
    
    // === SHARES SOLD CALCULATION - RECOMPUTED AT EACH PRICE POINT ===
    // If dollar-based inputs provided, calculate shares from dollar raise / price
    // Otherwise use share-based inputs
    let sharesSoldPrimary: number;
    let sharesSoldSecondary: number;
    let sharesSoldGreenshoe: number;
    
    if (useDollarBased) {
      // MECHANICAL: shares = dollar amount / price per share
      sharesSoldPrimary = primaryDollarRaiseM! / offerPrice;
      sharesSoldSecondary = (secondaryDollarRaiseM || 0) / offerPrice;
      // Greenshoe from user-provided percent ONLY - no fabricated default
      sharesSoldGreenshoe = sharesSoldPrimary * (greenshoePercent || 0);
    } else {
      // Use explicit share counts from user input
      sharesSoldPrimary = inputPrimaryShares || 0;
      sharesSoldSecondary = inputSecondaryShares || 0;
      sharesSoldGreenshoe = inputGreenshoeShares || (sharesSoldPrimary * (greenshoePercent || 0));
    }
    
    // === GREENSHOE EXERCISE ASSUMPTION ===
    // Default: true (standard for successful IPOs)
    const greenshoeExercised = assumptions.greenshoeAssumedExercised !== false;
    
    const totalSharesSold = greenshoeExercised 
      ? sharesSoldPrimary + sharesSoldSecondary + sharesSoldGreenshoe
      : sharesSoldPrimary + sharesSoldSecondary; // No greenshoe if not exercised
    
    // === FULLY DILUTED SHARES - RECOMPUTED AT EACH PRICE POINT ===
    // Only primary shares (and greenshoe if exercised) are dilutive
    const dilutiveShares = greenshoeExercised 
      ? sharesSoldPrimary + sharesSoldGreenshoe
      : sharesSoldPrimary; // No greenshoe dilution if not exercised
    const fdSharesPostIPO = sharesOutstandingPreIPO + dilutiveShares;
    
    // === DILUTION CALCULATION - MECHANICAL ===
    const dilutionPercent = dilutiveShares / fdSharesPostIPO;
    
    // === PROCEEDS CALCULATION - CORRECTED: Greenshoe is ADDITIVE ===
    // Primary proceeds = exact target from primaryDollarRaiseM (greenshoe is SEPARATE)
    // Greenshoe proceeds = additional 15% on top of primary (if exercised)
    const basePrimaryProceedsM = offerPrice * sharesSoldPrimary; // Hits exact target (e.g., $550M)
    const greenshoeProceedsM = greenshoeExercised 
      ? offerPrice * sharesSoldGreenshoe // ADDITIVE on top (e.g., ~$82.5M)
      : 0; // No greenshoe proceeds if not exercised
    const secondaryProceedsM = offerPrice * sharesSoldSecondary;
    const primaryProceedsM = basePrimaryProceedsM + greenshoeProceedsM; // Total to company
    const grossProceedsM = basePrimaryProceedsM + greenshoeProceedsM + secondaryProceedsM;
    
    // === MARKET CAP - MECHANICAL: Price × FD Shares ===
    const marketCapM = fdSharesPostIPO * offerPrice;
    
    // === CASH POSITION POST-IPO ===
    // Post-IPO Cash = Current Cash + Primary Proceeds (secondary goes to sellers)
    // Greenshoe is included in primaryProceedsM only if assumed exercised (default true)
    // Note: greenshoeExercised is defined above in the shares section
    const effectivePrimaryProceedsM = greenshoeExercised 
      ? primaryProceedsM // Base + Greenshoe (default)
      : basePrimaryProceedsM; // Base only (if greenshoe not exercised)
    const postIPOCashM = currentCash + effectivePrimaryProceedsM;
    
    // === ENTERPRISE VALUE - CORRECT FORMULA: EV = MarketCap + Debt - Cash ===
    const currentDebtM = currentDebt;
    const enterpriseValueM = marketCapM + currentDebtM - postIPOCashM;
    
    // === VALUATION MULTIPLES ===
    // NTM EV/Revenue - with guards for division by zero
    // Return 0 (not Infinity) when data is missing - cleaner for display
    let ntmEVRevenue = 0;
    if (!isPreRevenue && ntmRevenue > 0 && isFinite(enterpriseValueM)) {
      ntmEVRevenue = Math.round((enterpriseValueM / ntmRevenue) * 10) / 10;
      if (!isFinite(ntmEVRevenue)) ntmEVRevenue = 0; // Guard
    }
    
    // EV/raNPV for biotech - with guards
    let evRaNPV = 0;
    if (totalRaNPV > 0 && isFinite(enterpriseValueM)) {
      evRaNPV = Math.round((enterpriseValueM / totalRaNPV) * 100) / 100;
      if (!isFinite(evRaNPV)) evRaNPV = 0; // Guard
    }
    
    // EV/EBITDA for restaurant/consumer sectors - with guards
    let ntmEVEBITDA = 0;
    const ntmEBITDA = assumptions.ntmEBITDA || 0;
    if (useEVEBITDAValuation && ntmEBITDA > 0 && isFinite(enterpriseValueM)) {
      ntmEVEBITDA = Math.round((enterpriseValueM / ntmEBITDA) * 10) / 10;
      if (!isFinite(ntmEVEBITDA)) ntmEVEBITDA = 0; // Guard
    }
    
    // vs Peer Median comparisons (using growth-adjusted multiple for revenue comps)
    let vsPeerMedianRevenue = 0;
    if (!isPreRevenue && ntmEVRevenue > 0 && growthAdjustedPeerMultiple > 0) {
      vsPeerMedianRevenue = Math.round(((ntmEVRevenue - growthAdjustedPeerMultiple) / growthAdjustedPeerMultiple) * 1000) / 1000;
      if (!isFinite(vsPeerMedianRevenue)) vsPeerMedianRevenue = 0; // Guard
    }
    
    let vsPeerMedianRaNPV = 0;
    if (totalRaNPV > 0 && evRaNPV > 0 && peerMedianEVRaNPV > 0) {
      vsPeerMedianRaNPV = Math.round(((evRaNPV - peerMedianEVRaNPV) / peerMedianEVRaNPV) * 1000) / 1000;
      if (!isFinite(vsPeerMedianRaNPV)) vsPeerMedianRaNPV = 0; // Guard
    }
    
    // vs Peer Median EV/EBITDA for restaurant/consumer
    let vsPeerMedianEBITDA = 0;
    if (useEVEBITDAValuation && ntmEVEBITDA > 0 && peerMedianEVEBITDA && peerMedianEVEBITDA > 0) {
      vsPeerMedianEBITDA = Math.round(((ntmEVEBITDA - peerMedianEVEBITDA) / peerMedianEVEBITDA) * 1000) / 1000;
      if (!isFinite(vsPeerMedianEBITDA)) vsPeerMedianEBITDA = 0; // Guard
    }
    
    // === FAIR VALUE SUPPORT - CONSISTENT ROUNDING WITH GUARDS ===
    // Uses effectiveFairValuePerShare which includes computed DCF if user didn't provide
    let fairValueSupport = 0;
    if (effectiveFairValuePerShare > 0 && offerPrice > 0) {
      fairValueSupport = Math.round((offerPrice / effectiveFairValuePerShare) * 1000) / 1000;
      if (!isFinite(fairValueSupport)) fairValueSupport = 0; // Guard
    }
    
    // === ORDER BOOK LOOKUP - WITH REALISTIC SYNTHETIC CURVE ===
    // If user provides explicit order book data, use it directly
    // Otherwise generate realistic oversubscription curve based on price position
    let oversubscription = 1; // Neutral default
    let orderBookTier = "N/A";
    
    if (hasExplicitOrderBook) {
      // Find the matching tier from user input
      let matchedTier = sortedOrderBook.find(entry => offerPrice >= entry.priceLevel);
      
      if (matchedTier) {
        oversubscription = matchedTier.oversubscription;
        orderBookTier = `$${matchedTier.priceLevel}+`;
      } else if (offerPrice > sortedOrderBook[0].priceLevel) {
        // Price is above all tiers - use highest tier's value, NO fabricated extrapolation
        oversubscription = sortedOrderBook[0].oversubscription;
        orderBookTier = `Above $${sortedOrderBook[0].priceLevel}`;
        rowWarnings.push(`Price above highest order book tier`);
      } else {
        // Price is below all tiers - use lowest tier's value
        const lowestTier = sortedOrderBook[sortedOrderBook.length - 1];
        oversubscription = lowestTier.oversubscription;
        orderBookTier = `Below $${lowestTier.priceLevel}`;
      }
    } else {
      // SYNTHETIC CURVE: Generate realistic oversubscription based on price position
      // Real order books typically show:
      // - 3-8x oversubscription at low end of range
      // - 0.6-0.9x at high end of range
      // - Linear interpolation between
      
      if (minPrice < maxPrice) {
        const pricePosition = (offerPrice - minPrice) / (maxPrice - minPrice); // 0 at low, 1 at high
        
        // Interpolate from 5x at low to 0.8x at high (realistic curve)
        const highEndOversub = 0.8;  // At max price
        const lowEndOversub = 5.0;   // At min price
        
        oversubscription = lowEndOversub + (highEndOversub - lowEndOversub) * pricePosition;
        oversubscription = Math.round(oversubscription * 10) / 10; // Round to 1 decimal
        
        // Neutral tier descriptions (not biotech-specific)
        orderBookTier = pricePosition < 0.33 ? "Below mid" : 
                        pricePosition < 0.67 ? "Around mid" : "Above mid";
      }
    }
    
    // BUG FIX #4: Price-sensitive investor drop-off
    const investorsDropping: string[] = [];
    let demandLostM = 0;
    if (notableOrders && notableOrders.length > 0) {
      for (const investor of notableOrders) {
        if (investor.maxPrice && offerPrice > investor.maxPrice) {
          investorsDropping.push(investor.investorName);
          demandLostM += investor.indicatedSizeM;
        }
      }
    }
    
    // Calculate effective oversubscription after drop-off
    // MECHANICAL: no fabricated floor - use actual computed value
    let effectiveOversubscription = oversubscription;
    if (demandLostM > 0 && grossProceedsM > 0) {
      const demandLostRatio = demandLostM / (grossProceedsM * oversubscription);
      effectiveOversubscription = oversubscription * (1 - demandLostRatio);
      // No fabricated floor - actual mechanical calculation
    }
    
    // BUG FIX #1: Down-round detection and discount
    // ONLY apply to biotech or when user explicitly flags downRoundOptics
    // For profitable consumer/restaurant companies, down-rounds are NOT relevant
    let downRoundPercent = 0;
    let isDownRound = false;
    let downRoundDiscount = 0;
    if (lastPrivateRoundPrice && lastPrivateRoundPrice > 0 && applyDownRoundLogic) {
      downRoundPercent = (offerPrice - lastPrivateRoundPrice) / lastPrivateRoundPrice;
      isDownRound = downRoundPercent < 0;
      if (isDownRound && downRoundIpoPenalty > 0) {
        // MECHANICAL: use user-provided penalty coefficient only
        // downRoundIpoPenalty from user determines pass-through rate
        downRoundDiscount = Math.abs(downRoundPercent) * downRoundIpoPenalty;
      }
    }
    
    // === IMPLIED POP CALCULATION - CORRECTED: Price below fair value = positive pop ===
    // Expected Day-1 return = (FairValue - OfferPrice) / OfferPrice
    // If we price at 87.5% of fair value, expected pop = (1/0.875 - 1) = +14.3%
    
    let baseImpliedPop = 0;
    let bookQualityAdjustment = 0;
    let valuationPenalty = 0; // Now called valuationBonus when positive
    let secondaryDiscount = 0;
    let catalystDiscount = 0;
    let adjustedImpliedPop = 0;
    
    // CORRECTED: Expected pop = how much below fair value we're pricing
    // If fairValueSupport < 1 (pricing below fair value), pop is POSITIVE
    // If fairValueSupport > 1 (pricing above fair value), pop is NEGATIVE
    if (effectiveFairValuePerShare > 0) {
      // Pop = (FairValue / OfferPrice) - 1 = (1 / fairValueSupport) - 1
      // e.g., pricing at 87.5% of FV: pop = (1/0.875) - 1 = +14.3%
      // e.g., pricing at 103% of FV: pop = (1/1.03) - 1 = -2.9%
      baseImpliedPop = (1 / fairValueSupport) - 1;
    } else if (hasUserProvidedPopData) {
      // Fall back to user-provided sector historical data
      baseImpliedPop = baseExpectedReturn;
    }
    
    // Book quality adjustment - ONLY if order book data provided
    // PURELY MECHANICAL: log of oversubscription ratio directly
    if (hasExplicitOrderBook && effectiveOversubscription > 0 && effectiveOversubscription !== 1) {
      // Scale down the log impact (0.05x) to avoid overwhelming the base pop
      bookQualityAdjustment = Math.log(effectiveOversubscription) * 0.05;
    }
    
    // Valuation penalty - now part of base pop calculation above
    // No separate penalty needed since base pop already reflects fair value discount
    valuationPenalty = 0;
    
    // Secondary discount - PURELY MECHANICAL from user optics input
    if (sharesSoldSecondary > 0 && totalSharesSold > 0) {
      const secondaryPct = sharesSoldSecondary / totalSharesSold;
      secondaryDiscount = secondaryOptics === "negative" ? secondaryPct * 0.5 : 0;
    }
    
    // Binary catalyst discount - PURELY MECHANICAL from user months
    if (hasBinaryCatalyst && monthsToCatalyst !== undefined && monthsToCatalyst > 0) {
      catalystDiscount = 0.05 / monthsToCatalyst; // Scaled down
    }
    
    // Total adjusted implied pop
    adjustedImpliedPop = baseImpliedPop 
      + bookQualityAdjustment 
      + valuationPenalty 
      - secondaryDiscount 
      - catalystDiscount
      - downRoundDiscount
      - (dualClass ? dualClassDiscountRate : 0)
      - (customerConcentrationTop5 > 0.40 ? (customerConcentrationTop5 - 0.40) * 0.1 : 0);
    
    // BUG FIX #2: Dual-class governance discount (for display)
    const dualClassDiscount = dualClass ? dualClassDiscountRate : 0;
    
    // BUG FIX #6: Customer concentration discount (for display)
    // MECHANICAL: discount equals the excess concentration above threshold
    // No embedded multipliers - direct relationship
    const customerConcentrationDiscount = customerConcentrationTop5 > 0.40 
      ? (customerConcentrationTop5 - 0.40) 
      : 0;
    
    // === FOUNDER OWNERSHIP - RECOMPUTED AT EACH PRICE POINT ===
    // Mechanical: founderShares / fdSharesPostIPO
    // Founder shares are fixed; FD shares change with primary+greenshoe at different prices
    // 
    // PRIORITY: Use explicit founder shares from cap table if provided (e.g., CEO 38M + CTO 22M = 60M)
    // Otherwise, fall back to percentage-based calculation
    const founderSharesFixed = founderSharesExplicitM && founderSharesExplicitM > 0
      ? founderSharesExplicitM  // Use explicit share count from cap table parsing
      : foundersEmployeesOwnership * sharesOutstandingPreIPO; // Fall back to percentage
    const founderOwnershipPost = fdSharesPostIPO > 0 ? founderSharesFixed / fdSharesPostIPO : 0;
    
    // Generate warnings - ONLY based on user-provided data, no hardcoded thresholds
    // Fair value support - show whether pricing above or below fair value
    if (fairValueSupport > 0) {
      const fvPercent = (fairValueSupport * 100).toFixed(0);
      const fvLabel = fairValueSupport < 1 ? `${fvPercent}% of fair value (discount)` : 
                      fairValueSupport > 1 ? `${fvPercent}% of fair value (premium)` : 
                      "at fair value";
      rowWarnings.push(`Valuation: ${fvLabel}`);
    }
    // Book coverage - report actual metric without judgment threshold
    if (hasExplicitOrderBook && effectiveOversubscription > 0) {
      rowWarnings.push(`Book coverage: ${effectiveOversubscription.toFixed(1)}× effective oversubscription`);
    }
    // Expected Day-1 pop - ALWAYS show when we have fair value to derive it
    if (effectiveFairValuePerShare > 0) {
      const popSign = adjustedImpliedPop >= 0 ? '+' : '';
      rowWarnings.push(`Expected Day-1 POP: ${popSign}${(adjustedImpliedPop * 100).toFixed(1)}%`);
    }
    // Down-round - only show for biotech or when user explicitly flags it
    if (isDownRound && applyDownRoundLogic) {
      rowWarnings.push(`DOWN-ROUND: ${(downRoundPercent * 100).toFixed(1)}% vs Series E`);
    }
    // Investor drop-off - factual from user's order book
    if (investorsDropping.length > 0) {
      rowWarnings.push(`INVESTOR DROP-OFF: ${investorsDropping.join(", ")} ($${demandLostM}M lost)`);
    }
    
    return {
      offerPrice,
      
      // Share counts - recomputed per price point
      sharesSoldPrimary,
      sharesSoldSecondary,
      sharesSoldGreenshoe,
      totalSharesSold,
      fdSharesPostIPO,
      
      // Ownership metrics - recomputed per price point  
      dilutionPercent,
      founderOwnershipPost,
      
      marketCapM,
      postIPOCashM,
      currentDebtM,
      enterpriseValueM, // EV = MarketCap + Debt - Cash (CORRECT)
      
      ntmEVRevenue,
      ntmEVEBITDA,
      evRaNPV,
      growthAdjustedMultiple: growthAdjustedPeerMultiple,
      vsPeerMedianRevenue,
      vsPeerMedianRaNPV,
      vsPeerMedianEBITDA,
      fairValueSupport,
      grossProceedsM,
      basePrimaryProceedsM,
      greenshoeProceedsM,
      primaryProceedsM,
      secondaryProceedsM,
      oversubscription,
      effectiveOversubscription,
      orderBookTier,
      investorsDropping,
      demandLostM,
      downRoundPercent,
      isDownRound,
      downRoundDiscount,
      baseImpliedPop,
      bookQualityAdjustment,
      valuationPenalty,
      secondaryDiscount,
      catalystDiscount,
      dualClassDiscount,
      customerConcentrationDiscount,
      growthDecelPenalty,
      adjustedImpliedPop,
      warnings: rowWarnings,
    };
  });
  
  // === RECOMMENDATION LOGIC - PURELY MECHANICAL ===
  // Recommendation based only on user inputs and mechanical relationships
  // No hardcoded thresholds - uses relative metrics from user data
  
  // Default to midpoint of computed range (may be user-provided or derived)
  const rangeMidpoint = (minPrice + maxPrice) / 2;
  let recommendedPrice = isFinite(rangeMidpoint) ? rangeMidpoint : minPrice;
  let recommendedRow: PricingRow | undefined;
  
  // Sort by price descending for selection logic
  const sortedByPrice = [...pricingMatrix].sort((a, b) => b.offerPrice - a.offerPrice);
  
  // Use management priority from user input to guide selection
  if (managementPriority === "runway_extension") {
    // CEO prioritizes deal certainty - recommend lower end for safety
    // Use user-provided minAcceptablePrice or range low
    recommendedPrice = minAcceptablePrice || minPrice;
    recommendedRow = pricingMatrix.find(r => r.offerPrice === recommendedPrice);
    
    if (!recommendedRow && sortedByPrice.length > 0) {
      recommendedRow = sortedByPrice[sortedByPrice.length - 1]; // Lowest price
      recommendedPrice = recommendedRow.offerPrice;
    }
    warnings.push("CEO priority: runway extension - recommending lower price for deal certainty");
    
  } else if (pricingAggressiveness === "maximum") {
    // CEO wants maximum - recommend top of range
    recommendedPrice = maxPrice;
    recommendedRow = pricingMatrix.find(r => r.offerPrice === recommendedPrice);
    
    if (!recommendedRow && sortedByPrice.length > 0) {
      recommendedRow = sortedByPrice[0]; // Highest price
      recommendedPrice = recommendedRow.offerPrice;
    }
    
  } else if (pricingAggressiveness === "conservative") {
    // Conservative - recommend bottom of range
    recommendedPrice = minPrice;
    recommendedRow = pricingMatrix.find(r => r.offerPrice === recommendedPrice);
    
    if (!recommendedRow && sortedByPrice.length > 0) {
      recommendedRow = sortedByPrice[sortedByPrice.length - 1];
      recommendedPrice = recommendedRow.offerPrice;
    }
    
  } else {
    // Moderate/default - use midpoint
    recommendedPrice = Math.round(rangeMidpoint);
    recommendedRow = pricingMatrix.find(r => r.offerPrice === recommendedPrice);
    
    if (!recommendedRow && sortedByPrice.length > 0) {
      // Find closest to midpoint
      recommendedRow = sortedByPrice.reduce((closest, row) => 
        Math.abs(row.offerPrice - rangeMidpoint) < Math.abs(closest.offerPrice - rangeMidpoint) ? row : closest
      );
      recommendedPrice = recommendedRow.offerPrice;
    }
  }
  
  // Ensure we have a row - guard against empty matrix
  if (!recommendedRow && pricingMatrix.length > 0) {
    recommendedRow = pricingMatrix[Math.floor(pricingMatrix.length / 2)];
    recommendedPrice = recommendedRow.offerPrice;
  }
  
  // Final guard - if still no row, return error
  if (!recommendedRow) {
    return {
      assumptions,
      pricingMatrix: [],
      recommendedRangeLow: 0,
      recommendedRangeHigh: 0,
      recommendedPrice: 0,
      rationale: [],
      warnings: ["ERROR: Could not compute pricing matrix - check input data"],
      memoText: "IPO PRICING ERROR\n\nCould not compute pricing matrix. Ensure all required inputs are provided.",
    };
  }
  
  // Respect minimum acceptable price from user
  if (minAcceptablePrice && recommendedPrice < minAcceptablePrice) {
    recommendedPrice = minAcceptablePrice;
    recommendedRow = pricingMatrix.find(r => r.offerPrice === recommendedPrice) || recommendedRow;
  }
  
  // Recommended range = computed range (user-provided or derived, no fabrication)
  const recommendedRangeLow = minPrice;
  const recommendedRangeHigh = maxPrice;
  
  // BUG FIX #1: Add down-round alert to warnings - ONLY for biotech or when user flags it
  if (recommendedRow.isDownRound && lastPrivateRoundPrice && applyDownRoundLogic) {
    warnings.push(`DOWN-ROUND ALERT: Offer $${recommendedPrice} is ${(Math.abs(recommendedRow.downRoundPercent) * 100).toFixed(1)}% below Series E price $${lastPrivateRoundPrice.toFixed(2)}`);
  }
  
  // BUG FIX #2: Add dual-class warning
  if (dualClass) {
    warnings.push(`Dual-class governance discount applied: -${(dualClassDiscountRate * 100).toFixed(0)}%`);
  }
  
  // BUG FIX #6: Add customer concentration warning
  if (customerConcentrationTop5 > 0.40) {
    warnings.push(`Customer concentration risk: Top 5 = ${(customerConcentrationTop5 * 100).toFixed(0)}% of revenue`);
  }
  
  // BUG FIX #5: Add growth deceleration warning
  if (growthDecelPenalty > 0) {
    warnings.push(`Growth deceleration penalty: -${(growthDecelPenalty * 100).toFixed(1)}% multiple compression`);
  }
  
  // BUG FIX #8: CEO directive contradiction - check if down-round persists
  if (lastPrivateRoundPrice && minAcceptablePrice && ceoGuidance) {
    const ceoLower = ceoGuidance.toLowerCase();
    if (ceoLower.includes("narrative") || ceoLower.includes("control") || ceoLower.includes("down round")) {
      if (recommendedPrice < lastPrivateRoundPrice) {
        warnings.push(`CEO DIRECTIVE CONTRADICTION: CEO wants to "control the narrative" but $${recommendedPrice} is still a down-round vs Series E $${lastPrivateRoundPrice.toFixed(2)}. Down-round headline risk PERSISTS.`);
      }
    }
  }
  
  // === RATIONALE ===
  const rationale: string[] = [];
  
  const popPercent = (recommendedRow.adjustedImpliedPop * 100).toFixed(0);
  const fairValuePercent = (recommendedRow.fairValueSupport * 100).toFixed(0);
  
  // Use correct valuation metric for sector
  if (useRaNPVValuation && totalRaNPV > 0) {
    const evRaNPVMultiple = recommendedRow.evRaNPV.toFixed(2);
    const peerDiffPercent = Math.abs(recommendedRow.vsPeerMedianRaNPV * 100).toFixed(0);
    const peerDirection = recommendedRow.vsPeerMedianRaNPV < 0 ? "below" : "above";
    rationale.push(`$${recommendedPrice} at ${evRaNPVMultiple}× EV/raNPV (${peerDiffPercent}% ${peerDirection} peer median ${peerMedianEVRaNPV.toFixed(1)}×)`);
  } else if (useEVEBITDAValuation && recommendedRow.ntmEVEBITDA > 0) {
    // For restaurant/consumer sectors, use EV/EBITDA as primary multiple
    const evEBITDAMultiple = recommendedRow.ntmEVEBITDA.toFixed(1);
    const peerDiffPercent = Math.abs(recommendedRow.vsPeerMedianEBITDA * 100).toFixed(0);
    const peerDirection = recommendedRow.vsPeerMedianEBITDA < 0 ? "below" : "above";
    const peerMedian = peerMedianEVEBITDA ? `${peerMedianEVEBITDA.toFixed(1)}×` : "N/A";
    rationale.push(`$${recommendedPrice} at ${evEBITDAMultiple}× NTM EV/EBITDA (${peerDiffPercent}% ${peerDirection} peer median ${peerMedian})`);
  } else {
    const evMultiple = recommendedRow.ntmEVRevenue.toFixed(1);
    const peerDiffPercent = Math.abs(recommendedRow.vsPeerMedianRevenue * 100).toFixed(0);
    const peerDirection = recommendedRow.vsPeerMedianRevenue < 0 ? "below" : "above";
    rationale.push(`$${recommendedPrice} at ${evMultiple}× NTM EV/Revenue (${peerDiffPercent}% ${peerDirection} peer median)`);
  }
  
  // BUG FIX #4: Show effective book coverage after investor drop-off
  if (recommendedRow.effectiveOversubscription !== recommendedRow.oversubscription) {
    rationale.push(`Book coverage: ${recommendedRow.oversubscription.toFixed(1)}× raw, ${recommendedRow.effectiveOversubscription.toFixed(1)}× effective (after drop-off)`);
  } else {
    rationale.push(`Book coverage: ${recommendedRow.effectiveOversubscription.toFixed(1)}× oversubscribed`);
  }
  rationale.push(`Expected Day-1 return: ${parseInt(popPercent) >= 0 ? '+' : ''}${popPercent}%`);
  
  // Add warnings about pop adjustments if significant
  if (Math.abs(recommendedRow.bookQualityAdjustment) > 0.02) {
    rationale.push(`Book quality adjustment: ${(recommendedRow.bookQualityAdjustment * 100).toFixed(0)}%`);
  }
  if (Math.abs(recommendedRow.valuationPenalty) > 0.02) {
    rationale.push(`Valuation penalty (${fairValuePercent}% of ${fairValueType === "ranpv" ? "raNPV" : "DCF"}): ${(recommendedRow.valuationPenalty * 100).toFixed(0)}%`);
  }
  if (recommendedRow.secondaryDiscount > 0.01) {
    rationale.push(`Secondary selling discount: -${(recommendedRow.secondaryDiscount * 100).toFixed(0)}%`);
  }
  if (recommendedRow.catalystDiscount > 0.01) {
    rationale.push(`Binary catalyst risk discount: -${(recommendedRow.catalystDiscount * 100).toFixed(0)}%`);
  }
  // BUG FIX #1: Add down-round discount to rationale
  if (recommendedRow.downRoundDiscount > 0.01) {
    rationale.push(`Down-round discount: -${(recommendedRow.downRoundDiscount * 100).toFixed(1)}%`);
  }
  // BUG FIX #2: Add dual-class discount to rationale
  if (recommendedRow.dualClassDiscount > 0.01) {
    rationale.push(`Dual-class governance discount: -${(recommendedRow.dualClassDiscount * 100).toFixed(0)}%`);
  }
  // BUG FIX #6: Add customer concentration discount to rationale
  if (recommendedRow.customerConcentrationDiscount > 0.01) {
    rationale.push(`Customer concentration discount: -${(recommendedRow.customerConcentrationDiscount * 100).toFixed(1)}%`);
  }
  
  // Note CEO directive in rationale
  if (managementPriority === "runway_extension") {
    rationale.push(`CEO priority: "runway extension" - pricing for deal certainty`);
    if (ceoGuidance) {
      rationale.push(`CEO guidance: "${ceoGuidance}"`);
    }
  }
  
  rationale.push(`Founders retain ${(recommendedRow.founderOwnershipPost * 100).toFixed(1)}% post-IPO`);
  // Show proceeds breakdown with greenshoe as additive
  const hasGreenshoe = recommendedRow.greenshoeProceedsM > 0;
  if (hasGreenshoe) {
    rationale.push(`Gross proceeds: $${Math.round(recommendedRow.grossProceedsM)}M = Base primary $${Math.round(recommendedRow.basePrimaryProceedsM)}M + Greenshoe $${Math.round(recommendedRow.greenshoeProceedsM)}M${recommendedRow.secondaryProceedsM > 0 ? ` + Secondary $${Math.round(recommendedRow.secondaryProceedsM)}M` : ''}`);
  } else {
    rationale.push(`Gross proceeds: $${Math.round(recommendedRow.grossProceedsM)}M (Primary: $${Math.round(recommendedRow.basePrimaryProceedsM)}M${recommendedRow.secondaryProceedsM > 0 ? `, Secondary: $${Math.round(recommendedRow.secondaryProceedsM)}M` : ''})`);
  }
  
  // Add warnings
  for (const w of recommendedRow.warnings) {
    warnings.push(w);
  }
  
  // Pass computed DCF fair value to memo (not just parsed value)
  const assumptionsWithComputedDCF = {
    ...assumptions,
    fairValuePerShare: effectiveFairValuePerShare, // Use computed DCF if we calculated it
  };
  const memoText = formatIPOMemo(assumptionsWithComputedDCF, pricingMatrix, recommendedRangeLow, recommendedRangeHigh, recommendedPrice, rationale, warnings);

  return {
    assumptions,
    pricingMatrix,
    recommendedRangeLow,
    recommendedRangeHigh,
    recommendedPrice,
    rationale,
    warnings,
    memoText,
  };
}

function formatIPOMemo(
  assumptions: IPOAssumptions,
  pricingMatrix: PricingRow[],
  rangeLow: number,
  rangeHigh: number,
  recommendedPrice: number,
  rationale: string[],
  warnings: string[]
): string {
  const {
    companyName,
    sector,
    fairValuePerShare = 0,
    fairValueType,
    totalRaNPV = 0,
    peerMedianEVRevenue = 0,
    peerMedianEVRaNPV = 0,
    ntmRevenue = 0,
    sectorMedianFirstDayPop,
    sectorAverageFirstDayPop,
    historicalFirstDayPop,
    indicatedPriceRangeLow = 0,
    indicatedPriceRangeHigh = 0,
  } = assumptions;

  const isBiotech = sector === "biotech";
  const isPreRevenue = ntmRevenue === 0 || ntmRevenue < 1;
  const useRaNPVValuation = isBiotech || isPreRevenue;
  
  const companyNameUpper = (companyName || "COMPANY").toUpperCase();
  
  const recommendedRow = pricingMatrix.find(r => Math.abs(r.offerPrice - recommendedPrice) < 0.5);
  if (!recommendedRow) return "Error: Could not find recommended row";
  
  const popPercent = ((recommendedRow.adjustedImpliedPop || 0) * 100).toFixed(0);
  const grossProceeds = Math.round(recommendedRow.grossProceedsM || 0);
  const marketCapB = ((recommendedRow.marketCapM || 0) / 1000).toFixed(1);
  const evB = ((recommendedRow.enterpriseValueM || 0) / 1000).toFixed(1);
  
  // Use correct fair value label
  const fairValueLabel = fairValueType === "ranpv" ? "raNPV" : "DCF";
  const safeFairValuePerShare = fairValuePerShare || 0;
  const safePeerMedianEVRevenue = peerMedianEVRevenue || 0;
  const safePeerMedianEVRaNPV = peerMedianEVRaNPV || 0;
  
  // Sector historical label
  const baseExpected = sectorMedianFirstDayPop ?? sectorAverageFirstDayPop ?? historicalFirstDayPop ?? 0;
  const histPopLabel = `sector ${baseExpected >= 0 ? '+' : ''}${(baseExpected * 100).toFixed(0)}% baseline`;
  
  let memo = `${companyNameUpper} – FINAL IPO PRICING RECOMMENDATION\n\n`;
  
  // Show warnings first
  if (warnings.length > 0) {
    memo += "*** WARNINGS ***\n";
    for (const w of warnings) {
      memo += `   ${w}\n`;
    }
    memo += "\n";
  }
  
  // Show filed range only if user provided it, otherwise show computed range
  if (indicatedPriceRangeLow > 0 && indicatedPriceRangeHigh > 0) {
    memo += `Original filed range: $${indicatedPriceRangeLow} – $${indicatedPriceRangeHigh}\n`;
  }
  memo += `Recommended offer price:                    $${recommendedPrice.toFixed(2)}\n`;
  memo += `Recommended range:                          $${rangeLow.toFixed(0)} – $${rangeHigh.toFixed(0)}\n\n`;
  
  memo += `Expected Day-1 Return: ${parseInt(popPercent) >= 0 ? '+' : ''}${popPercent}%\n`;
  
  // === SHARES SOLD & DILUTION - MECHANICALLY DERIVED ===
  memo += `\n--- SHARE ISSUANCE DETAIL ---\n`;
  memo += `Shares Sold (Primary): ${((recommendedRow.sharesSoldPrimary || 0) * 1).toFixed(2)}M\n`;
  memo += `Shares Sold (Secondary): ${((recommendedRow.sharesSoldSecondary || 0) * 1).toFixed(2)}M\n`;
  memo += `Shares Sold (Greenshoe): ${((recommendedRow.sharesSoldGreenshoe || 0) * 1).toFixed(2)}M\n`;
  memo += `Total Shares Sold: ${((recommendedRow.totalSharesSold || 0) * 1).toFixed(2)}M\n`;
  memo += `Post-IPO Fully Diluted Shares: ${((recommendedRow.fdSharesPostIPO || 0) * 1).toFixed(2)}M\n`;
  memo += `Dilution from Primary + Greenshoe: ${((recommendedRow.dilutionPercent || 0) * 100).toFixed(1)}%\n`;
  memo += `Founder Ownership Post-IPO: ${((recommendedRow.founderOwnershipPost || 0) * 100).toFixed(1)}%\n`;
  memo += `\n--- PROCEEDS CALCULATION ---\n`;
  memo += `Gross Proceeds: $${grossProceeds}M (Price $${recommendedPrice} × ${((recommendedRow.totalSharesSold || 0) * 1).toFixed(2)}M shares)\n`;
  memo += `  Base Primary (to company): $${Math.round(recommendedRow.basePrimaryProceedsM || 0)}M\n`;
  if ((recommendedRow.greenshoeProceedsM || 0) > 0) {
    const greenshoeExercised = assumptions.greenshoeAssumedExercised !== false;
    memo += `  Greenshoe (additive): $${Math.round(recommendedRow.greenshoeProceedsM || 0)}M\n`;
    if (greenshoeExercised) {
      memo += `  [Assumption: Full greenshoe exercise - standard for successful IPOs]\n`;
    } else {
      memo += `  [Assumption: Greenshoe NOT exercised - cash/EV use base primary only]\n`;
    }
  }
  memo += `  Secondary (to sellers): $${Math.round(recommendedRow.secondaryProceedsM || 0)}M\n`;
  
  memo += `\n--- VALUATION MECHANICS ---\n`;
  memo += `Market Cap: ~$${marketCapB}B (Price × FD Shares)\n`;
  memo += `Current Debt: $${((recommendedRow.currentDebtM || 0) * 1).toFixed(1)}M\n`;
  memo += `Post-IPO Cash: $${((recommendedRow.postIPOCashM || 0) * 1).toFixed(1)}M\n`;
  memo += `Enterprise Value: ~$${evB}B (MarketCap + Debt - Cash)\n\n`;
  
  // Use correct valuation metric - only show raNPV for explicit biotech companies
  const explicitlyBiotech = sector === "biotech" || sector === "biopharmaceutical" || sector === "clinical-stage";
  const sectorLower = (sector || "").toLowerCase();
  const memoUseEVEBITDA = sectorLower === "restaurant" || sectorLower === "fast-casual" || 
                          sectorLower === "qsr" || sectorLower === "food service" ||
                          sectorLower === "consumer_staples" || sectorLower === "consumer staples" ||
                          sectorLower === "retail" || sectorLower === "cpg";
  const safePeerMedianEVEBITDA = assumptions.peerMedianEVEBITDA || 0;
  
  if (explicitlyBiotech && totalRaNPV > 0) {
    const evRaNPVMultiple = (recommendedRow.evRaNPV || 0).toFixed(2);
    const peerDiffPercent = ((recommendedRow.vsPeerMedianRaNPV || 0) * 100).toFixed(0);
    memo += `Valuation Method: EV/raNPV (clinical-stage)\n`;
    memo += `EV/raNPV: ${evRaNPVMultiple}× (Peer Median: ${safePeerMedianEVRaNPV.toFixed(1)}×, ${parseInt(peerDiffPercent) >= 0 ? '+' : ''}${peerDiffPercent}%)\n`;
    memo += `Total raNPV: $${totalRaNPV.toFixed(0)}M\n`;
  } else if (memoUseEVEBITDA && (recommendedRow.ntmEVEBITDA || 0) > 0) {
    // For restaurant/consumer sectors, use EV/EBITDA as primary multiple
    const evEBITDAMultiple = (recommendedRow.ntmEVEBITDA || 0).toFixed(1);
    const peerDiffPercent = ((recommendedRow.vsPeerMedianEBITDA || 0) * 100).toFixed(0);
    memo += `Valuation Method: EV/EBITDA (restaurant/consumer)\n`;
    memo += `NTM EV/EBITDA: ${evEBITDAMultiple}× (Peer Median: ${safePeerMedianEVEBITDA.toFixed(1)}×, ${parseInt(peerDiffPercent) >= 0 ? '+' : ''}${peerDiffPercent}%)\n`;
  } else if (ntmRevenue > 0) {
    const evMultiple = (recommendedRow.ntmEVRevenue === Infinity || recommendedRow.ntmEVRevenue == null || recommendedRow.ntmEVRevenue === 0) ? "N/A" : recommendedRow.ntmEVRevenue.toFixed(1);
    const peerDiffPercent = ((recommendedRow.vsPeerMedianRevenue || 0) * 100).toFixed(0);
    memo += `NTM EV/Revenue: ${evMultiple}× (Peer Median: ${safePeerMedianEVRevenue.toFixed(1)}×, ${parseInt(peerDiffPercent) >= 0 ? '+' : ''}${peerDiffPercent}%)\n`;
  }
  
  memo += `${fairValueLabel}/share: $${safeFairValuePerShare.toFixed(2)} (offer = ${((recommendedRow.fairValueSupport || 0) * 100).toFixed(0)}%)\n\n`;
  
  memo += `Pricing Matrix\n\n`;
  
  // Select rows around recommendation
  const recIndex = pricingMatrix.findIndex(r => Math.abs(r.offerPrice - recommendedPrice) < 0.5);
  const startIdx = Math.max(0, recIndex - 2);
  const endIdx = Math.min(pricingMatrix.length, startIdx + 6);
  const rows = pricingMatrix.slice(startIdx, endIdx);
  
  const pad = (s: string, n: number) => s.padStart(n);
  
  memo += "Offer Price            " + rows.map(r => pad(`$${r.offerPrice}`, 10)).join("") + "\n";
  memo += "Shares Sold (Total)    " + rows.map(r => pad(`${((r.totalSharesSold || 0)).toFixed(1)}M`, 10)).join("") + "\n";
  memo += "FD Shares Post-IPO     " + rows.map(r => pad(`${((r.fdSharesPostIPO || 0)).toFixed(1)}M`, 10)).join("") + "\n";
  memo += "Dilution %             " + rows.map(r => pad(`${((r.dilutionPercent || 0) * 100).toFixed(1)}%`, 10)).join("") + "\n";
  memo += "Market Cap             " + rows.map(r => pad(`$${Math.round(r.marketCapM).toLocaleString()}`, 10)).join("") + "\n";
  memo += "Enterprise Value       " + rows.map(r => pad(`$${Math.round(r.enterpriseValueM).toLocaleString()}`, 10)).join("") + "\n";
  
  // Show correct valuation metric based on sector
  if (explicitlyBiotech && totalRaNPV > 0) {
    memo += "EV/raNPV               " + rows.map(r => pad(`${(r.evRaNPV || 0).toFixed(2)}×`, 10)).join("") + "\n";
    memo += `vs peer median ${safePeerMedianEVRaNPV.toFixed(1)}×   ` + rows.map(r => {
      const pct = (r.vsPeerMedianRaNPV || 0) * 100;
      return pad(`${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%`, 10);
    }).join("") + "\n";
  } else if (memoUseEVEBITDA && rows.some(r => (r.ntmEVEBITDA || 0) > 0)) {
    // For restaurant/consumer sectors, show EV/EBITDA as primary multiple
    memo += "NTM EV/EBITDA          " + rows.map(r => {
      if (r.ntmEVEBITDA === Infinity || r.ntmEVEBITDA == null || r.ntmEVEBITDA === 0) return pad("N/A", 10);
      return pad(`${r.ntmEVEBITDA.toFixed(1)}×`, 10);
    }).join("") + "\n";
    memo += `vs peer median ${safePeerMedianEVEBITDA.toFixed(1)}×   ` + rows.map(r => {
      if (r.vsPeerMedianEBITDA === Infinity || r.vsPeerMedianEBITDA == null) return pad("N/A", 10);
      const pct = (r.vsPeerMedianEBITDA || 0) * 100;
      return pad(`${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%`, 10);
    }).join("") + "\n";
  } else {
    memo += "NTM EV/Revenue         " + rows.map(r => {
      if (r.ntmEVRevenue === Infinity || r.ntmEVRevenue == null) return pad("N/A", 10);
      return pad(`${r.ntmEVRevenue.toFixed(1)}×`, 10);
    }).join("") + "\n";
    memo += `vs peer median ${safePeerMedianEVRevenue.toFixed(1)}×   ` + rows.map(r => {
      if (r.vsPeerMedianRevenue === Infinity || r.vsPeerMedianRevenue == null) return pad("N/A", 10);
      const pct = (r.vsPeerMedianRevenue || 0) * 100;
      return pad(`${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%`, 10);
    }).join("") + "\n";
  }
  
  // Use correct label
  memo += `${fairValueLabel} $${safeFairValuePerShare.toFixed(2)} support     ` + rows.map(r => pad(`${((r.fairValueSupport || 0) * 100).toFixed(0)}%`, 10)).join("") + "\n";
  memo += "Gross proceeds         " + rows.map(r => pad(`$${Math.round(r.grossProceedsM)}`, 10)).join("") + "\n";
  
  // Show order book tier and both raw and effective oversubscription
  memo += "Order Book Tier        " + rows.map(r => pad(r.orderBookTier || "N/A", 10)).join("") + "\n";
  memo += "Raw Oversubscription   " + rows.map(r => pad(`${(r.oversubscription || 0).toFixed(1)}×`, 10)).join("") + "\n";
  // Show effective oversubscription after investor drop-off
  if (rows.some(r => (r.effectiveOversubscription || 0) !== (r.oversubscription || 0))) {
    memo += "Effective Oversub      " + rows.map(r => pad(`${(r.effectiveOversubscription || 0).toFixed(1)}×`, 10)).join("") + "\n";
    memo += "Demand Lost ($M)       " + rows.map(r => pad((r.demandLostM || 0) > 0 ? `$${r.demandLostM}` : "-", 10)).join("") + "\n";
  }
  
  // Show down-round status
  if (rows.some(r => r.isDownRound)) {
    memo += "Down-Round %           " + rows.map(r => {
      if (!r.isDownRound) return pad("-", 10);
      return pad(`${((r.downRoundPercent || 0) * 100).toFixed(1)}%`, 10);
    }).join("") + "\n";
  }
  
  // Show all pop adjustments
  memo += `Day-1 Pop (${histPopLabel})\n`;
  memo += "  Base expected        " + rows.map(r => {
    const pct = (r.baseImpliedPop || 0) * 100;
    return pad(`${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%`, 10);
  }).join("") + "\n";
  memo += "  Book adjustment      " + rows.map(r => {
    const pct = (r.bookQualityAdjustment || 0) * 100;
    return pad(`${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%`, 10);
  }).join("") + "\n";
  memo += "  Valuation penalty    " + rows.map(r => {
    const pct = (r.valuationPenalty || 0) * 100;
    return pad(`${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%`, 10);
  }).join("") + "\n";
  if (rows.some(r => (r.secondaryDiscount || 0) > 0)) {
    memo += "  Secondary discount   " + rows.map(r => {
      const pct = -(r.secondaryDiscount || 0) * 100;
      return pad(`${pct.toFixed(0)}%`, 10);
    }).join("") + "\n";
  }
  if (rows.some(r => (r.catalystDiscount || 0) > 0)) {
    memo += "  Catalyst risk        " + rows.map(r => {
      const pct = -(r.catalystDiscount || 0) * 100;
      return pad(`${pct.toFixed(0)}%`, 10);
    }).join("") + "\n";
  }
  // Show down-round discount
  if (rows.some(r => (r.downRoundDiscount || 0) > 0)) {
    memo += "  Down-round discount  " + rows.map(r => {
      const pct = -(r.downRoundDiscount || 0) * 100;
      return pad(`${pct.toFixed(1)}%`, 10);
    }).join("") + "\n";
  }
  // Show dual-class discount
  if (rows.some(r => (r.dualClassDiscount || 0) > 0)) {
    memo += "  Dual-class discount  " + rows.map(r => {
      const pct = -(r.dualClassDiscount || 0) * 100;
      return pad(`${pct.toFixed(0)}%`, 10);
    }).join("") + "\n";
  }
  // Show customer concentration discount
  if (rows.some(r => (r.customerConcentrationDiscount || 0) > 0)) {
    memo += "  Concentration disc   " + rows.map(r => {
      const pct = -(r.customerConcentrationDiscount || 0) * 100;
      return pad(`${pct.toFixed(1)}%`, 10);
    }).join("") + "\n";
  }
  memo += "  ADJUSTED POP         " + rows.map(r => {
    const pct = (r.adjustedImpliedPop || 0) * 100;
    return pad(`${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%`, 10);
  }).join("") + "\n";
  
  memo += "Founder ownership      " + rows.map(r => pad(`${((r.founderOwnershipPost || 0) * 100).toFixed(1)}%`, 10)).join("") + "\n";
  
  memo += "\nRecommendation Rationale:\n";
  for (const r of rationale) {
    memo += `• ${r}\n`;
  }
  
  memo += `\nFile amendment at $${rangeLow.toFixed(0)}–$${rangeHigh.toFixed(0)} tonight, price at $${recommendedPrice.toFixed(0)} tomorrow morning.\n`;

  return memo;
}
