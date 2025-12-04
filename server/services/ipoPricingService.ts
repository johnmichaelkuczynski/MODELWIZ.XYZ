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
}

const IPO_PARSING_PROMPT = `You are an investment banking expert. Parse the IPO description and extract ALL parameters with precision.

CRITICAL PARSING RULES:

1. SECTOR DETECTION:
   - "biotech" / "biopharmaceutical" / "clinical-stage" / "Phase" → sector = "biotech"
   - "SaaS" / "enterprise software" → sector = "saas"
   - "AI infrastructure" / "GPU cloud" → sector = "ai_infrastructure"
   - "defense-tech" / "national security" → sector = "defense_tech"

2. VALUATION TYPE - CRITICAL FOR BIOTECH:
   - If input mentions "risk-adjusted NPV" / "raNPV" / "rNPV" → fairValueType = "ranpv"
   - Parse "ranpv_per_share" or "raNPV/share" as fairValuePerShare
   - Parse "total_ranpv" in millions as totalRaNPV
   - If input has "dcf" / "DCF valuation" → fairValueType = "dcf"

3. PEER COMPS FOR BIOTECH:
   - If "median EV/raNPV" or "EV/rNPV" is given, set peerMedianEVRaNPV (e.g., 2.4)
   - Regular EV/Revenue goes to peerMedianEVRevenue

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

9. REVENUE:
   - For pre-revenue biotech, ntmRevenue = 0
   - Parse FY2026 guidance if present

10. DOWN-ROUND DETECTION (CRITICAL):
   - Parse "last_private_round.price_per_share" or "Series E price" → lastPrivateRoundPrice
   - Parse "risk_factors.down_round_optics" → downRoundOptics (true if mentioned)
   - Parse "down_round_ipo_penalty.avg_additional_discount" → downRoundIpoPenalty (ONLY if explicitly provided, NO DEFAULT)

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

Return JSON:
{
  "companyName": "string",
  "filingDate": "YYYY-MM-DD",
  "sector": "biotech" | "saas" | "ai_infrastructure" | "defense_tech" | "tech",
  
  "sharesOutstandingPreIPO": number (millions),
  "primarySharesOffered": number (millions),
  "secondarySharesOffered": number (millions, default 0),
  "greenshoeShares": number (millions),
  "greenshoePercent": number (decimal),
  
  "targetGrossProceeds": number (millions),
  "primaryDollarRaiseM": number (millions, primary proceeds target),
  "secondaryDollarRaiseM": number (millions, secondary proceeds if any),
  "indicatedPriceRangeLow": number,
  "indicatedPriceRangeHigh": number,
  
  "currentCash": number (millions),
  "currentDebt": number (millions, total debt on balance sheet),
  
  "currentYearRevenue": number (millions),
  "ntmRevenue": number (millions - 0 for pre-revenue),
  "ntmRevenueGrowth": number (decimal),
  "ntmEBITDA": number (millions),
  "ntmEBITDAMargin": number (decimal),
  
  "fairValuePerShare": number,
  "fairValueType": "dcf" | "ranpv",
  "totalRaNPV": number (millions, if provided),
  
  "peerMedianEVRevenue": number,
  "peerMedianEVRaNPV": number (if biotech),
  
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
  
  "customerConcentrationTop5": number (decimal, e.g., 0.47 for 47%)
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
  evRaNPV: number;
  growthAdjustedMultiple: number; // BUG FIX #5: peer multiple adjusted for deceleration
  
  vsPeerMedianRevenue: number;
  vsPeerMedianRaNPV: number;
  
  fairValueSupport: number;
  grossProceedsM: number;
  primaryProceedsM: number; // BUG FIX #7: proceeds to company
  secondaryProceedsM: number; // BUG FIX #7: proceeds to sellers
  
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
  } = assumptions;

  const isBiotech = sector === "biotech";
  const isPreRevenue = ntmRevenue === 0 || ntmRevenue < 1;
  const useRaNPVValuation = isBiotech || isPreRevenue;
  
  const warnings: string[] = [];
  
  // Determine if we're using dollar-based or share-based inputs
  const useDollarBased = (primaryDollarRaiseM !== undefined && primaryDollarRaiseM > 0);
  
  // BUG FIX #5: Calculate growth deceleration penalty for peer multiple
  // MECHANICAL: compression equals the deceleration rate itself (no embedded multiplier)
  let growthDecelPenalty = 0;
  let growthAdjustedPeerMultiple = peerMedianEVRevenue;
  if (growthRates && growthRates.fy2024to2025Growth && growthRates.fy2025to2026Growth) {
    const decelRate = 1 - (growthRates.fy2025to2026Growth / growthRates.fy2024to2025Growth);
    if (decelRate > 0) {
      // MECHANICAL: growth decel penalty = the deceleration rate itself
      // If growth decelerates by 30%, multiple compresses by 30%
      growthDecelPenalty = decelRate;
      growthAdjustedPeerMultiple = peerMedianEVRevenue * (1 - growthDecelPenalty);
    }
  }
  
  // Determine price range - REQUIRE USER INPUT STRICTLY
  // If user doesn't provide range, pricing calculation cannot proceed correctly
  const hasUserProvidedRange = (indicatedPriceRangeLow !== undefined && indicatedPriceRangeLow > 0) && 
                                (indicatedPriceRangeHigh !== undefined && indicatedPriceRangeHigh > 0);
  
  if (!hasUserProvidedRange) {
    // SHORT-CIRCUIT: Cannot proceed without user-provided price range
    const errorWarning = "ERROR: Price range (indicatedPriceRangeLow/High) is REQUIRED - cannot price without user-provided range. Please provide both indicatedPriceRangeLow and indicatedPriceRangeHigh.";
    return {
      assumptions, // Return original assumptions
      pricingMatrix: [],
      recommendedRangeLow: 0,
      recommendedRangeHigh: 0,
      recommendedPrice: 0,
      rationale: [],
      warnings: [errorWarning],
      memoText: `IPO PRICING ERROR\n\n${errorWarning}`,
    };
  }
  
  // Guard for inverted or zero range
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
  
  // Use user-provided values strictly - no fabrication
  let minPrice = indicatedPriceRangeLow;
  let maxPrice = indicatedPriceRangeHigh;
  
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
    
    const totalSharesSold = sharesSoldPrimary + sharesSoldSecondary + sharesSoldGreenshoe;
    
    // === FULLY DILUTED SHARES - RECOMPUTED AT EACH PRICE POINT ===
    // Only primary shares and greenshoe are dilutive (secondary is existing shares)
    const dilutiveShares = sharesSoldPrimary + sharesSoldGreenshoe;
    const fdSharesPostIPO = sharesOutstandingPreIPO + dilutiveShares;
    
    // === DILUTION CALCULATION - MECHANICAL ===
    const dilutionPercent = dilutiveShares / fdSharesPostIPO;
    
    // === PROCEEDS CALCULATION - MECHANICAL: Price × Shares ===
    const primaryProceedsM = offerPrice * (sharesSoldPrimary + sharesSoldGreenshoe);
    const secondaryProceedsM = offerPrice * sharesSoldSecondary;
    const grossProceedsM = primaryProceedsM + secondaryProceedsM;
    
    // === MARKET CAP - MECHANICAL: Price × FD Shares ===
    const marketCapM = fdSharesPostIPO * offerPrice;
    
    // === CASH POSITION POST-IPO ===
    // Post-IPO Cash = Current Cash + Primary Proceeds (secondary goes to sellers)
    const postIPOCashM = currentCash + primaryProceedsM;
    
    // === ENTERPRISE VALUE - CORRECT FORMULA: EV = MarketCap + Debt - Cash ===
    const currentDebtM = currentDebt;
    const enterpriseValueM = marketCapM + currentDebtM - postIPOCashM;
    
    // === VALUATION MULTIPLES ===
    // NTM EV/Revenue - with consistent rounding
    const ntmEVRevenue = isPreRevenue ? Infinity : Math.round((enterpriseValueM / ntmRevenue) * 10) / 10;
    
    // EV/raNPV for biotech - with consistent rounding
    const evRaNPV = totalRaNPV > 0 ? Math.round((enterpriseValueM / totalRaNPV) * 100) / 100 : 0;
    
    // vs Peer Median comparisons (using growth-adjusted multiple for revenue comps)
    const vsPeerMedianRevenue = isPreRevenue ? Infinity : 
      Math.round(((ntmEVRevenue - growthAdjustedPeerMultiple) / growthAdjustedPeerMultiple) * 1000) / 1000;
    const vsPeerMedianRaNPV = (totalRaNPV > 0 && peerMedianEVRaNPV > 0) 
      ? Math.round(((evRaNPV - peerMedianEVRaNPV) / peerMedianEVRaNPV) * 1000) / 1000
      : 0;
    
    // === FAIR VALUE SUPPORT - CONSISTENT ROUNDING ===
    const fairValueSupport = Math.round((offerPrice / fairValuePerShare) * 1000) / 1000;
    
    // === ORDER BOOK LOOKUP - NO FABRICATED EXTRAPOLATION ===
    // Use ONLY explicit user-provided tiers, no invented decay rates
    let oversubscription = 1; // Neutral default if no order book
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
    // MECHANICAL: use user-provided downRoundIpoPenalty coefficient
    let downRoundPercent = 0;
    let isDownRound = false;
    let downRoundDiscount = 0;
    if (lastPrivateRoundPrice && lastPrivateRoundPrice > 0) {
      downRoundPercent = (offerPrice - lastPrivateRoundPrice) / lastPrivateRoundPrice;
      isDownRound = downRoundPercent < 0;
      if (isDownRound && downRoundOptics) {
        // MECHANICAL: use user-provided penalty coefficient, or the absolute discount itself
        // downRoundIpoPenalty from user determines pass-through rate
        downRoundDiscount = Math.abs(downRoundPercent) * (downRoundIpoPenalty || 0);
      }
    }
    
    // === IMPLIED POP CALCULATION - FULLY MECHANICAL FROM USER INPUTS ===
    // POP is ONLY computed if user provided sector benchmark data
    // All adjustments use user-provided values directly - NO embedded constants
    
    let baseImpliedPop = 0;
    let bookQualityAdjustment = 0;
    let valuationPenalty = 0;
    let secondaryDiscount = 0;
    let catalystDiscount = 0;
    let adjustedImpliedPop = 0;
    
    if (hasUserProvidedPopData) {
      // Base implied pop = user-provided sector historical data (no modification)
      baseImpliedPop = baseExpectedReturn;
      
      // Book quality adjustment - ONLY if order book data provided
      // PURELY MECHANICAL: log of oversubscription ratio directly
      // No embedded multipliers - natural log relationship
      if (hasExplicitOrderBook && effectiveOversubscription > 0 && effectiveOversubscription !== 1) {
        // MECHANICAL: natural log gives proportional relationship
        // 2x oversub = +0.69, 0.5x = -0.69 (symmetric around 1x)
        bookQualityAdjustment = Math.log(effectiveOversubscription);
      }
      
      // Valuation penalty - PURELY MECHANICAL
      // Pricing above fair value directly reduces expected return
      if (fairValueSupport > 1) {
        valuationPenalty = -(fairValueSupport - 1);
      }
      
      // Secondary discount - PURELY MECHANICAL from user optics input
      // Discount only applies when user specifies negative optics
      if (sharesSoldSecondary > 0 && totalSharesSold > 0) {
        const secondaryPct = sharesSoldSecondary / totalSharesSold;
        // MECHANICAL: discount = secondary % only when optics is negative
        secondaryDiscount = secondaryOptics === "negative" ? secondaryPct : 0;
      }
      
      // Binary catalyst discount - PURELY MECHANICAL from user months
      // No fabricated caps - just inverse relationship
      // Only applies if user provides both hasBinaryCatalyst and monthsToCatalyst
      if (hasBinaryCatalyst && monthsToCatalyst !== undefined && monthsToCatalyst > 0) {
        // MECHANICAL: 1/months gives natural decay (no cap)
        catalystDiscount = 1 / monthsToCatalyst;
      }
      
      // Total adjusted implied pop - ALL from user inputs
      adjustedImpliedPop = baseImpliedPop 
        + bookQualityAdjustment 
        + valuationPenalty 
        - secondaryDiscount 
        - catalystDiscount
        - downRoundDiscount      // From user's lastPrivateRoundPrice
        - (dualClass ? dualClassDiscountRate : 0)      // From user's dualClassDiscount
        - (customerConcentrationTop5 > 0.40 ? (customerConcentrationTop5 - 0.40) : 0); // From user's concentration
    }
    // If no user POP data provided, all values remain 0 (neutral/omitted)
    
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
    const founderSharesFixed = foundersEmployeesOwnership * sharesOutstandingPreIPO;
    const founderOwnershipPost = founderSharesFixed / fdSharesPostIPO;
    
    // Generate warnings - ONLY based on user-provided data, no hardcoded thresholds
    // Fair value warning - only if fairValueSupport was calculated from user data
    if (fairValueSupport > 1) {
      rowWarnings.push(`Valuation: ${(fairValueSupport * 100).toFixed(0)}% of estimated fair value`);
    }
    // Book coverage - report actual metric without judgment threshold
    if (hasExplicitOrderBook && effectiveOversubscription > 0) {
      rowWarnings.push(`Book coverage: ${effectiveOversubscription.toFixed(1)}× effective oversubscription`);
    }
    // Implied POP - report actual value without judgment threshold
    if (baseImpliedPop !== 0) {
      rowWarnings.push(`Implied POP: ${(adjustedImpliedPop * 100).toFixed(1)}%`);
    }
    // Down-round - factual from user's lastPrivateRoundPrice
    if (isDownRound) {
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
      evRaNPV,
      growthAdjustedMultiple: growthAdjustedPeerMultiple,
      vsPeerMedianRevenue,
      vsPeerMedianRaNPV,
      fairValueSupport,
      grossProceedsM,
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
  
  // Default to midpoint of user-provided range
  const rangeMidpoint = (indicatedPriceRangeLow + indicatedPriceRangeHigh) / 2;
  let recommendedPrice = rangeMidpoint;
  let recommendedRow: PricingRow | undefined;
  
  // Sort by price descending for selection logic
  const sortedByPrice = [...pricingMatrix].sort((a, b) => b.offerPrice - a.offerPrice);
  
  // Use management priority from user input to guide selection
  if (managementPriority === "runway_extension") {
    // CEO prioritizes deal certainty - recommend lower end for safety
    // Use user-provided minAcceptablePrice or range low
    recommendedPrice = minAcceptablePrice || indicatedPriceRangeLow;
    recommendedRow = pricingMatrix.find(r => r.offerPrice === recommendedPrice);
    
    if (!recommendedRow) {
      recommendedRow = sortedByPrice[sortedByPrice.length - 1]; // Lowest price
      recommendedPrice = recommendedRow.offerPrice;
    }
    warnings.push("CEO priority: runway extension - recommending lower price for deal certainty");
    
  } else if (pricingAggressiveness === "maximum") {
    // CEO wants maximum - recommend top of range
    recommendedPrice = indicatedPriceRangeHigh;
    recommendedRow = pricingMatrix.find(r => r.offerPrice === recommendedPrice);
    
    if (!recommendedRow) {
      recommendedRow = sortedByPrice[0]; // Highest price
      recommendedPrice = recommendedRow.offerPrice;
    }
    
  } else if (pricingAggressiveness === "conservative") {
    // Conservative - recommend bottom of range
    recommendedPrice = indicatedPriceRangeLow;
    recommendedRow = pricingMatrix.find(r => r.offerPrice === recommendedPrice);
    
    if (!recommendedRow) {
      recommendedRow = sortedByPrice[sortedByPrice.length - 1];
      recommendedPrice = recommendedRow.offerPrice;
    }
    
  } else {
    // Moderate/default - use midpoint
    recommendedPrice = Math.round(rangeMidpoint);
    recommendedRow = pricingMatrix.find(r => r.offerPrice === recommendedPrice);
    
    if (!recommendedRow) {
      // Find closest to midpoint
      recommendedRow = sortedByPrice.reduce((closest, row) => 
        Math.abs(row.offerPrice - rangeMidpoint) < Math.abs(closest.offerPrice - rangeMidpoint) ? row : closest
      );
      recommendedPrice = recommendedRow.offerPrice;
    }
  }
  
  // Ensure we have a row
  if (!recommendedRow) {
    recommendedRow = pricingMatrix[Math.floor(pricingMatrix.length / 2)];
    recommendedPrice = recommendedRow.offerPrice;
  }
  
  // Respect minimum acceptable price from user
  if (minAcceptablePrice && recommendedPrice < minAcceptablePrice) {
    recommendedPrice = minAcceptablePrice;
    recommendedRow = pricingMatrix.find(r => r.offerPrice === recommendedPrice) || recommendedRow;
  }
  
  // Recommended range = user-provided range (no fabricated offsets)
  const recommendedRangeLow = indicatedPriceRangeLow;
  const recommendedRangeHigh = indicatedPriceRangeHigh;
  
  // BUG FIX #1: Add down-round alert to warnings
  if (recommendedRow.isDownRound && lastPrivateRoundPrice) {
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
  // BUG FIX #7: Show primary vs secondary proceeds
  rationale.push(`Gross proceeds: $${Math.round(recommendedRow.grossProceedsM)}M (Primary to company: $${Math.round(recommendedRow.primaryProceedsM)}M, Secondary to sellers: $${Math.round(recommendedRow.secondaryProceedsM)}M)`);
  
  // Add warnings
  for (const w of recommendedRow.warnings) {
    warnings.push(w);
  }
  
  const memoText = formatIPOMemo(assumptions, pricingMatrix, recommendedRangeLow, recommendedRangeHigh, recommendedPrice, rationale, warnings);

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
  
  memo += `Filed range: $${indicatedPriceRangeLow} – $${indicatedPriceRangeHigh}\n`;
  memo += `Recommended offer price:                    $${recommendedPrice.toFixed(2)}\n`;
  memo += `Recommended amendment range:                $${rangeLow.toFixed(2)} – $${rangeHigh.toFixed(2)}\n\n`;
  
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
  memo += `  Primary (to company): $${Math.round(recommendedRow.primaryProceedsM || 0)}M\n`;
  memo += `  Secondary (to sellers): $${Math.round(recommendedRow.secondaryProceedsM || 0)}M\n`;
  
  memo += `\n--- VALUATION MECHANICS ---\n`;
  memo += `Market Cap: ~$${marketCapB}B (Price × FD Shares)\n`;
  memo += `Current Debt: $${((recommendedRow.currentDebtM || 0) * 1).toFixed(1)}M\n`;
  memo += `Post-IPO Cash: $${((recommendedRow.postIPOCashM || 0) * 1).toFixed(1)}M\n`;
  memo += `Enterprise Value: ~$${evB}B (MarketCap + Debt - Cash)\n\n`;
  
  // Use correct valuation metric
  if (useRaNPVValuation && totalRaNPV > 0) {
    const evRaNPVMultiple = (recommendedRow.evRaNPV || 0).toFixed(2);
    const peerDiffPercent = ((recommendedRow.vsPeerMedianRaNPV || 0) * 100).toFixed(0);
    memo += `Valuation Method: EV/raNPV (biotech/pre-revenue)\n`;
    memo += `EV/raNPV: ${evRaNPVMultiple}× (Peer Median: ${safePeerMedianEVRaNPV.toFixed(1)}×, ${parseInt(peerDiffPercent) >= 0 ? '+' : ''}${peerDiffPercent}%)\n`;
    memo += `Total raNPV: $${totalRaNPV.toFixed(0)}M\n`;
  } else {
    const evMultiple = (recommendedRow.ntmEVRevenue === Infinity || recommendedRow.ntmEVRevenue == null) ? "N/A" : recommendedRow.ntmEVRevenue.toFixed(1);
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
  
  // Show correct valuation metric
  if (useRaNPVValuation && totalRaNPV > 0) {
    memo += "EV/raNPV               " + rows.map(r => pad(`${(r.evRaNPV || 0).toFixed(2)}×`, 10)).join("") + "\n";
    memo += `vs peer median ${safePeerMedianEVRaNPV.toFixed(1)}×   ` + rows.map(r => {
      const pct = (r.vsPeerMedianRaNPV || 0) * 100;
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
