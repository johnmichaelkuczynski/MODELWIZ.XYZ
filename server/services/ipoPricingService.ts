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
  
  targetGrossProceeds: number;
  indicatedPriceRangeLow: number;
  indicatedPriceRangeHigh: number;
  
  currentCash: number;
  
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
  "indicatedPriceRangeLow": number,
  "indicatedPriceRangeHigh": number,
  
  "currentCash": number (millions),
  
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
  
  "historicalFirstDayPop": number (decimal),
  "sectorAverageFirstDayPop": number (decimal, can be negative),
  "sectorMedianFirstDayPop": number (decimal, can be negative),
  
  "foundersEmployeesOwnership": number (decimal),
  "vcPeOwnership": number (decimal),
  
  "underwritingFeePercent": number (default 0.07),
  
  "ceoGuidance": "exact quote",
  "boardGuidance": "exact quote",
  "pricingAggressiveness": "conservative" | "moderate" | "aggressive" | "maximum",
  "managementPriority": "valuation_maximization" | "runway_extension" | "deal_certainty",
  "minAcceptablePrice": number,
  
  "hasBinaryCatalyst": boolean,
  "monthsToCatalyst": number,
  "catalystDescription": "string",
  
  "secondaryOptics": "neutral" | "negative" | "positive"
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
  
  // Set defaults
  if (!assumptions.currentCash) assumptions.currentCash = 0;
  if (!assumptions.secondarySharesOffered) assumptions.secondarySharesOffered = 0;
  if (!assumptions.fairValueType) assumptions.fairValueType = "dcf";
  
  // Derive aggressiveness and priority from guidance
  const ceoLower = (assumptions.ceoGuidance || "").toLowerCase();
  
  if (!assumptions.pricingAggressiveness) {
    if (ceoLower.includes("biggest") || ceoLower.includes("maximum") || 
        ceoLower.includes("absolute limit") || ceoLower.includes("history")) {
      assumptions.pricingAggressiveness = "maximum";
    } else if (ceoLower.includes("rather") && ceoLower.includes("get the deal done")) {
      assumptions.pricingAggressiveness = "conservative";
    } else {
      assumptions.pricingAggressiveness = "moderate";
    }
  }
  
  if (!assumptions.managementPriority) {
    if (ceoLower.includes("runway") || ceoLower.includes("get the deal done") || 
        ceoLower.includes("rather price at") || ceoLower.includes("certainty")) {
      assumptions.managementPriority = "runway_extension";
    } else if (ceoLower.includes("biggest") || ceoLower.includes("maximum")) {
      assumptions.managementPriority = "valuation_maximization";
    }
  }
  
  return { assumptions, providerUsed };
}

interface PricingRow {
  offerPrice: number;
  fdSharesPostIPO: number;
  marketCapM: number;
  postIPOCashM: number;
  enterpriseValueM: number;
  
  ntmEVRevenue: number;
  evRaNPV: number;
  
  vsPeerMedianRevenue: number;
  vsPeerMedianRaNPV: number;
  
  fairValueSupport: number;
  grossProceedsM: number;
  oversubscription: number;
  orderBookTier: string; // e.g., "$24+", "$22+", "<$18"
  
  baseImpliedPop: number;
  bookQualityAdjustment: number;
  valuationPenalty: number;
  secondaryDiscount: number;
  catalystDiscount: number;
  adjustedImpliedPop: number;
  
  founderOwnershipPost: number;
  
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
    primarySharesOffered,
    secondarySharesOffered = 0,
    greenshoeShares,
    greenshoePercent,
    currentCash = 0,
    ntmRevenue,
    fairValuePerShare,
    fairValueType,
    totalRaNPV = 0,
    peerMedianEVRevenue,
    peerMedianEVRaNPV = 0,
    orderBook,
    sectorMedianFirstDayPop,
    sectorAverageFirstDayPop,
    historicalFirstDayPop,
    foundersEmployeesOwnership,
    pricingAggressiveness,
    managementPriority,
    minAcceptablePrice,
    ceoGuidance,
    hasBinaryCatalyst = false,
    monthsToCatalyst = 12,
    secondaryOptics = "neutral",
    indicatedPriceRangeLow,
    indicatedPriceRangeHigh,
  } = assumptions;

  const isBiotech = sector === "biotech";
  const isPreRevenue = ntmRevenue === 0 || ntmRevenue < 1;
  const useRaNPVValuation = isBiotech || isPreRevenue;
  
  const warnings: string[] = [];

  // Calculate greenshoe
  const actualGreenshoeShares = greenshoeShares || (primarySharesOffered * greenshoePercent);
  const totalSharesForProceeds = primarySharesOffered + secondarySharesOffered + actualGreenshoeShares;
  const fdSharesPostIPO = sharesOutstandingPreIPO + primarySharesOffered + actualGreenshoeShares;
  
  // Determine price range
  let minPrice = indicatedPriceRangeLow || 15;
  let maxPrice = indicatedPriceRangeHigh || 30;
  
  if (orderBook && orderBook.length > 0) {
    const bookPrices = orderBook.map(ob => ob.priceLevel);
    minPrice = Math.min(minPrice, ...bookPrices) - 3;
    maxPrice = Math.max(maxPrice, ...bookPrices) + 3;
  }
  
  const pricePoints: number[] = [];
  for (let p = minPrice; p <= maxPrice; p += 1) {
    if (p > 0) pricePoints.push(p);
  }
  
  // Sort order book by price DESCENDING
  const sortedOrderBook = orderBook ? [...orderBook].sort((a, b) => b.priceLevel - a.priceLevel) : [];
  
  // Determine base expected return for sector
  const baseExpectedReturn = sectorMedianFirstDayPop ?? sectorAverageFirstDayPop ?? historicalFirstDayPop ?? 0;
  
  const pricingMatrix: PricingRow[] = pricePoints.map(offerPrice => {
    const rowWarnings: string[] = [];
    
    // Gross Proceeds
    const grossProceedsM = offerPrice * totalSharesForProceeds;
    
    // Market Cap
    const marketCapM = fdSharesPostIPO * offerPrice;
    
    // Post-IPO Cash = Current Cash + Primary Proceeds (secondary goes to sellers)
    const primaryProceeds = offerPrice * (primarySharesOffered + actualGreenshoeShares);
    const postIPOCashM = currentCash + primaryProceeds;
    
    // Enterprise Value = Market Cap - Post-IPO Cash
    const enterpriseValueM = marketCapM - postIPOCashM;
    
    // NTM EV/Revenue
    const ntmEVRevenue = isPreRevenue ? Infinity : enterpriseValueM / ntmRevenue;
    
    // EV/raNPV for biotech
    const evRaNPV = totalRaNPV > 0 ? enterpriseValueM / totalRaNPV : 0;
    
    // vs Peer Median comparisons
    const vsPeerMedianRevenue = isPreRevenue ? Infinity : (ntmEVRevenue - peerMedianEVRevenue) / peerMedianEVRevenue;
    const vsPeerMedianRaNPV = (totalRaNPV > 0 && peerMedianEVRaNPV > 0) 
      ? (evRaNPV - peerMedianEVRaNPV) / peerMedianEVRaNPV 
      : 0;
    
    // Fair value support
    const fairValueSupport = offerPrice / fairValuePerShare;
    
    // Order book lookup with tier tracking
    let oversubscription = 1;
    let orderBookTier = "";
    if (sortedOrderBook.length > 0) {
      for (const entry of sortedOrderBook) {
        if (offerPrice >= entry.priceLevel) {
          oversubscription = entry.oversubscription;
          orderBookTier = `$${entry.priceLevel}+`;
          break;
        }
      }
      if (oversubscription === 1) {
        const lowestEntry = sortedOrderBook[sortedOrderBook.length - 1];
        const priceDiff = lowestEntry.priceLevel - offerPrice;
        oversubscription = Math.round(lowestEntry.oversubscription * (1 + priceDiff * 0.05));
        orderBookTier = `<$${lowestEntry.priceLevel}`;
      }
    }
    
    // === IMPLIED POP CALCULATION WITH ALL ADJUSTMENTS ===
    
    // Base implied pop from sector historical
    let baseImpliedPop = baseExpectedReturn;
    
    // BUG FIX #3: Book quality adjustment - weak book = lower pop
    let bookQualityAdjustment = 0;
    if (oversubscription < 5) {
      // -3% per turn under 5×
      bookQualityAdjustment = (5 - oversubscription) * -0.03;
    } else if (oversubscription > 20) {
      // Bonus for very strong book
      bookQualityAdjustment = Math.min(0.10, (oversubscription - 20) * 0.005);
    }
    
    // BUG FIX #3: Valuation penalty - pricing above fair value hurts pop
    let valuationPenalty = 0;
    if (fairValueSupport > 1) {
      // Premium to fair value penalizes expected return
      valuationPenalty = (fairValueSupport - 1) * -0.15;
    }
    
    // BUG FIX #5: Secondary selling discount
    let secondaryDiscount = 0;
    if (secondarySharesOffered > 0) {
      const secondaryPct = secondarySharesOffered / totalSharesForProceeds;
      if (secondaryOptics === "negative") {
        secondaryDiscount = 0.03 + (secondaryPct * 0.05); // 3-8% discount
      } else {
        secondaryDiscount = secondaryPct * 0.02; // Small discount regardless
      }
    }
    
    // BUG FIX #6: Binary catalyst risk discount
    let catalystDiscount = 0;
    if (hasBinaryCatalyst) {
      if (monthsToCatalyst < 6) {
        catalystDiscount = 0.10; // 10% discount for near-term binary
      } else if (monthsToCatalyst < 12) {
        catalystDiscount = 0.05;
      }
    }
    
    // Total adjusted implied pop
    const adjustedImpliedPop = baseImpliedPop + bookQualityAdjustment + valuationPenalty - secondaryDiscount - catalystDiscount;
    
    // Founder ownership post-IPO
    const founderOwnershipPost = foundersEmployeesOwnership * (sharesOutstandingPreIPO / fdSharesPostIPO);
    
    // Generate warnings
    if (fairValueSupport > 2) {
      rowWarnings.push(`WARNING: ${(fairValueSupport * 100).toFixed(0)}% of fair value`);
    }
    if (oversubscription < 3) {
      rowWarnings.push(`WARNING: Weak book coverage (${oversubscription}×)`);
    }
    if (adjustedImpliedPop < -0.20) {
      rowWarnings.push(`WARNING: High probability of negative Day-1 return`);
    }
    
    return {
      offerPrice,
      fdSharesPostIPO,
      marketCapM,
      postIPOCashM,
      enterpriseValueM,
      ntmEVRevenue,
      evRaNPV,
      vsPeerMedianRevenue,
      vsPeerMedianRaNPV,
      fairValueSupport,
      grossProceedsM,
      oversubscription,
      orderBookTier,
      baseImpliedPop,
      bookQualityAdjustment,
      valuationPenalty,
      secondaryDiscount,
      catalystDiscount,
      adjustedImpliedPop,
      founderOwnershipPost,
      warnings: rowWarnings,
    };
  });
  
  // === RECOMMENDATION LOGIC ===
  
  let recommendedPrice = indicatedPriceRangeLow || pricingMatrix[Math.floor(pricingMatrix.length / 2)].offerPrice;
  let recommendedRow: PricingRow | undefined;
  
  // BUG FIX #4: Respect CEO directive when book is weak
  const topOfRangeRow = pricingMatrix.find(r => r.offerPrice === indicatedPriceRangeHigh);
  const topOfRangeOversubscription = topOfRangeRow?.oversubscription || 0;
  
  if (managementPriority === "runway_extension" && topOfRangeOversubscription < 5) {
    // CEO wants deal certainty and book is weak at top - find comfortable coverage price
    warnings.push("CEO priority is runway extension with weak book at top of range");
    
    // Find price where book is comfortably covered (>5×)
    const sortedByPrice = [...pricingMatrix].sort((a, b) => b.offerPrice - a.offerPrice);
    for (const row of sortedByPrice) {
      if (row.oversubscription >= 5) {
        recommendedPrice = row.offerPrice;
        recommendedRow = row;
        break;
      }
    }
    
    // Never recommend above midpoint when book is weak and CEO wants certainty
    const midpoint = (indicatedPriceRangeLow + indicatedPriceRangeHigh) / 2;
    if (recommendedPrice > midpoint) {
      recommendedPrice = Math.floor(midpoint);
      recommendedRow = pricingMatrix.find(r => r.offerPrice === recommendedPrice);
    }
    
    // Respect minimum acceptable price
    if (minAcceptablePrice && recommendedPrice < minAcceptablePrice) {
      recommendedPrice = minAcceptablePrice;
      recommendedRow = pricingMatrix.find(r => r.offerPrice === recommendedPrice);
    }
  } else if (pricingAggressiveness === "maximum") {
    // CEO wants maximum - highest price with acceptable book
    const sortedByPrice = [...pricingMatrix].sort((a, b) => b.offerPrice - a.offerPrice);
    for (const row of sortedByPrice) {
      if (row.oversubscription >= 3) {
        recommendedPrice = row.offerPrice;
        recommendedRow = row;
        break;
      }
    }
  } else if (pricingAggressiveness === "conservative") {
    // Conservative - strong book coverage required
    const sortedByPrice = [...pricingMatrix].sort((a, b) => b.offerPrice - a.offerPrice);
    for (const row of sortedByPrice) {
      if (row.oversubscription >= 10 && row.adjustedImpliedPop >= 0.10) {
        recommendedPrice = row.offerPrice;
        recommendedRow = row;
        break;
      }
    }
  } else {
    // Moderate - balanced approach
    const sortedByPrice = [...pricingMatrix].sort((a, b) => b.offerPrice - a.offerPrice);
    for (const row of sortedByPrice) {
      if (row.oversubscription >= 5 && row.adjustedImpliedPop >= 0) {
        recommendedPrice = row.offerPrice;
        recommendedRow = row;
        break;
      }
    }
  }
  
  if (!recommendedRow) {
    recommendedRow = pricingMatrix.find(r => r.offerPrice === recommendedPrice) || pricingMatrix[Math.floor(pricingMatrix.length / 2)];
    recommendedPrice = recommendedRow.offerPrice;
  }
  
  const recommendedRangeLow = Math.max(minAcceptablePrice || (recommendedPrice - 2), recommendedPrice - 2);
  const recommendedRangeHigh = recommendedPrice + 1;
  
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
  
  rationale.push(`Book coverage: ${recommendedRow.oversubscription}× oversubscribed`);
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
  
  // BUG FIX #4: Note CEO directive in rationale
  if (managementPriority === "runway_extension") {
    rationale.push(`CEO priority: "runway extension" - pricing for deal certainty`);
    if (ceoGuidance) {
      rationale.push(`CEO guidance: "${ceoGuidance}"`);
    }
  }
  
  rationale.push(`Founders retain ${(recommendedRow.founderOwnershipPost * 100).toFixed(1)}% post-IPO`);
  rationale.push(`Gross proceeds: $${Math.round(recommendedRow.grossProceedsM)}M`);
  
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
    fairValuePerShare,
    fairValueType,
    totalRaNPV = 0,
    peerMedianEVRevenue,
    peerMedianEVRaNPV = 0,
    ntmRevenue,
    sectorMedianFirstDayPop,
    sectorAverageFirstDayPop,
    historicalFirstDayPop,
    indicatedPriceRangeLow,
    indicatedPriceRangeHigh,
  } = assumptions;

  const isBiotech = sector === "biotech";
  const isPreRevenue = ntmRevenue === 0 || ntmRevenue < 1;
  const useRaNPVValuation = isBiotech || isPreRevenue;
  
  const companyNameUpper = companyName.toUpperCase();
  
  const recommendedRow = pricingMatrix.find(r => Math.abs(r.offerPrice - recommendedPrice) < 0.5);
  if (!recommendedRow) return "Error: Could not find recommended row";
  
  const popPercent = (recommendedRow.adjustedImpliedPop * 100).toFixed(0);
  const grossProceeds = Math.round(recommendedRow.grossProceedsM);
  const marketCapB = (recommendedRow.marketCapM / 1000).toFixed(1);
  const evB = (recommendedRow.enterpriseValueM / 1000).toFixed(1);
  
  // BUG FIX #2: Use correct fair value label
  const fairValueLabel = fairValueType === "ranpv" ? "raNPV" : "DCF";
  
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
  memo += `Gross Proceeds: $${grossProceeds}M\n`;
  memo += `Market Cap: ~$${marketCapB}B post-greenshoe\n`;
  memo += `Enterprise Value: ~$${evB}B\n\n`;
  
  // BUG FIX #1 & #2: Use correct valuation metric
  if (useRaNPVValuation && totalRaNPV > 0) {
    const evRaNPVMultiple = recommendedRow.evRaNPV.toFixed(2);
    const peerDiffPercent = (recommendedRow.vsPeerMedianRaNPV * 100).toFixed(0);
    memo += `Valuation Method: EV/raNPV (biotech/pre-revenue)\n`;
    memo += `EV/raNPV: ${evRaNPVMultiple}× (Peer Median: ${peerMedianEVRaNPV.toFixed(1)}×, ${parseInt(peerDiffPercent) >= 0 ? '+' : ''}${peerDiffPercent}%)\n`;
    memo += `Total raNPV: $${totalRaNPV.toFixed(0)}M\n`;
  } else {
    const evMultiple = recommendedRow.ntmEVRevenue.toFixed(1);
    const peerDiffPercent = (recommendedRow.vsPeerMedianRevenue * 100).toFixed(0);
    memo += `NTM EV/Revenue: ${evMultiple}× (Peer Median: ${peerMedianEVRevenue.toFixed(1)}×, ${parseInt(peerDiffPercent) >= 0 ? '+' : ''}${peerDiffPercent}%)\n`;
  }
  
  memo += `${fairValueLabel}/share: $${fairValuePerShare.toFixed(2)} (offer = ${(recommendedRow.fairValueSupport * 100).toFixed(0)}%)\n\n`;
  
  memo += `Pricing Matrix\n\n`;
  
  // Select rows around recommendation
  const recIndex = pricingMatrix.findIndex(r => Math.abs(r.offerPrice - recommendedPrice) < 0.5);
  const startIdx = Math.max(0, recIndex - 2);
  const endIdx = Math.min(pricingMatrix.length, startIdx + 6);
  const rows = pricingMatrix.slice(startIdx, endIdx);
  
  const pad = (s: string, n: number) => s.padStart(n);
  
  memo += "Offer Price            " + rows.map(r => pad(`$${r.offerPrice}`, 10)).join("") + "\n";
  memo += "Market Cap             " + rows.map(r => pad(`$${Math.round(r.marketCapM).toLocaleString()}`, 10)).join("") + "\n";
  memo += "Enterprise Value       " + rows.map(r => pad(`$${Math.round(r.enterpriseValueM).toLocaleString()}`, 10)).join("") + "\n";
  
  // BUG FIX #1: Show correct valuation metric
  if (useRaNPVValuation && totalRaNPV > 0) {
    memo += "EV/raNPV               " + rows.map(r => pad(`${r.evRaNPV.toFixed(2)}×`, 10)).join("") + "\n";
    memo += `vs peer median ${peerMedianEVRaNPV.toFixed(1)}×   ` + rows.map(r => {
      const pct = r.vsPeerMedianRaNPV * 100;
      return pad(`${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%`, 10);
    }).join("") + "\n";
  } else {
    memo += "NTM EV/Revenue         " + rows.map(r => {
      if (r.ntmEVRevenue === Infinity) return pad("N/A", 10);
      return pad(`${r.ntmEVRevenue.toFixed(1)}×`, 10);
    }).join("") + "\n";
    memo += `vs peer median ${peerMedianEVRevenue.toFixed(1)}×   ` + rows.map(r => {
      if (r.vsPeerMedianRevenue === Infinity) return pad("N/A", 10);
      const pct = r.vsPeerMedianRevenue * 100;
      return pad(`${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%`, 10);
    }).join("") + "\n";
  }
  
  // BUG FIX #2: Use correct label
  memo += `${fairValueLabel} $${fairValuePerShare.toFixed(2)} support     ` + rows.map(r => pad(`${(r.fairValueSupport * 100).toFixed(0)}%`, 10)).join("") + "\n";
  memo += "Gross proceeds         " + rows.map(r => pad(`$${Math.round(r.grossProceedsM)}`, 10)).join("") + "\n";
  
  // BUG FIX #7: Show order book tier boundaries with tier info
  memo += "Order Book Tier        " + rows.map(r => pad(r.orderBookTier || "N/A", 10)).join("") + "\n";
  memo += "Oversubscription       " + rows.map(r => pad(`${r.oversubscription.toFixed(1)}×`, 10)).join("") + "\n";
  
  // BUG FIX #3: Show all pop adjustments
  memo += `Day-1 Pop (${histPopLabel})\n`;
  memo += "  Base expected        " + rows.map(r => {
    const pct = r.baseImpliedPop * 100;
    return pad(`${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%`, 10);
  }).join("") + "\n";
  memo += "  Book adjustment      " + rows.map(r => {
    const pct = r.bookQualityAdjustment * 100;
    return pad(`${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%`, 10);
  }).join("") + "\n";
  memo += "  Valuation penalty    " + rows.map(r => {
    const pct = r.valuationPenalty * 100;
    return pad(`${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%`, 10);
  }).join("") + "\n";
  if (rows.some(r => r.secondaryDiscount > 0)) {
    memo += "  Secondary discount   " + rows.map(r => {
      const pct = -r.secondaryDiscount * 100;
      return pad(`${pct.toFixed(0)}%`, 10);
    }).join("") + "\n";
  }
  if (rows.some(r => r.catalystDiscount > 0)) {
    memo += "  Catalyst risk        " + rows.map(r => {
      const pct = -r.catalystDiscount * 100;
      return pad(`${pct.toFixed(0)}%`, 10);
    }).join("") + "\n";
  }
  memo += "  ADJUSTED POP         " + rows.map(r => {
    const pct = r.adjustedImpliedPop * 100;
    return pad(`${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%`, 10);
  }).join("") + "\n";
  
  memo += "Founder ownership      " + rows.map(r => pad(`${(r.founderOwnershipPost * 100).toFixed(1)}%`, 10)).join("") + "\n";
  
  memo += "\nRecommendation Rationale:\n";
  for (const r of rationale) {
    memo += `• ${r}\n`;
  }
  
  memo += `\nFile amendment at $${rangeLow.toFixed(0)}–$${rangeHigh.toFixed(0)} tonight, price at $${recommendedPrice.toFixed(0)} tomorrow morning.\n`;

  return memo;
}
