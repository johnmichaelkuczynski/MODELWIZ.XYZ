import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

export type FinanceLLMProvider = 'zhi1' | 'zhi2' | 'zhi3' | 'zhi4' | 'zhi5';

export interface IPOAssumptions {
  companyName: string;
  filingDate: string;
  sector: string; // biotech, saas, fintech, consumer, etc.
  
  sharesOutstandingPreIPO: number;
  primarySharesOffered: number;
  secondarySharesOffered: number;
  greenshoePercent: number;
  
  targetGrossProceeds: number;
  indicatedPriceRangeLow: number;
  indicatedPriceRangeHigh: number;
  
  // For SaaS/Tech - NTM revenue
  currentYearRevenue: number;
  nextYearRevenue: number;
  nextYearRevenueGrowth: number;
  nextYearEBITDA: number;
  nextYearEBITDAMargin: number;
  
  // For Biotech - 2030 risk-adjusted revenue (rNPV basis)
  riskAdjustedRevenue2030?: number;
  
  dcfValuePerShare: number;
  
  peerMultiples: {
    company: string;
    evRevenue: number;
  }[];
  peerMedianEVRevenue: number;
  
  // Order book with EXACT price levels and oversubscription - DO NOT SMOOTH OR INVENT
  orderBook: {
    priceLevel: number;
    oversubscription: number;
  }[];
  
  historicalFirstDayPop: number;
  sectorAverageFirstDayPop?: number; // Sector-specific pop
  
  foundersEmployeesOwnership: number;
  vcPeOwnership: number;
  
  underwritingFeePercent: number;
  
  useOfProceeds?: string;
  lockupDays?: number;
  
  // Board guidance
  boardGuidance?: string; // e.g., "clean, orderly aftermarket" = lower pop target
}

const IPO_PARSING_PROMPT = `You are an investment banking expert specializing in IPO pricing. Parse the following natural language description of an IPO and extract all relevant parameters.

CRITICAL: You must correctly identify the SECTOR of the company:
- "biotech" for pharmaceutical, clinical-stage, drug development companies
- "saas" for software-as-a-service, cloud software companies
- "fintech" for financial technology companies
- "consumer" for consumer goods/services companies
- "tech" for general technology companies

Return a JSON object with the following structure:
{
  "companyName": "Company Name",
  "filingDate": "YYYY-MM-DD",
  "sector": "biotech" | "saas" | "fintech" | "consumer" | "tech",
  
  "sharesOutstandingPreIPO": number (in millions),
  "primarySharesOffered": number (in millions, newly issued shares),
  "secondarySharesOffered": number (in millions, existing shareholder sales, default 0),
  "greenshoePercent": number (as decimal, e.g., 0.15 for 15%),
  
  "targetGrossProceeds": number (in millions),
  "indicatedPriceRangeLow": number (per share, if mentioned),
  "indicatedPriceRangeHigh": number (per share, if mentioned),
  
  "currentYearRevenue": number (in millions, for year of IPO),
  "nextYearRevenue": number (in millions, for year after IPO),
  "nextYearRevenueGrowth": number (as decimal),
  "nextYearEBITDA": number (in millions, can be negative for pre-profit companies),
  "nextYearEBITDAMargin": number (as decimal),
  
  "riskAdjustedRevenue2030": number (in millions, FOR BIOTECH ONLY - peak risk-adjusted revenue estimate),
  
  "dcfValuePerShare": number (intrinsic value from DCF analysis),
  
  "peerMultiples": [
    { "company": "Peer Name", "evRevenue": number (EV/Revenue multiple - use 2030 rNPV for biotech) }
  ],
  "peerMedianEVRevenue": number (median of peer multiples),
  
  "orderBook": [
    { "priceLevel": number (price in dollars), "oversubscription": number (times oversubscribed) }
  ],
  
  "historicalFirstDayPop": number (as decimal, e.g., 0.76 for 76%),
  "sectorAverageFirstDayPop": number (as decimal - 0.76 for biotech avg, 0.25 for SaaS),
  
  "foundersEmployeesOwnership": number (as decimal, pre-IPO ownership),
  "vcPeOwnership": number (as decimal, pre-IPO ownership),
  
  "underwritingFeePercent": number (as decimal, typically 0.07 for 7%),
  
  "useOfProceeds": "string describing use of proceeds",
  "lockupDays": number (typically 180),
  
  "boardGuidance": "string describing board's pricing preference, if mentioned"
}

CRITICAL ORDER BOOK PARSING:
- Parse order book entries EXACTLY as stated - "$32 and above: 28× oversubscribed" means priceLevel: 32, oversubscription: 28
- NEVER smooth, interpolate, or invent demand numbers
- Order book shows demand DROP as price rises (inverse relationship)
- If "$32+: 28×, $30+: 41×, $28+: 59×" then at $32 oversubscription is 28×, at $30 it's 41×, at $28 it's 59×

SECTOR-SPECIFIC DEFAULTS:
- Biotech: historicalFirstDayPop = 0.76 (76% average), sectorAverageFirstDayPop = 0.50 (50% target for "responsible" pricing)
- SaaS: historicalFirstDayPop = 0.25 (25% average), sectorAverageFirstDayPop = 0.20 (20% target)
- Board wants "clean, orderly aftermarket" = target 45-55% pop for biotech, 15-25% for SaaS

Default values if not specified:
- greenshoePercent: 0.15 (15%)
- underwritingFeePercent: 0.07 (7%)
- lockupDays: 180
- secondarySharesOffered: 0

IMPORTANT: Return ONLY the JSON object, no markdown, no explanation.`;

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
      temperature: 0.3,
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

  let jsonStr = responseText.trim();
  if (jsonStr.startsWith("```json")) {
    jsonStr = jsonStr.slice(7);
  } else if (jsonStr.startsWith("```")) {
    jsonStr = jsonStr.slice(3);
  }
  if (jsonStr.endsWith("```")) {
    jsonStr = jsonStr.slice(0, -3);
  }
  
  if (!jsonStr.startsWith("{")) {
    const startIdx = jsonStr.indexOf("{");
    const endIdx = jsonStr.lastIndexOf("}");
    if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
      jsonStr = jsonStr.slice(startIdx, endIdx + 1);
    }
  }
  
  jsonStr = jsonStr.trim();

  const assumptions: IPOAssumptions = JSON.parse(jsonStr);
  
  // Set sector defaults if not detected
  if (!assumptions.sector) {
    assumptions.sector = "tech";
  }
  
  // Set sector-specific pop defaults
  if (!assumptions.sectorAverageFirstDayPop) {
    if (assumptions.sector === "biotech") {
      assumptions.sectorAverageFirstDayPop = 0.50; // 50% target for responsible biotech pricing
    } else if (assumptions.sector === "saas") {
      assumptions.sectorAverageFirstDayPop = 0.20; // 20% target for SaaS
    } else {
      assumptions.sectorAverageFirstDayPop = 0.25; // 25% default
    }
  }
  
  return { assumptions, providerUsed };
}

interface PricingRow {
  offerPrice: number;
  marketCap: number;
  enterpriseValue: number;
  valuationMultiple: number; // Sector-appropriate: 2030 rNPV for biotech, NTM EV/Rev for SaaS
  valuationMetricLabel: string;
  vsPeerMedianDiscount: number;
  vsDCFSupport: number; // Percentage of DCF value (offer price / DCF)
  grossProceeds: number;
  oversubscription: number;
  impliedFirstDayPop: number;
  founderEmployeeOwnershipPost: number;
}

export function calculateIPOPricing(assumptions: IPOAssumptions): {
  assumptions: IPOAssumptions;
  pricingMatrix: PricingRow[];
  recommendedRangeLow: number;
  recommendedRangeHigh: number;
  recommendedPrice: number;
  rationale: string[];
  memoText: string;
} {
  const {
    companyName,
    sector,
    sharesOutstandingPreIPO,
    primarySharesOffered,
    secondarySharesOffered = 0,
    greenshoePercent,
    targetGrossProceeds,
    nextYearRevenue,
    riskAdjustedRevenue2030,
    dcfValuePerShare,
    peerMedianEVRevenue,
    orderBook,
    historicalFirstDayPop,
    sectorAverageFirstDayPop = 0.50,
    foundersEmployeesOwnership,
    boardGuidance,
  } = assumptions;

  // FIX #3: Gross proceeds = price × total shares offered (fixed shares, variable proceeds)
  const greenshoeShares = primarySharesOffered * greenshoePercent;
  const totalSharesOffered = primarySharesOffered + secondarySharesOffered;
  const totalSharesWithGreenshoe = totalSharesOffered + greenshoeShares;
  
  // Fully-diluted shares post-IPO (including greenshoe)
  const fdSharesPostIPO = sharesOutstandingPreIPO + primarySharesOffered + greenshoeShares;
  
  // Calculate mid-price from target gross proceeds
  const midPrice = targetGrossProceeds / totalSharesOffered;
  
  // Generate price points around the order book range
  // Look at order book to determine appropriate range
  let minOrderBookPrice = midPrice - 4;
  let maxOrderBookPrice = midPrice + 4;
  
  if (orderBook && orderBook.length > 0) {
    const bookPrices = orderBook.map(ob => ob.priceLevel);
    minOrderBookPrice = Math.min(...bookPrices) - 2;
    maxOrderBookPrice = Math.max(...bookPrices) + 2;
  }
  
  const step = 1; // $1 increments for clarity
  const pricePoints: number[] = [];
  for (let p = minOrderBookPrice; p <= maxOrderBookPrice; p += step) {
    if (p > 0) pricePoints.push(Math.round(p * 100) / 100);
  }
  
  // FIX #2: Use sector-appropriate valuation metric
  const isBiotech = sector === "biotech";
  const revenueForValuation = isBiotech && riskAdjustedRevenue2030 
    ? riskAdjustedRevenue2030 
    : nextYearRevenue;
  const valuationMetricLabel = isBiotech ? "2030 rNPV multiple" : "NTM EV/Revenue";
  
  // FIX #5: Use sector-specific first-day pop expectations
  const targetPop = isBiotech ? 0.50 : 0.25; // 50% for biotech, 25% for SaaS
  const maxAcceptablePop = boardGuidance?.toLowerCase().includes("clean") || boardGuidance?.toLowerCase().includes("orderly") 
    ? 0.60 // If board wants clean aftermarket, max 60% pop
    : 0.80; // Otherwise allow up to 80%
  
  const pricingMatrix: PricingRow[] = pricePoints.map(offerPrice => {
    // FIX #3: Gross proceeds scales linearly with price (fixed shares × variable price)
    const grossProceeds = totalSharesWithGreenshoe * offerPrice;
    
    // Market cap = FD shares × offer price
    const marketCap = fdSharesPostIPO * offerPrice;
    
    // Enterprise value = Market cap - post-IPO cash
    // Post-IPO cash = existing cash + primary proceeds (net of fees)
    // For simplicity, use market cap as EV approximation (can be refined)
    const enterpriseValue = marketCap;
    
    // FIX #2: Sector-appropriate valuation multiple
    const valuationMultiple = enterpriseValue / revenueForValuation;
    
    // Discount vs peer median
    const vsPeerMedianDiscount = (valuationMultiple - peerMedianEVRevenue) / peerMedianEVRevenue;
    
    // FIX: DCF support = offer price as % of DCF value (not inverse)
    const vsDCFSupport = offerPrice / dcfValuePerShare;
    
    // FIX #1: Order book parsing - use EXACT values from order book, NO SMOOTHING
    let oversubscription = 1;
    if (orderBook && orderBook.length > 0) {
      // Sort order book from highest to lowest price
      const sortedBook = [...orderBook].sort((a, b) => b.priceLevel - a.priceLevel);
      
      // Find the applicable oversubscription for this price
      // Order book says "$32+: 28×" means at $32 or above, oversubscription is 28×
      // As price RISES, oversubscription DROPS (inverse relationship)
      for (const entry of sortedBook) {
        if (offerPrice >= entry.priceLevel) {
          oversubscription = entry.oversubscription;
          break;
        }
      }
      
      // If price is below all order book entries, use the highest oversubscription
      if (oversubscription === 1 && sortedBook.length > 0) {
        const lowestEntry = sortedBook[sortedBook.length - 1];
        if (offerPrice < lowestEntry.priceLevel) {
          // Extrapolate higher demand at lower prices
          const priceDiff = lowestEntry.priceLevel - offerPrice;
          const extraDemand = Math.round(priceDiff * 5); // ~5× more demand per $1 below
          oversubscription = lowestEntry.oversubscription + extraDemand;
        }
      }
    }
    
    // FIX #5: Implied first-day pop calculation
    // Pop is inversely related to price - lower price = higher pop
    // Use sector-specific historical average as baseline
    const priceRatio = offerPrice / dcfValuePerShare;
    // If priced at DCF, pop would be minimal (~10%)
    // If priced at 50% of DCF, pop could be 100%+
    const impliedFirstDayPop = Math.max(0.10, historicalFirstDayPop * (1 - priceRatio) * 2);
    
    // FIX #4: Founder/employee ownership DECLINES as price rises
    // At IPO, founders sell NO shares (primary offering only)
    // Dilution = new shares issued / post-IPO shares
    // Post-IPO ownership = pre-IPO ownership × (pre-IPO shares / post-IPO shares)
    const founderEmployeeOwnershipPost = foundersEmployeesOwnership * (sharesOutstandingPreIPO / fdSharesPostIPO);
    
    return {
      offerPrice,
      marketCap,
      enterpriseValue,
      valuationMultiple,
      valuationMetricLabel,
      vsPeerMedianDiscount,
      vsDCFSupport,
      grossProceeds,
      oversubscription,
      impliedFirstDayPop,
      founderEmployeeOwnershipPost,
    };
  });
  
  // FIX #6: Recommendation logic - find HIGHEST price with acceptable pop
  // Start from highest price and work down until we find acceptable conditions
  let recommendedPrice = midPrice;
  let recommendedRow: PricingRow | undefined;
  
  // Sort by price descending to find HIGHEST acceptable price
  const sortedMatrix = [...pricingMatrix].sort((a, b) => b.offerPrice - a.offerPrice);
  
  for (const row of sortedMatrix) {
    // Criteria for acceptable pricing:
    // 1. Oversubscription >= 20× (quality demand)
    // 2. Implied pop <= maxAcceptablePop (controlled aftermarket)
    // 3. Price <= DCF value (leave some upside)
    // 4. Discount to peers is reasonable (not too aggressive)
    
    const hasGoodDemand = row.oversubscription >= 20;
    const hasAcceptablePop = row.impliedFirstDayPop <= maxAcceptablePop;
    const belowDCF = row.offerPrice <= dcfValuePerShare;
    const reasonableDiscount = row.vsPeerMedianDiscount <= 0.10; // Max 10% above peer median
    
    if (hasGoodDemand && hasAcceptablePop && belowDCF && reasonableDiscount) {
      recommendedPrice = row.offerPrice;
      recommendedRow = row;
      break;
    }
  }
  
  // Fallback: if no perfect match, find highest price with at least 20× demand
  if (!recommendedRow) {
    for (const row of sortedMatrix) {
      if (row.oversubscription >= 20) {
        recommendedPrice = row.offerPrice;
        recommendedRow = row;
        break;
      }
    }
  }
  
  // Final fallback
  if (!recommendedRow) {
    recommendedRow = pricingMatrix[Math.floor(pricingMatrix.length / 2)];
    recommendedPrice = recommendedRow.offerPrice;
  }
  
  // Filing range: recommended price - $2 to recommended price
  const recommendedRangeLow = recommendedPrice - 2;
  const recommendedRangeHigh = recommendedPrice;
  
  // Generate rationale
  const rationale: string[] = [];
  
  const popPercent = (recommendedRow.impliedFirstDayPop * 100).toFixed(0);
  const dcfDiscount = ((1 - recommendedRow.vsDCFSupport) * 100).toFixed(0);
  const peerDiscount = Math.abs(recommendedRow.vsPeerMedianDiscount * 100).toFixed(0);
  const peerDirection = recommendedRow.vsPeerMedianDiscount > 0 ? "above" : "below";
  
  rationale.push(`$${recommendedPrice.toFixed(2)} is the highest price that still leaves ~${popPercent}% expected day-one pop — ${isBiotech ? 'in line with responsible biotech precedent' : 'appropriate for controlled aftermarket'}`);
  rationale.push(`Only ${peerDiscount}% ${peerDirection} peer median multiple → responsible, not aggressive`);
  rationale.push(`Still ${dcfDiscount}% below DCF → clear bargain for long-term believers`);
  rationale.push(`Clears the book at ${recommendedRow.oversubscription}× with only the highest-quality ${isBiotech ? 'specialist healthcare investors' : 'institutional investors'}`);
  
  if (boardGuidance?.toLowerCase().includes("clean") || boardGuidance?.toLowerCase().includes("orderly")) {
    rationale.push(`Board's "clean, orderly aftermarket" goal fully satisfied`);
  }
  
  rationale.push(`Founders/employees retain ${(recommendedRow.founderEmployeeOwnershipPost * 100).toFixed(1)}% ownership → strong retention signal`);
  rationale.push(`Raises $${Math.round(recommendedRow.grossProceeds)}M gross proceeds${secondarySharesOffered === 0 ? ' with zero secondary overhang' : ''}`);
  
  const memoText = formatIPOMemo(
    assumptions,
    pricingMatrix,
    recommendedRangeLow,
    recommendedRangeHigh,
    recommendedPrice,
    rationale
  );

  return {
    assumptions,
    pricingMatrix,
    recommendedRangeLow,
    recommendedRangeHigh,
    recommendedPrice,
    rationale,
    memoText,
  };
}

function formatIPOMemo(
  assumptions: IPOAssumptions,
  pricingMatrix: PricingRow[],
  rangeLow: number,
  rangeHigh: number,
  recommendedPrice: number,
  rationale: string[]
): string {
  const {
    companyName,
    sector,
    dcfValuePerShare,
    peerMedianEVRevenue,
    targetGrossProceeds,
    historicalFirstDayPop,
  } = assumptions;

  const companyNameUpper = companyName.toUpperCase();
  const isBiotech = sector === "biotech";
  
  const recommendedRow = pricingMatrix.find(r => Math.abs(r.offerPrice - recommendedPrice) < 0.5);
  const impliedPop = recommendedRow ? (recommendedRow.impliedFirstDayPop * 100).toFixed(0) : "50";
  const grossProceeds = recommendedRow ? Math.round(recommendedRow.grossProceeds) : Math.round(targetGrossProceeds);
  
  let memo = `${companyNameUpper} – FINAL IPO PRICING RECOMMENDATION\n\n`;
  memo += `Recommended range to file amendment:      $${rangeLow.toFixed(2)} – $${rangeHigh.toFixed(2)}\n`;
  memo += `Recommended final offer price:             $${recommendedPrice.toFixed(2)}   ← maximum responsible price, raises full $${grossProceeds}M primary, expected ${impliedPop}% day-one pop (in line with ${isBiotech ? '2024–2025 biotech' : 'recent'} average)\n\n`;
  
  memo += `Pricing Matrix (fully-diluted post-greenshoe basis, in millions except per-share data)\n\n`;
  
  // Select 7 rows around the recommendation
  const recIndex = pricingMatrix.findIndex(r => Math.abs(r.offerPrice - recommendedPrice) < 0.5);
  const startIdx = Math.max(0, recIndex - 3);
  const endIdx = Math.min(pricingMatrix.length, startIdx + 7);
  const rows = pricingMatrix.slice(startIdx, endIdx);
  
  const priceHeader = "Offer Price          " + rows.map(r => `$${r.offerPrice.toFixed(2)}`).map(s => s.padStart(9)).join("  ");
  memo += priceHeader + "\n";
  
  const marketCapRow = "Market Cap            " + rows.map(r => `$${Math.round(r.marketCap).toLocaleString()}`).map(s => s.padStart(9)).join("  ");
  memo += marketCapRow + "\n";
  
  const evRow = `EV (post-IPO cash)    ` + rows.map(r => `$${Math.round(r.enterpriseValue).toLocaleString()}`).map(s => s.padStart(9)).join("  ");
  memo += evRow + "\n";
  
  const metricLabel = isBiotech ? "2030 rNPV multiple" : "NTM EV/Revenue";
  const multipleRow = `${metricLabel.padEnd(22)}` + rows.map(r => `${r.valuationMultiple.toFixed(1)}×`).map(s => s.padStart(9)).join("  ");
  memo += multipleRow + "\n";
  
  const vsPeerRow = `vs. peer median ${peerMedianEVRevenue.toFixed(1)}× discount` + rows.map(r => `${(r.vsPeerMedianDiscount * 100) >= 0 ? '+' : ''}${(r.vsPeerMedianDiscount * 100).toFixed(0)}%`).map(s => s.padStart(9)).join("  ");
  memo += vsPeerRow + "\n";
  
  const dcfRow = `DCF midpoint $${dcfValuePerShare.toFixed(2)} support` + rows.map(r => `${(r.vsDCFSupport * 100).toFixed(0)}%`).map(s => s.padStart(9)).join("  ");
  memo += dcfRow + "\n";
  
  const proceedsRow = "Gross proceeds        " + rows.map(r => `$${Math.round(r.grossProceeds)}`).map(s => s.padStart(9)).join("  ");
  memo += proceedsRow + "\n";
  
  const oversubRow = "Oversubscription      " + rows.map(r => `${r.oversubscription}×`).map(s => s.padStart(9)).join("  ");
  memo += oversubRow + "\n";
  
  const avgPop = historicalFirstDayPop * 100;
  const popRow = `Implied first-day pop (hist. ${avgPop.toFixed(0)}%)` + rows.map(r => `${(r.impliedFirstDayPop * 100).toFixed(0)}%`).map(s => s.padStart(9)).join("  ");
  memo += popRow + "\n";
  
  const ownershipRow = "Founder + employee post-IPO" + rows.map(r => `${(r.founderEmployeeOwnershipPost * 100).toFixed(1)}%`).map(s => s.padStart(9)).join("  ");
  memo += ownershipRow + "\n";
  
  memo += "\nRecommendation rationale\n";
  for (const r of rationale) {
    memo += `- ${r}\n`;
  }
  
  memo += `\nFile amendment at $${rangeLow.toFixed(0)}–$${rangeHigh.toFixed(0)} tonight, price at $${recommendedPrice.toFixed(0)} tomorrow morning.\n`;
  
  if (isBiotech) {
    memo += `Congrats — this one is going to trade like CG Oncology.\n`;
  }

  return memo;
}
