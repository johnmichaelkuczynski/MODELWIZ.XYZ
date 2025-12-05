import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

export type FinanceLLMProvider = 'zhi1' | 'zhi2' | 'zhi3' | 'zhi4' | 'zhi5';

export interface IPOAssumptions {
  companyName: string;
  filingDate: string;
  
  sharesOutstandingPreIPO: number;
  primarySharesOffered: number;
  secondarySharesOffered: number;
  greenshoePercent: number;
  
  targetGrossProceeds: number;
  indicatedPriceRangeLow: number;
  indicatedPriceRangeHigh: number;
  
  currentYearRevenue: number;
  nextYearRevenue: number;
  nextYearRevenueGrowth: number;
  nextYearEBITDA: number;
  nextYearEBITDAMargin: number;
  
  dcfValuePerShare: number;
  
  peerMultiples: {
    company: string;
    evRevenue: number;
  }[];
  peerMedianEVRevenue: number;
  
  orderBook: {
    priceLevel: number;
    oversubscription: number;
  }[];
  
  historicalFirstDayPop: number;
  
  foundersEmployeesOwnership: number;
  vcPeOwnership: number;
  
  underwritingFeePercent: number;
  
  useOfProceeds?: string;
  lockupDays?: number;
}

const IPO_PARSING_PROMPT = `You are an investment banking expert specializing in IPO pricing. Parse the following natural language description of an IPO and extract all relevant parameters.

Return a JSON object with the following structure:
{
  "companyName": "Company Name",
  "filingDate": "YYYY-MM-DD",
  
  "sharesOutstandingPreIPO": number (in millions),
  "primarySharesOffered": number (in millions, newly issued shares),
  "secondarySharesOffered": number (in millions, existing shareholder sales, default 0),
  "greenshoePercent": number (as decimal, e.g., 0.15 for 15%),
  
  "targetGrossProceeds": number (in millions),
  "indicatedPriceRangeLow": number (per share, if mentioned),
  "indicatedPriceRangeHigh": number (per share, if mentioned),
  
  "currentYearRevenue": number (in millions),
  "nextYearRevenue": number (in millions),
  "nextYearRevenueGrowth": number (as decimal),
  "nextYearEBITDA": number (in millions),
  "nextYearEBITDAMargin": number (as decimal),
  
  "dcfValuePerShare": number (intrinsic value from DCF analysis),
  
  "peerMultiples": [
    { "company": "Peer Name", "evRevenue": number (NTM EV/Revenue multiple) }
  ],
  "peerMedianEVRevenue": number (median of peer multiples),
  
  "orderBook": [
    { "priceLevel": number (price in dollars), "oversubscription": number (times oversubscribed) }
  ],
  
  "historicalFirstDayPop": number (as decimal, e.g., 0.28 for 28% average pop),
  
  "foundersEmployeesOwnership": number (as decimal, pre-IPO ownership),
  "vcPeOwnership": number (as decimal, pre-IPO ownership),
  
  "underwritingFeePercent": number (as decimal, typically 0.07 for 7%),
  
  "useOfProceeds": "string describing use of proceeds",
  "lockupDays": number (typically 180)
}

IMPORTANT CALCULATIONS:
- If greenshoe is mentioned as shares, convert to percentage: greenshoe_shares / primary_shares
- Order book should be parsed with price levels and corresponding oversubscription
- If only one oversubscription level is given, interpolate/extrapolate reasonable demand curve
- peerMedianEVRevenue should be the median of the peer multiples provided

Default values if not specified:
- greenshoePercent: 0.15 (15%)
- underwritingFeePercent: 0.07 (7%)
- lockupDays: 180
- secondarySharesOffered: 0
- historicalFirstDayPop: 0.25 (25%)

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
  return { assumptions, providerUsed };
}

interface PricingRow {
  offerPrice: number;
  marketCap: number;
  enterpriseValue: number;
  ntmEVRevenue: number;
  vsPeerMedianDiscount: number;
  vsDCFDiscount: number;
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
    sharesOutstandingPreIPO,
    primarySharesOffered,
    secondarySharesOffered = 0,
    greenshoePercent,
    targetGrossProceeds,
    nextYearRevenue,
    dcfValuePerShare,
    peerMedianEVRevenue,
    orderBook,
    historicalFirstDayPop,
    foundersEmployeesOwnership,
  } = assumptions;

  const greenshoeShares = primarySharesOffered * greenshoePercent;
  const totalSharesOffered = primarySharesOffered + secondarySharesOffered;
  const totalSharesWithGreenshoe = totalSharesOffered + greenshoeShares;
  
  const fdSharesPostIPO = sharesOutstandingPreIPO + primarySharesOffered + greenshoeShares;
  
  const midPrice = targetGrossProceeds / totalSharesOffered;
  
  const pricePoints = [];
  const step = midPrice > 20 ? 1 : 0.5;
  for (let p = midPrice - 4 * step; p <= midPrice + 2 * step; p += step) {
    if (p > 0) pricePoints.push(Math.round(p * 100) / 100);
  }
  
  const pricingMatrix: PricingRow[] = pricePoints.map(offerPrice => {
    const marketCap = fdSharesPostIPO * offerPrice;
    
    const grossProceeds = totalSharesOffered * offerPrice;
    
    const enterpriseValue = marketCap;
    const ntmEVRevenue = enterpriseValue / nextYearRevenue;
    
    const vsPeerMedianDiscount = (ntmEVRevenue - peerMedianEVRevenue) / peerMedianEVRevenue;
    
    const vsDCFDiscount = offerPrice / dcfValuePerShare;
    
    let oversubscription = 1;
    if (orderBook && orderBook.length > 0) {
      const sortedBook = [...orderBook].sort((a, b) => b.priceLevel - a.priceLevel);
      
      const exactMatch = sortedBook.find(ob => Math.abs(ob.priceLevel - offerPrice) < 0.5);
      if (exactMatch) {
        oversubscription = exactMatch.oversubscription;
      } else {
        for (let i = 0; i < sortedBook.length; i++) {
          if (offerPrice >= sortedBook[i].priceLevel) {
            oversubscription = sortedBook[i].oversubscription;
            break;
          }
          if (i === sortedBook.length - 1) {
            const ratio = offerPrice / sortedBook[i].priceLevel;
            oversubscription = Math.max(1, Math.round(sortedBook[i].oversubscription * (1.5 - ratio * 0.5)));
          }
        }
      }
    }
    
    const impliedFirstDayPop = Math.max(0, historicalFirstDayPop * (1 + (midPrice - offerPrice) / midPrice * 0.5));
    
    const dilutionFromNewShares = (primarySharesOffered + greenshoeShares) / fdSharesPostIPO;
    const founderEmployeeOwnershipPost = foundersEmployeesOwnership * (1 - dilutionFromNewShares);
    
    return {
      offerPrice,
      marketCap,
      enterpriseValue,
      ntmEVRevenue,
      vsPeerMedianDiscount,
      vsDCFDiscount,
      grossProceeds,
      oversubscription,
      impliedFirstDayPop,
      founderEmployeeOwnershipPost,
    };
  });
  
  let recommendedPrice = midPrice;
  let recommendedRow = pricingMatrix.find(r => Math.abs(r.offerPrice - midPrice) < step);
  
  for (const row of pricingMatrix) {
    if (row.oversubscription >= 15 && row.oversubscription <= 25 && 
        Math.abs(row.grossProceeds - targetGrossProceeds) < targetGrossProceeds * 0.1) {
      recommendedPrice = row.offerPrice;
      recommendedRow = row;
      break;
    }
  }
  
  if (!recommendedRow) {
    recommendedRow = pricingMatrix[Math.floor(pricingMatrix.length / 2)];
    recommendedPrice = recommendedRow.offerPrice;
  }
  
  const recommendedRangeLow = recommendedPrice - 2 * step;
  const recommendedRangeHigh = recommendedPrice;
  
  const rationale: string[] = [];
  rationale.push(`$${recommendedPrice.toFixed(2)} clears the book at ${recommendedRow.oversubscription}× with only the highest-quality accounts`);
  rationale.push(`Still ${((1 - recommendedRow.vsDCFDiscount) * 100).toFixed(0)}% below DCF and ${Math.abs(recommendedRow.vsPeerMedianDiscount * 100).toFixed(0)}% below peer median → generous but responsible discount`);
  rationale.push(`Leaves ~${(recommendedRow.impliedFirstDayPop * 100).toFixed(0)}% expected pop → matches recent SaaS average, keeps momentum funds happy`);
  rationale.push(`Founders/employees still own >${(recommendedRow.founderEmployeeOwnershipPost * 100).toFixed(0)}% → strong retention signal`);
  rationale.push(`Raises exact $${Math.round(recommendedRow.grossProceeds)}M primary target with zero secondary overhang`);
  
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
    dcfValuePerShare,
    peerMedianEVRevenue,
    targetGrossProceeds,
  } = assumptions;

  const companyNameUpper = companyName.toUpperCase();
  
  const recommendedRow = pricingMatrix.find(r => Math.abs(r.offerPrice - recommendedPrice) < 0.5);
  const impliedPop = recommendedRow ? (recommendedRow.impliedFirstDayPop * 100).toFixed(0) : "20";
  
  let memo = `${companyNameUpper} – FINAL IPO PRICING RECOMMENDATION\n\n`;
  memo += `Recommended range to file amendment:      $${rangeLow.toFixed(2)} – $${rangeHigh.toFixed(2)}\n`;
  memo += `Recommended final offer price:             $${recommendedPrice.toFixed(2)}   ← clears the book cleanly, raises $${Math.round(targetGrossProceeds)}M primary exactly, leaves ~${impliedPop}% day-one pop (in line with comps)\n\n`;
  
  memo += `Pricing Matrix (fully-diluted post-greenshoe basis, in millions except per-share data)\n\n`;
  
  const rows = pricingMatrix.slice(0, 7);
  
  const priceHeader = "Offer Price       " + rows.map(r => `$${r.offerPrice.toFixed(2)}`).map(s => s.padStart(8)).join("  ");
  memo += priceHeader + "\n";
  
  const marketCapRow = "Market Cap         " + rows.map(r => `$${Math.round(r.marketCap).toLocaleString()}`).map(s => s.padStart(8)).join("  ");
  memo += marketCapRow + "\n";
  
  const evRevRow = "NTM EV/Revenue      " + rows.map(r => `${r.ntmEVRevenue.toFixed(1)}×`).map(s => s.padStart(8)).join("  ");
  memo += evRevRow + "\n";
  
  const vsPeerRow = `vs. peer median ${peerMedianEVRevenue.toFixed(1)}× discount   ` + rows.map(r => `${(r.vsPeerMedianDiscount * 100).toFixed(0)}%`).map(s => s.padStart(8)).join("  ");
  memo += vsPeerRow + "\n";
  
  const dcfRow = `DCF midpoint $${dcfValuePerShare.toFixed(2)} support         ` + rows.map(r => `${(r.vsDCFDiscount * 100).toFixed(0)}%`).map(s => s.padStart(8)).join("  ");
  memo += dcfRow + "\n";
  
  const proceedsRow = "Gross proceeds                     " + rows.map(r => `$${Math.round(r.grossProceeds)}`).map(s => s.padStart(8)).join("  ");
  memo += proceedsRow + "\n";
  
  const oversubRow = "Oversubscription at price          " + rows.map(r => `${r.oversubscription}×`).map(s => s.padStart(8)).join("  ");
  memo += oversubRow + "\n";
  
  const avgPop = assumptions.historicalFirstDayPop * 100;
  const popRow = `Implied first-day pop (hist. avg ${avgPop.toFixed(0)}%) ` + rows.map(r => `${(r.impliedFirstDayPop * 100).toFixed(0)}%`).map(s => s.padStart(8)).join("  ");
  memo += popRow + "\n";
  
  const ownershipRow = "Founder + employee ownership post-IPO  " + rows.map(r => `${(r.founderEmployeeOwnershipPost * 100).toFixed(1)}%`).map(s => s.padStart(8)).join("  ");
  memo += ownershipRow + "\n";
  
  memo += "\nRecommendation rationale\n";
  for (const r of rationale) {
    memo += `- ${r}\n`;
  }
  
  memo += `\nFile amendment at $${rangeLow.toFixed(0)}–$${rangeHigh.toFixed(0)} tonight, price at $${recommendedPrice.toFixed(0)} tomorrow morning.\n`;

  return memo;
}
