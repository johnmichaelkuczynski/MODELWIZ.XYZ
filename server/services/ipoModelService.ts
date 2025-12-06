import ExcelJS from 'exceljs';
import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';
import { runInstrumentEngine, InstrumentEngineResult } from './ipoInstrumentEngine';

export type FinanceLLMProvider = 'zhi1' | 'zhi2' | 'zhi3' | 'zhi4' | 'zhi5';

// ============ ENHANCED MULTI-INSTRUMENT TYPES ============

// Convertible Instrument Types
export type ConvertibleTriggerType = 
  | 'price_gt'       // Converts if IPO price > threshold
  | 'price_gte'      // Converts if IPO price >= threshold
  | 'lower_of'       // Converts at lower of two prices
  | 'at_ipo_price'   // Converts at IPO price
  | 'fixed_shares'   // Fixed share conversion
  | 'conditional';   // Complex conditional (evaluated by probability)

export interface ConvertibleInstrument {
  name: string;                          // e.g., "Series D SAFEs", "Venture Debt"
  type: 'safe' | 'debt' | 'loan' | 'note' | 'other';
  amountMillions: number;                // Principal amount in millions
  triggerType: ConvertibleTriggerType;
  triggerPrice?: number;                 // Price threshold for conversion
  triggerPrice2?: number;                // Second price for lower_of (e.g., 80% of IPO)
  triggerMultiplier?: number;            // Multiplier for IPO price (e.g., 0.8 for 80%)
  triggerCondition?: string;             // Condition string (e.g., "IPO>25")
  fixedShares?: number;                  // Fixed shares if type is fixed_shares (in millions)
  probability?: number;                  // Probability of conversion (0-1, default 1.0)
  interestRate?: number;                 // Interest rate for debt instruments
}

// Contingent Liability Types
export type ContingencyType = 
  | 'earnout'        // Share issuance on performance milestone
  | 'warrant'        // Performance warrants with strike
  | 'grant'          // Share grant (e.g., partner agreement)
  | 'litigation'     // Cash payment contingency
  | 'royalty'        // Ongoing payment obligation
  | 'milestone';     // General milestone-based obligation

export interface ContingentLiability {
  name: string;                          // e.g., "Acquisition Earnout", "FDA Milestone"
  type: ContingencyType;
  sharesMillions?: number;               // Shares to be issued (if share-based)
  strikePrice?: number;                  // Strike price for warrants
  paymentMillions?: number;              // Cash payment (if payment-based)
  condition?: string;                    // Condition description
  probability: number;                   // Probability of occurring (0-1)
}

// Strategic Deal Types
export interface StrategicDeal {
  partnerName: string;                   // e.g., "Google", "Pfizer"
  sharesMillions: number;                // Shares allocated
  priceType: 'ipo_price' | 'ipo_premium' | 'fixed' | 'discounted';
  pricePremium?: number;                 // Premium over IPO (e.g., 0.05 for +5%)
  priceDiscount?: number;                // Discount from IPO (e.g., 0.10 for -10%)
  fixedPrice?: number;                   // Fixed price per share
  probability?: number;                  // Probability of deal closing (default 1.0)
  isAnchorOrder?: boolean;               // True if this is an anchor order
}

// Anchor Order (demand boost)
export interface AnchorOrder {
  investorName: string;                  // e.g., "Sovereign Wealth Fund"
  amountMillions: number;                // Committed investment amount
  priceType: 'ipo_price' | 'ipo_premium' | 'ipo_discount';
  pricePremium?: number;
  priceDiscount?: number;
}

// Employee Option Pool
export interface EmployeeOptionPool {
  sharesMillions: number;                // Total unexercised options
  avgStrikePrice: number;                // Weighted average strike price
  vestedPercent?: number;                // Percent already vested (default 100%)
}

// Blended Valuation Component
export interface ValuationMultiple {
  name: string;                          // e.g., "Industry Revenue", "AI Proxy"
  type: 'revenue' | 'ebitda';
  multiple: number;                      // The multiple (e.g., 28.0x)
  weight: number;                        // Weight in blend (0-1, all weights should sum to 1)
}

export interface IPOAssumptions {
  companyName: string;
  transactionDate: string;
  
  // Required Inputs
  ltmRevenue: number;           // LTM Revenue in millions
  ltmEbitda?: number;           // LTM EBITDA in millions (optional for EBITDA multiple)
  industryRevenueMultiple: number;  // e.g., 10.0x
  industryEbitdaMultiple?: number;  // Optional EBITDA multiple
  preIpoShares: number;         // Pre-IPO Fully Diluted Shares in millions
  primaryRaiseTarget: number;   // Primary Cash to Raise in millions
  ipoDiscount: number;          // Target Investor Discount as decimal (e.g., 0.20 for 20%)
  
  // Optional Inputs
  secondaryShares?: number;     // Secondary shares sold by existing holders (in millions)
  greenshoePercent?: number;    // Over-allotment option as decimal (typically 0.15 = 15%)
  underwritingFeePercent?: number;  // Underwriting commission as decimal (typically 0.07 = 7%)
  
  // Convertible Debt (Optional)
  convertibleDebtAmount?: number;     // Convertible debt amount in millions
  conversionTriggerPrice?: number;    // Price per share that triggers conversion
  conversionShares?: number;          // Shares debt converts into (in millions)
  
  // Valuation Method
  valuationMethod: 'revenue' | 'ebitda' | 'blended';
  blendWeight?: number;         // Weight for revenue multiple in blended (0-1)
  
  // Dual-Class Share Structure (Optional)
  founderSharesMillions?: number;     // Number of founder shares in millions
  founderVoteMultiplier?: number;     // Votes per founder share (e.g., 10 for 10x voting)
  controlThreshold?: number;          // Minimum voting % founders require (e.g., 0.40 for 40%)
  
  // Milestone Warrants (Optional - Contingent Dilution)
  warrantSharesMillions?: number;     // Shares to be issued if milestone hit (in millions)
  warrantStrikePrice?: number;        // Price at which those shares can be bought
  milestoneProbability?: number;      // Estimated chance of milestone being hit (0.0 to 1.0)
  
  // Strategic Partner Block Allocation (Optional) - Legacy single partner
  strategicPartnerSharesMillions?: number;  // Shares guaranteed to strategic partner (in millions)
  strategicPartnerName?: string;            // Name of strategic partner
  
  // ============ ENHANCED MULTI-INSTRUMENT ARRAYS ============
  // These override the single-instrument fields above when present
  
  // Multiple Convertible Instruments
  convertibles?: ConvertibleInstrument[];
  
  // Contingent Liabilities (earnouts, warrants, litigation, etc.)
  contingencies?: ContingentLiability[];
  
  // Strategic Deals (partners with premiums/discounts)
  strategicDeals?: StrategicDeal[];
  
  // Anchor Orders (demand boost)
  anchorOrders?: AnchorOrder[];
  
  // Employee Option Pool (treasury stock method dilution)
  employeeOptions?: EmployeeOptionPool;
  
  // Multi-proxy Blended Valuation
  valuationMultiples?: ValuationMultiple[];
  
  // Growth Premium (for high-growth companies)
  revenueGrowthRate?: number;    // Revenue growth rate as decimal (e.g., 2.0 for 200%)
  growthPremiumThreshold?: number; // Growth threshold to trigger premium (e.g., 2.0 for 200%)
  growthPremium?: number;         // Growth premium multiplier as decimal (e.g., 0.15 for 15%)
}

export interface IPOPricingResult {
  companyName: string;
  transactionDate: string;
  
  // Valuation Metrics
  preMoneyValuation: number;    // Pre-Money Valuation in millions
  theoreticalPrice: number;     // Undiscounted price per share
  offerPrice: number;           // Final discounted offer price per share
  postMoneyValuation: number;   // Post-Money Valuation in millions
  impliedPreMoneyAtOffer: number;  // Implied Pre-Money at offer price
  
  // Offering Structure
  newSharesIssued: number;      // Primary shares in millions
  secondarySharesSold: number;  // Secondary shares in millions
  totalSharesOffered: number;   // Total shares in offering in millions
  greenshoeShares: number;      // Greenshoe shares in millions
  
  // Proceeds
  grossPrimaryProceeds: number;   // Gross primary proceeds in millions
  netPrimaryProceeds: number;     // Net to company after fees in millions
  secondaryProceeds: number;      // To selling shareholders in millions
  totalGrossProceeds: number;     // Total gross proceeds in millions
  underwritingFees: number;       // Total underwriting fees in millions
  
  // Ownership & Dilution
  postIpoSharesOutstanding: number;  // Total shares post-IPO in millions
  percentageSold: number;            // Percentage of company sold in IPO
  existingHoldersDilution: number;   // Dilution to existing shareholders
  
  // Trading Metrics
  expectedFirstDayPop: number;   // Expected first day gain (discount inverse)
  marketCapAtOffer: number;      // Market cap at offer price in millions
  
  // Convertible Debt Treatment
  convertibleDebtTreatment?: {
    triggerPrice: number;
    debtAmount: number;
    conversionShares: number;
    conversionActivated: boolean;
    originalPreIpoShares: number;
    adjustedPreIpoShares: number;
    tentativeOfferPrice: number;
  };
  
  // Dual-Class Share Voting Control Analysis
  votingControlAnalysis?: {
    founderSharesMillions: number;
    founderVoteMultiplier: number;
    controlThreshold: number;
    founderVotes: number;           // Total founder votes (in millions)
    publicVotes: number;            // Total public votes (in millions)
    totalVotes: number;             // Total votes (in millions)
    founderVotingPower: number;     // Founder voting % as decimal
    controlSecured: boolean;        // true if founders maintain control
    votingPowerShortfall?: number;  // How much below threshold (if any)
  };
  
  // Milestone Warrant Treatment (Contingent Dilution)
  milestoneWarrantTreatment?: {
    warrantSharesMillions: number;      // Shares to be issued if milestone hit
    warrantStrikePrice: number;         // Strike price per share
    milestoneProbability: number;       // Probability of milestone being hit (0-1)
    theoreticalPriceBeforeAdjustment: number;  // Theoretical price before warrant adjustment
    expectedDilutionCost: number;       // Expected cost in millions
    originalPreMoneyValuation: number;  // Original pre-money before warrant adjustment (millions)
    adjustedPreMoneyValuation: number;  // Pre-money after warrant adjustment (millions)
    warrantInTheMoney: boolean;         // true if theoretical price > strike price
  };
  
  // Strategic Partner Block Allocation Treatment
  strategicPartnerTreatment?: {
    partnerName: string;                    // Name of strategic partner
    partnerSharesMillions: number;          // Shares allocated to partner
    partnerPercentageOfFloat: number;       // Partner's allocation as % of expected float
    confidenceMultiplier: number;           // Boost applied to valuation (e.g., 1.08 = 8% boost)
    originalTheoreticalPrice: number;       // Theoretical price before partner boost
    boostedTheoreticalPrice: number;        // Theoretical price after partner boost
    originalOfferPrice: number;             // Offer price without partner
    boostedOfferPrice: number;              // Offer price with partner boost
    priceImpact: number;                    // Dollar increase per share
    priceImpactPercent: number;             // Percentage increase in offer price
  };
  
  // ============ ENHANCED MULTI-INSTRUMENT ANALYSIS ============
  
  // Multi-Convertible Analysis
  convertibleAnalysis?: {
    instruments: Array<{
      name: string;
      type: string;
      amountMillions: number;
      conversionPrice: number;
      sharesIssued: number;           // In millions
      triggered: boolean;
      probability: number;
      expectedShares: number;         // probability-weighted shares in millions
    }>;
    totalDeterministicShares: number;   // Shares that definitely convert
    totalExpectedShares: number;        // Probability-weighted expected shares
    totalConvertibleAmount: number;     // Total debt/SAFEs in millions
  };
  
  // Multi-Contingency Analysis
  contingencyAnalysis?: {
    liabilities: Array<{
      name: string;
      type: string;
      sharesMillions?: number;
      paymentMillions?: number;
      probability: number;
      expectedShares: number;           // probability-weighted
      expectedCostMillions: number;     // probability-weighted valuation impact
    }>;
    totalExpectedShares: number;        // Sum of expected shares
    totalExpectedCostMillions: number;  // Sum of expected costs
  };
  
  // Multi-Strategic Deal Analysis
  strategicDealAnalysis?: {
    deals: Array<{
      partnerName: string;
      sharesMillions: number;
      priceType: string;
      effectivePrice: number;
      premiumOrDiscount: number;
      demandBoostPercent: number;
    }>;
    totalAnchorAmount: number;
    totalDemandBoostPercent: number;
  };
  
  // Employee Option Dilution
  employeeOptionAnalysis?: {
    totalOptions: number;               // In millions
    avgStrikePrice: number;
    inTheMoneyOptions: number;          // Options with strike < offer price
    treasurySharesAdded: number;        // Net dilution using treasury stock method
    dilutionPercent: number;
  };
  
  // Blended Valuation Breakdown
  blendedValuationBreakdown?: {
    components: Array<{
      name: string;
      type: string;
      multiple: number;
      weight: number;
      weightedMultiple: number;          // multiple × weight
      valuationContribution: number;     // In millions
    }>;
    baseBlendedMultiple: number;         // Sum of weighted multiples
    effectiveMultiple: number;           // After growth premium
    growthPremiumApplied: boolean;
    growthPremiumPercent: number;        // e.g., 15 for 15%
    totalBlendedValuation: number;       // In millions
  };
  
  // Warnings
  warnings: string[];
  
  // Input assumptions for display
  assumptions: IPOAssumptions;
}

const IPO_PARSING_PROMPT = `You are an investment banking analyst expert at IPO pricing. Parse the following natural language description of an IPO (Initial Public Offering) and extract all relevant parameters.

Return a JSON object with the following structure:
{
  "companyName": "Company Name",
  "transactionDate": "YYYY-MM-DD" (use today if not specified),
  
  "ltmRevenue": number (Last Twelve Months Revenue in millions),
  "ltmEbitda": number or null (LTM EBITDA in millions, null if not provided),
  "industryRevenueMultiple": number (e.g., 10.0 for 10.0x revenue),
  "industryEbitdaMultiple": number or null (e.g., 15.0 for 15.0x EBITDA, null if not provided),
  "preIpoShares": number (REQUIRED - Pre-IPO Fully Diluted Shares Outstanding in millions. Look for "shares outstanding", "pre-IPO shares", "existing shares", "FD shares". Example: 18.5 for 18.5 million shares. NEVER return 0 or null for this field.),
  "primaryRaiseTarget": number (Primary Cash to Raise in millions),
  "ipoDiscount": number (as decimal, e.g., 0.20 for 20% discount),
  
  "secondaryShares": number or 0 (Secondary shares in millions, 0 if primary only),
  "greenshoePercent": number (as decimal, default 0.15 for 15% over-allotment),
  "underwritingFeePercent": number (as decimal, default 0.07 for 7% fee),
  
  "valuationMethod": "revenue" or "ebitda" or "blended" (default "revenue"),
  "blendWeight": number or 0.5 (weight for revenue in blended, default 0.5),
  
  "founderSharesMillions": number or null (Founder shares with super-voting rights in millions, null if no dual-class),
  "founderVoteMultiplier": number or null (Votes per founder share, e.g., 10 for 10x voting, null if no dual-class),
  "controlThreshold": number or null (Minimum voting % founders require as decimal, e.g., 0.40 for 40%, null if not specified),
  
  // ENHANCED MULTI-INSTRUMENT ARRAYS - Use these for complex scenarios with multiple instruments
  
  "convertibles": [ // Array of convertible instruments (SAFEs, debt, loans with conversion rights)
    {
      "name": "Series D SAFEs" (descriptive name),
      "type": "safe" | "debt" | "loan" | "note" | "other",
      "amountMillions": number (principal in millions),
      "triggerType": "lower_of" | "price_gt" | "price_gte" | "at_ipo_price" | "fixed_shares" | "conditional",
      "triggerPrice": number or null (price threshold),
      "triggerMultiplier": number or null (e.g., 0.8 for "80% of IPO price"),
      "triggerCondition": string or null (e.g., "IPO>25"),
      "fixedShares": number or null (shares in millions if fixed conversion),
      "probability": number (0-1, default 1.0 if certain)
    }
  ] or null,
  
  "contingencies": [ // Array of contingent liabilities (earnouts, warrants, litigation, grants)
    {
      "name": "Acquisition Earnout" (descriptive name),
      "type": "earnout" | "warrant" | "grant" | "litigation" | "royalty" | "milestone",
      "sharesMillions": number or null (shares to issue in millions),
      "strikePrice": number or null (for warrants),
      "paymentMillions": number or null (cash payment for litigation/royalty),
      "condition": string or null (description of trigger condition),
      "probability": number (0-1, estimated probability)
    }
  ] or null,
  
  "strategicDeals": [ // Array of strategic partner deals
    {
      "partnerName": "Google" (partner name),
      "sharesMillions": number (shares in millions),
      "priceType": "ipo_price" | "ipo_premium" | "fixed" | "discounted",
      "pricePremium": number or null (e.g., 0.05 for +5%),
      "priceDiscount": number or null (e.g., 0.10 for -10%),
      "fixedPrice": number or null,
      "probability": number (default 1.0),
      "isAnchorOrder": boolean (true if this is committed anchor demand)
    }
  ] or null,
  
  "anchorOrders": [ // Array of anchor/cornerstone investors
    {
      "investorName": "Sovereign Wealth Fund" (investor name),
      "amountMillions": number (committed investment in millions),
      "priceType": "ipo_price" | "ipo_premium" | "ipo_discount"
    }
  ] or null,
  
  "employeeOptions": { // Employee option pool for dilution
    "sharesMillions": number (total unexercised options in millions),
    "avgStrikePrice": number (weighted average strike price),
    "vestedPercent": number (0-1, percent vested, default 1.0)
  } or null,
  
  "valuationMultiples": [ // Multi-proxy blended valuation
    {
      "name": "Industry Revenue" (descriptive name),
      "type": "revenue" | "ebitda",
      "multiple": number (e.g., 28.0),
      "weight": number (0-1, weight in blend - all weights should sum to 1.0)
    }
  ] or null,
  
  // GROWTH PREMIUM (for high-growth companies)
  "revenueGrowthRate": number or null (revenue growth as decimal, e.g., 2.5 for 250% growth),
  "growthPremiumThreshold": number or 2.0 (threshold above which premium applies, e.g., 2.0 for 200%),
  "growthPremium": number or null (premium multiplier as decimal, e.g., 0.15 for 15% premium),
  
  // LEGACY SINGLE-INSTRUMENT FIELDS (for backwards compatibility)
  "convertibleDebtAmount": number or null,
  "conversionTriggerPrice": number or null,
  "conversionShares": number or null,
  "warrantSharesMillions": number or null,
  "warrantStrikePrice": number or null,
  "milestoneProbability": number or null,
  "strategicPartnerSharesMillions": number or null,
  "strategicPartnerName": string or null
}

PARSING RULES FOR COMPLEX INSTRUMENTS:

1. Multiple Convertibles: If there are multiple SAFEs, debt instruments, or loans with different conversion terms, use the "convertibles" array.
   - "lower_of" for terms like "converts at lower of $X or Y% of IPO price"
   - "price_gt" for terms like "converts if IPO price > $X"
   - "conditional" for probability-based conversions

2. Blended Valuations: If multiple valuation proxies are mentioned (e.g., "quantum computing 48x weighted 60%, AI proxy 24x weighted 40%"), use "valuationMultiples" array and set valuationMethod to "blended".
   - IMPORTANT: When using valuationMultiples, the effective multiple = sum of (each multiple × its weight)
   - Example: 48x × 0.6 + 24x × 0.4 = 28.8x + 9.6x = 38.4x base blended multiple

3. Growth Premium: If a growth premium or growth adjustment is mentioned (e.g., "15% premium for >200% growth"), extract:
   - revenueGrowthRate: the company's growth rate as decimal (e.g., 2.5 for 250%)
   - growthPremiumThreshold: the threshold for premium to apply (default 2.0 for 200%)
   - growthPremium: the premium multiplier as decimal (e.g., 0.15 for 15%)
   - The effective multiple is then: baseBlendedMultiple × (1 + growthPremium)

4. Contingent Liabilities: Use the "contingencies" array for earnouts, warrants, litigation, grants, and other probability-weighted items:
   - "earnout": Share issuance tied to performance (e.g., "4.1M earnout shares if revenue exceeds $250M (70% probability)")
     → { "name": "Revenue Earnout", "type": "earnout", "sharesMillions": 4.1, "condition": "revenue>250", "probability": 0.70 }
   - "warrant": Options with strike price (e.g., "3.3M warrants at $12 strike (55% chance they vest)")
     → { "name": "Performance Warrants", "type": "warrant", "sharesMillions": 3.3, "strikePrice": 12.0, "probability": 0.55 }
   - "litigation": Cash liability (e.g., "$42M patent settlement risk (30% probability)")
     → { "name": "Patent Litigation", "type": "litigation", "paymentMillions": 42.0, "probability": 0.30 }
   - "grant": Milestone-based shares (e.g., "2.25M shares for tech milestone (80% probability)")
     → { "name": "Tech Milestone Grant", "type": "grant", "sharesMillions": 2.25, "probability": 0.80 }

5. Strategic Deals with Premiums: If a partner pays "IPO price + X%", use priceType: "ipo_premium" with pricePremium: X/100.

6. Anchor Orders: Large committed investments that reduce execution risk should be captured in "anchorOrders".

7. Employee Options: Unexercised employee options should be captured in "employeeOptions" for treasury stock method dilution.

8. Probability Estimation: For contingent items without explicit probability, estimate based on:
   - FDA/regulatory approval: 0.60-0.80
   - Revenue/performance targets: 0.50-0.70
   - Litigation loss: 0.20-0.40
   - Partnership execution: 0.70-0.90

Default values if not specified:
- ipoDiscount: 0.15-0.25 (15-25% is standard, use 0.20 if not specified)
- greenshoePercent: 0.15 (15% standard over-allotment)
- underwritingFeePercent: 0.07 (7% standard for IPOs)
- secondaryShares: 0 (no secondary if not mentioned)
- valuationMethod: "revenue" (unless EBITDA or blended is specified)
- founderSharesMillions, founderVoteMultiplier: null if no dual-class share structure mentioned
- controlThreshold: 0.50 (50%) if dual-class is mentioned but no specific threshold given

IMPORTANT: Return ONLY valid JSON, no markdown, no explanations.`;

export async function parseIPODescription(
  description: string,
  customInstructions?: string,
  llmProvider: FinanceLLMProvider = 'zhi5'
): Promise<IPOAssumptions> {
  
  let userPrompt = `Extract IPO pricing assumptions from this description:\n\n${description}`;
  
  if (customInstructions) {
    userPrompt += `\n\nAdditional instructions: ${customInstructions}`;
  }

  let responseText: string;

  if (llmProvider === 'zhi1') {
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      max_tokens: 8000,
      temperature: 0,
      messages: [
        { role: 'system', content: IPO_PARSING_PROMPT },
        { role: 'user', content: userPrompt }
      ]
    });
    responseText = response.choices[0]?.message?.content || '';
    
  } else if (llmProvider === 'zhi2') {
    const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    const response = await anthropic.messages.create({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 8000,
      temperature: 0,
      system: IPO_PARSING_PROMPT,
      messages: [{ role: 'user', content: userPrompt }]
    });
    const content = response.content[0];
    if (content.type !== 'text') {
      throw new Error('Unexpected response type');
    }
    responseText = content.text;
    
  } else if (llmProvider === 'zhi3') {
    const deepseek = new OpenAI({
      baseURL: 'https://api.deepseek.com',
      apiKey: process.env.DEEPSEEK_API_KEY,
    });
    const response = await deepseek.chat.completions.create({
      model: 'deepseek-chat',
      max_tokens: 8000,
      temperature: 0,
      messages: [
        { role: 'system', content: IPO_PARSING_PROMPT },
        { role: 'user', content: userPrompt }
      ]
    });
    responseText = response.choices[0]?.message?.content || '';
    
  } else if (llmProvider === 'zhi4') {
    const response = await fetch('https://api.perplexity.ai/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.PERPLEXITY_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'sonar-pro',
        max_tokens: 8000,
        temperature: 0,
        messages: [
          { role: 'system', content: IPO_PARSING_PROMPT },
          { role: 'user', content: userPrompt }
        ]
      }),
    });
    const data = await response.json();
    responseText = data.choices?.[0]?.message?.content || '';
    
  } else {
    // zhi5 - Grok (default for finance)
    const grok = new OpenAI({
      baseURL: 'https://api.x.ai/v1',
      apiKey: process.env.GROK_API_KEY,
    });
    const response = await grok.chat.completions.create({
      model: 'grok-3',
      max_tokens: 8000,
      temperature: 0,
      messages: [
        { role: 'system', content: IPO_PARSING_PROMPT },
        { role: 'user', content: userPrompt }
      ]
    });
    responseText = response.choices[0]?.message?.content || '';
  }

  // Parse JSON response
  let jsonStr = responseText.trim();
  if (jsonStr.startsWith('```json')) {
    jsonStr = jsonStr.slice(7);
  }
  if (jsonStr.startsWith('```')) {
    jsonStr = jsonStr.slice(3);
  }
  if (jsonStr.endsWith('```')) {
    jsonStr = jsonStr.slice(0, -3);
  }
  
  const parsed = JSON.parse(jsonStr.trim());
  
  // Debug: Log what the LLM returned for key fields
  console.log(`[IPO Parser] LLM returned preIpoShares: ${parsed.preIpoShares} (type: ${typeof parsed.preIpoShares})`);
  console.log(`[IPO Parser] LLM returned ltmRevenue: ${parsed.ltmRevenue}, multiple: ${parsed.industryRevenueMultiple}`);
  console.log(`[IPO Parser] LLM returned primaryRaiseTarget: ${parsed.primaryRaiseTarget}`);
  
  // Handle alternative field names the LLM might use
  if (!parsed.preIpoShares && parsed.sharesOutstanding) {
    console.log(`[IPO Parser] Using sharesOutstanding as preIpoShares: ${parsed.sharesOutstanding}`);
    parsed.preIpoShares = parsed.sharesOutstanding;
  }
  if (!parsed.preIpoShares && parsed.existingShares) {
    console.log(`[IPO Parser] Using existingShares as preIpoShares: ${parsed.existingShares}`);
    parsed.preIpoShares = parsed.existingShares;
  }
  if (!parsed.preIpoShares && parsed.preIPOShares) {
    console.log(`[IPO Parser] Using preIPOShares as preIpoShares: ${parsed.preIPOShares}`);
    parsed.preIpoShares = parsed.preIPOShares;
  }
  if (!parsed.preIpoShares && parsed.totalShares) {
    console.log(`[IPO Parser] Using totalShares as preIpoShares: ${parsed.totalShares}`);
    parsed.preIpoShares = parsed.totalShares;
  }
  
  // Normalize values to millions - LLM may return in raw dollars instead of millions
  // If a dollar value is > 10,000, assume it's in raw dollars and convert to millions
  const normalizeToMillions = (val: number | undefined, fieldName: string): number | undefined => {
    if (val === undefined || val === null) return undefined;
    if (val > 10000) {
      console.log(`[IPO Parser] Normalizing ${fieldName}: ${val} -> ${val / 1000000}M (detected raw dollar value)`);
      return val / 1000000;
    }
    return val;
  };
  
  // Normalize share counts - if > 1000, likely in raw shares instead of millions
  const normalizeShares = (val: number | undefined, fieldName: string): number | undefined => {
    if (val === undefined || val === null) return undefined;
    if (val > 1000) {
      console.log(`[IPO Parser] Normalizing ${fieldName}: ${val} -> ${val / 1000000}M (detected raw share count)`);
      return val / 1000000;
    }
    return val;
  };
  
  // Process enhanced multi-instrument arrays
  let convertibles: ConvertibleInstrument[] | undefined = undefined;
  if (parsed.convertibles && Array.isArray(parsed.convertibles) && parsed.convertibles.length > 0) {
    convertibles = parsed.convertibles.map((c: any) => ({
      name: c.name || 'Unnamed Convertible',
      type: c.type || 'other',
      amountMillions: normalizeToMillions(c.amountMillions, 'convertible.amount') || 0,
      triggerType: c.triggerType || 'at_ipo_price',
      triggerPrice: c.triggerPrice || undefined,
      triggerPrice2: c.triggerPrice2 || undefined,
      triggerMultiplier: c.triggerMultiplier || undefined,
      triggerCondition: c.triggerCondition || undefined,
      fixedShares: normalizeShares(c.fixedShares, 'convertible.fixedShares'),
      probability: c.probability ?? 1.0,
      interestRate: c.interestRate || undefined,
    }));
    console.log(`[IPO Parser] Parsed ${convertibles.length} convertible instruments`);
  }
  
  let contingencies: ContingentLiability[] | undefined = undefined;
  if (parsed.contingencies && Array.isArray(parsed.contingencies) && parsed.contingencies.length > 0) {
    contingencies = parsed.contingencies.map((c: any) => ({
      name: c.name || 'Unnamed Contingency',
      type: c.type || 'milestone',
      sharesMillions: normalizeShares(c.sharesMillions, 'contingency.shares'),
      strikePrice: c.strikePrice || undefined,
      paymentMillions: normalizeToMillions(c.paymentMillions, 'contingency.payment'),
      condition: c.condition || undefined,
      probability: c.probability ?? 0.5,
    }));
    console.log(`[IPO Parser] Parsed ${contingencies.length} contingent liabilities:`);
    contingencies.forEach((cont, idx) => {
      console.log(`  [${idx + 1}] ${cont.name} (${cont.type}): ${cont.sharesMillions ?? 0}M shares, $${cont.paymentMillions ?? 0}M payment, ${(cont.probability * 100).toFixed(0)}% probability`);
    });
  } else {
    console.log(`[IPO Parser] No contingencies array found in parsed response`);
  }
  
  let strategicDeals: StrategicDeal[] | undefined = undefined;
  if (parsed.strategicDeals && Array.isArray(parsed.strategicDeals) && parsed.strategicDeals.length > 0) {
    strategicDeals = parsed.strategicDeals.map((d: any) => ({
      partnerName: d.partnerName || 'Strategic Partner',
      sharesMillions: normalizeShares(d.sharesMillions, 'deal.shares') || 0,
      priceType: d.priceType || 'ipo_price',
      pricePremium: d.pricePremium || undefined,
      priceDiscount: d.priceDiscount || undefined,
      fixedPrice: d.fixedPrice || undefined,
      probability: d.probability ?? 1.0,
      isAnchorOrder: d.isAnchorOrder ?? false,
    }));
    console.log(`[IPO Parser] Parsed ${strategicDeals.length} strategic deals`);
  }
  
  let anchorOrders: AnchorOrder[] | undefined = undefined;
  if (parsed.anchorOrders && Array.isArray(parsed.anchorOrders) && parsed.anchorOrders.length > 0) {
    anchorOrders = parsed.anchorOrders.map((a: any) => ({
      investorName: a.investorName || 'Anchor Investor',
      amountMillions: normalizeToMillions(a.amountMillions, 'anchor.amount') || 0,
      priceType: a.priceType || 'ipo_price',
      pricePremium: a.pricePremium || undefined,
      priceDiscount: a.priceDiscount || undefined,
    }));
    console.log(`[IPO Parser] Parsed ${anchorOrders.length} anchor orders`);
  }
  
  let employeeOptions: EmployeeOptionPool | undefined = undefined;
  if (parsed.employeeOptions && typeof parsed.employeeOptions === 'object') {
    employeeOptions = {
      sharesMillions: normalizeShares(parsed.employeeOptions.sharesMillions, 'options.shares') || 0,
      avgStrikePrice: parsed.employeeOptions.avgStrikePrice || 0,
      vestedPercent: parsed.employeeOptions.vestedPercent ?? 1.0,
    };
    console.log(`[IPO Parser] Parsed employee options: ${employeeOptions.sharesMillions}M @ $${employeeOptions.avgStrikePrice}`);
  }
  
  let valuationMultiples: ValuationMultiple[] | undefined = undefined;
  if (parsed.valuationMultiples && Array.isArray(parsed.valuationMultiples) && parsed.valuationMultiples.length > 0) {
    valuationMultiples = parsed.valuationMultiples.map((m: any) => ({
      name: m.name || 'Valuation Multiple',
      type: m.type || 'revenue',
      multiple: m.multiple || 0,
      weight: m.weight || 0,
    }));
    console.log(`[IPO Parser] Parsed ${valuationMultiples.length} valuation multiples`);
  }
  
  // Apply defaults with normalization
  return {
    companyName: parsed.companyName || 'Target Company',
    transactionDate: parsed.transactionDate || new Date().toISOString().split('T')[0],
    ltmRevenue: normalizeToMillions(parsed.ltmRevenue, 'ltmRevenue') || 0,
    ltmEbitda: normalizeToMillions(parsed.ltmEbitda, 'ltmEbitda'),
    industryRevenueMultiple: parsed.industryRevenueMultiple,
    industryEbitdaMultiple: parsed.industryEbitdaMultiple || undefined,
    preIpoShares: normalizeShares(parsed.preIpoShares, 'preIpoShares') || 0,
    primaryRaiseTarget: normalizeToMillions(parsed.primaryRaiseTarget, 'primaryRaiseTarget') || 0,
    ipoDiscount: parsed.ipoDiscount || 0.20,
    secondaryShares: normalizeShares(parsed.secondaryShares, 'secondaryShares') || 0,
    greenshoePercent: parsed.greenshoePercent ?? 0.15,
    underwritingFeePercent: parsed.underwritingFeePercent ?? 0.07,
    // Legacy single-instrument fields
    convertibleDebtAmount: normalizeToMillions(parsed.convertibleDebtAmount, 'convertibleDebtAmount'),
    conversionTriggerPrice: parsed.conversionTriggerPrice || undefined,
    conversionShares: normalizeShares(parsed.conversionShares, 'conversionShares'),
    valuationMethod: parsed.valuationMethod || 'revenue',
    blendWeight: parsed.blendWeight ?? 0.5,
    // Dual-class share structure
    founderSharesMillions: normalizeShares(parsed.founderSharesMillions, 'founderSharesMillions'),
    founderVoteMultiplier: parsed.founderVoteMultiplier || undefined,
    controlThreshold: parsed.controlThreshold ?? (parsed.founderSharesMillions ? 0.50 : undefined),
    // Legacy milestone warrants
    warrantSharesMillions: normalizeShares(parsed.warrantSharesMillions, 'warrantSharesMillions'),
    warrantStrikePrice: parsed.warrantStrikePrice || undefined,
    milestoneProbability: parsed.milestoneProbability || undefined,
    // Legacy strategic partner allocation
    strategicPartnerSharesMillions: normalizeShares(parsed.strategicPartnerSharesMillions, 'strategicPartnerSharesMillions'),
    strategicPartnerName: parsed.strategicPartnerName || undefined,
    // Enhanced multi-instrument arrays
    convertibles,
    contingencies,
    strategicDeals,
    anchorOrders,
    employeeOptions,
    valuationMultiples,
    // Growth premium for high-growth companies
    revenueGrowthRate: parsed.revenueGrowthRate || undefined,
    growthPremiumThreshold: parsed.growthPremiumThreshold || 2.0,
    growthPremium: parsed.growthPremium || undefined,
  };
}

export function calculateIPOPricing(assumptions: IPOAssumptions): IPOPricingResult {
  const {
    companyName,
    transactionDate,
    ltmRevenue,
    ltmEbitda,
    industryRevenueMultiple,
    industryEbitdaMultiple,
    preIpoShares: originalPreIpoShares,
    primaryRaiseTarget,
    ipoDiscount,
    secondaryShares = 0,
    greenshoePercent = 0.15,
    underwritingFeePercent = 0.07,
    convertibleDebtAmount,
    conversionTriggerPrice,
    conversionShares,
    valuationMethod,
    blendWeight = 0.5,
    founderSharesMillions,
    founderVoteMultiplier,
    controlThreshold,
    warrantSharesMillions,
    warrantStrikePrice,
    milestoneProbability,
    strategicPartnerSharesMillions,
    strategicPartnerName,
  } = assumptions;

  const warnings: string[] = [];
  
  // CRITICAL VALIDATION: Prevent division by zero
  if (!originalPreIpoShares || originalPreIpoShares <= 0) {
    console.error(`[IPO Model] CRITICAL ERROR: preIpoShares is ${originalPreIpoShares} - cannot calculate pricing`);
    throw new Error(`Pre-IPO shares outstanding is required and must be greater than 0. The LLM failed to extract this value from the input. Please specify the number of shares outstanding (e.g., "18.5 million shares outstanding" or "pre-IPO shares: 25M").`);
  }
  
  if (!ltmRevenue || ltmRevenue <= 0) {
    console.error(`[IPO Model] CRITICAL ERROR: ltmRevenue is ${ltmRevenue}`);
    throw new Error(`LTM Revenue is required and must be greater than 0. Please specify the company's revenue (e.g., "$92 million revenue").`);
  }
  
  if (!industryRevenueMultiple || industryRevenueMultiple <= 0) {
    console.error(`[IPO Model] CRITICAL ERROR: industryRevenueMultiple is ${industryRevenueMultiple}`);
    throw new Error(`Valuation multiple is required and must be greater than 0. Please specify a revenue or EBITDA multiple (e.g., "40x revenue multiple").`);
  }
  let convertibleDebtTreatment: IPOPricingResult['convertibleDebtTreatment'] = undefined;

  // ============ PHASE 1: Calculate Pre-Money Valuation ============
  let preMoneyValuation: number;
  
  if (valuationMethod === 'revenue') {
    // Pre-Money Valuation = LTM Revenue × Revenue Multiple
    preMoneyValuation = ltmRevenue * industryRevenueMultiple;
    console.log(`[IPO Model] Revenue Multiple: $${ltmRevenue}M × ${industryRevenueMultiple}x = $${preMoneyValuation}M`);
    
  } else if (valuationMethod === 'ebitda' && ltmEbitda && industryEbitdaMultiple) {
    // Pre-Money Valuation = LTM EBITDA × EBITDA Multiple
    preMoneyValuation = ltmEbitda * industryEbitdaMultiple;
    console.log(`[IPO Model] EBITDA Multiple: $${ltmEbitda}M × ${industryEbitdaMultiple}x = $${preMoneyValuation}M`);
    
  } else if (valuationMethod === 'blended' && ltmEbitda && industryEbitdaMultiple) {
    // Blended: Weight between revenue and EBITDA multiples
    const revenueVal = ltmRevenue * industryRevenueMultiple;
    const ebitdaVal = ltmEbitda * industryEbitdaMultiple;
    preMoneyValuation = (revenueVal * blendWeight) + (ebitdaVal * (1 - blendWeight));
    console.log(`[IPO Model] Blended: Rev($${revenueVal}M × ${(blendWeight*100).toFixed(0)}%) + EBITDA($${ebitdaVal}M × ${((1-blendWeight)*100).toFixed(0)}%) = $${preMoneyValuation}M`);
    
  } else {
    // Fallback to revenue - this happens if EBITDA/blended was selected but data is missing
    preMoneyValuation = ltmRevenue * industryRevenueMultiple;
    warnings.push('Missing EBITDA data for selected valuation method. Defaulted to revenue multiple.');
  }

  // ============ ENHANCED MULTI-INSTRUMENT PROCESSING ============
  // Check if enhanced instrument arrays are present
  const hasEnhancedInstruments = !!(
    (assumptions.convertibles && assumptions.convertibles.length > 0) ||
    (assumptions.contingencies && assumptions.contingencies.length > 0) ||
    (assumptions.strategicDeals && assumptions.strategicDeals.length > 0) ||
    (assumptions.anchorOrders && assumptions.anchorOrders.length > 0) ||
    assumptions.employeeOptions ||
    (assumptions.valuationMultiples && assumptions.valuationMultiples.length > 0)
  );
  
  let engineResult: InstrumentEngineResult | undefined;
  let convertibleAnalysis: IPOPricingResult['convertibleAnalysis'] = undefined;
  let contingencyAnalysis: IPOPricingResult['contingencyAnalysis'] = undefined;
  let strategicDealAnalysis: IPOPricingResult['strategicDealAnalysis'] = undefined;
  let employeeOptionAnalysis: IPOPricingResult['employeeOptionAnalysis'] = undefined;
  let blendedValuationBreakdown: IPOPricingResult['blendedValuationBreakdown'] = undefined;
  
  // Calculate first-pass theoretical price for engine
  const firstPassTheoreticalPrice = preMoneyValuation / originalPreIpoShares;
  const tentativeOfferPrice = firstPassTheoreticalPrice * (1 - ipoDiscount);
  
  if (hasEnhancedInstruments) {
    console.log(`[IPO Model] ============ ENHANCED MULTI-INSTRUMENT ENGINE ============`);
    
    // Run instrument engine with all complex instruments
    engineResult = runInstrumentEngine(
      assumptions,
      preMoneyValuation,
      firstPassTheoreticalPrice,
      tentativeOfferPrice
    );
    
    // Log engine output
    for (const log of engineResult.logs) {
      console.log(log);
    }
    
    // Apply engine adjustments
    if (assumptions.valuationMultiples && assumptions.valuationMultiples.length > 0) {
      preMoneyValuation = engineResult.adjustedPreMoneyValuation;
      console.log(`[IPO Model] Engine adjusted pre-money: $${preMoneyValuation.toFixed(2)}M`);
    }
    
    // Build enhanced result objects
    if (engineResult.convertibleResults.length > 0) {
      convertibleAnalysis = {
        instruments: engineResult.convertibleResults.map(c => ({
          name: c.name,
          type: c.type,
          amountMillions: c.amountMillions,
          conversionPrice: c.conversionPrice,
          sharesIssued: c.sharesIssued,
          triggered: c.triggered,
          probability: c.probability,
          expectedShares: c.expectedShares,
        })),
        totalDeterministicShares: engineResult.totalDeterministicConversionShares,
        totalExpectedShares: engineResult.totalExpectedConversionShares,
        totalConvertibleAmount: engineResult.convertibleResults.reduce((sum, c) => sum + c.amountMillions, 0),
      };
    }
    
    if (engineResult.contingencyResults.length > 0) {
      contingencyAnalysis = {
        liabilities: engineResult.contingencyResults.map(c => ({
          name: c.name,
          type: c.type,
          sharesMillions: c.sharesMillions,
          paymentMillions: c.paymentMillions,
          probability: c.probability,
          expectedShares: c.expectedShares,
          expectedCostMillions: c.expectedCostMillions,
        })),
        totalExpectedShares: engineResult.totalExpectedContingencyShares,
        totalExpectedCostMillions: engineResult.totalExpectedContingencyCost,
      };
    }
    
    if (engineResult.strategicDealResults.length > 0) {
      strategicDealAnalysis = {
        deals: engineResult.strategicDealResults,
        totalAnchorAmount: engineResult.totalAnchorAmount,
        totalDemandBoostPercent: (engineResult.demandBoostMultiplier - 1) * 100,
      };
    }
    
    if (assumptions.employeeOptions && engineResult.employeeOptionDilution > 0) {
      const options = assumptions.employeeOptions;
      employeeOptionAnalysis = {
        totalOptions: options.sharesMillions,
        avgStrikePrice: options.avgStrikePrice,
        inTheMoneyOptions: options.sharesMillions * (options.vestedPercent ?? 1.0),
        treasurySharesAdded: engineResult.employeeOptionDilution,
        dilutionPercent: (engineResult.employeeOptionDilution / originalPreIpoShares) * 100,
      };
    }
    
    if (engineResult.blendedValuationComponents.length > 0) {
      blendedValuationBreakdown = {
        components: engineResult.blendedValuationComponents.map(c => ({
          name: c.name,
          type: c.type,
          multiple: c.multiple,
          weight: c.weight,
          weightedMultiple: c.weightedMultiple,
          valuationContribution: c.contribution,
        })),
        baseBlendedMultiple: engineResult.baseBlendedMultiple,
        effectiveMultiple: engineResult.blendedMultiple,
        growthPremiumApplied: engineResult.growthPremiumApplied,
        growthPremiumPercent: engineResult.growthPremiumPercent,
        totalBlendedValuation: engineResult.adjustedPreMoneyValuation,
      };
    }
    
    // Add engine warnings
    if (engineResult.totalExpectedContingencyCost > 0) {
      warnings.push(`Probability-weighted contingent liabilities: $${engineResult.totalExpectedContingencyCost.toFixed(2)}M expected cost.`);
    }
    if (engineResult.totalExpectedConversionShares > 0) {
      warnings.push(`Expected conversion shares: ${engineResult.totalExpectedConversionShares.toFixed(3)}M (includes probability-weighted instruments).`);
    }
    if (engineResult.totalExpectedContingencyShares > 0) {
      warnings.push(`Expected contingent shares: ${engineResult.totalExpectedContingencyShares.toFixed(3)}M (probability-weighted).`);
    }
    if (engineResult.employeeOptionDilution > 0) {
      warnings.push(`Employee option dilution: ${engineResult.employeeOptionDilution.toFixed(3)}M shares (treasury stock method).`);
    }
    if (engineResult.growthPremiumApplied) {
      warnings.push(`Growth premium applied: ${engineResult.growthPremiumPercent.toFixed(0)}% premium on ${engineResult.baseBlendedMultiple.toFixed(2)}x base multiple = ${engineResult.blendedMultiple.toFixed(2)}x effective multiple.`);
    }
  }

  // ============ PHASE 1.5: Handle Convertible Debt FIRST (Adjusts Share Count) ============
  // IMPORTANT: Convertible debt must be processed BEFORE milestone warrants
  // because warrants use the adjusted share count for theoretical price calculation
  // Note: This is legacy single-convertible processing; enhanced processing above handles arrays
  
  let adjustedPreIpoShares = originalPreIpoShares;
  let conversionActivated = false;
  
  // Check if we have convertible debt and if it triggers
  if (convertibleDebtAmount && conversionTriggerPrice && conversionShares) {
    console.log(`[IPO Model] ============ CONVERTIBLE DEBT CHECK (PHASE 1.5) ============`);
    console.log(`[IPO Model] First-Pass Theoretical Price: $${firstPassTheoreticalPrice.toFixed(2)}`);
    console.log(`[IPO Model] Tentative Offer Price: $${tentativeOfferPrice.toFixed(2)}`);
    console.log(`[IPO Model] Conversion Trigger: $${conversionTriggerPrice.toFixed(2)}`);
    
    if (tentativeOfferPrice > conversionTriggerPrice) {
      // DEBT CONVERTS: Add conversion shares
      conversionActivated = true;
      adjustedPreIpoShares = originalPreIpoShares + conversionShares;
      console.log(`[IPO Model] CONVERSION ACTIVATED! Price $${tentativeOfferPrice.toFixed(2)} > Trigger $${conversionTriggerPrice.toFixed(2)}`);
      console.log(`[IPO Model] Adding ${conversionShares}M conversion shares`);
      console.log(`[IPO Model] Adjusted Pre-IPO Shares: ${originalPreIpoShares}M + ${conversionShares}M = ${adjustedPreIpoShares}M`);
      warnings.push(`Convertible debt ($${convertibleDebtAmount}M) converted at $${conversionTriggerPrice.toFixed(2)} trigger, adding ${conversionShares}M shares.`);
    } else {
      console.log(`[IPO Model] No conversion: Price $${tentativeOfferPrice.toFixed(2)} <= Trigger $${conversionTriggerPrice.toFixed(2)}`);
    }
    
    // Record convertible debt treatment for output
    convertibleDebtTreatment = {
      triggerPrice: conversionTriggerPrice,
      debtAmount: convertibleDebtAmount,
      conversionShares: conversionShares,
      conversionActivated,
      originalPreIpoShares,
      adjustedPreIpoShares,
      tentativeOfferPrice,
    };
  }

  // ============ PHASE 2: Milestone Warrant Adjustment (Contingent Dilution) ============
  // This adjusts valuation DOWNWARD to account for potential future share issuance
  // IMPORTANT: Uses adjustedPreIpoShares (post-conversion) for theoretical price calculation
  let milestoneWarrantTreatment: IPOPricingResult['milestoneWarrantTreatment'] = undefined;
  const originalPreMoneyValuation = preMoneyValuation; // Store original for reporting
  
  if (warrantSharesMillions && warrantStrikePrice !== undefined && milestoneProbability !== undefined && milestoneProbability > 0) {
    console.log(`[IPO Model] ============ MILESTONE WARRANT CHECK (PHASE 2) ============`);
    
    // Calculate theoretical price USING ADJUSTED SHARE COUNT (post-conversion if applicable)
    const theoreticalPriceForWarrantCheck = preMoneyValuation / adjustedPreIpoShares;
    console.log(`[IPO Model] Shares for Warrant Calculation: ${adjustedPreIpoShares}M (post-conversion if applicable)`);
    console.log(`[IPO Model] Pre-Adjustment Theoretical Price: $${theoreticalPriceForWarrantCheck.toFixed(2)}`);
    console.log(`[IPO Model] Warrant Strike Price: $${warrantStrikePrice.toFixed(2)}`);
    console.log(`[IPO Model] Warrant Shares: ${warrantSharesMillions}M`);
    console.log(`[IPO Model] Milestone Probability: ${(milestoneProbability * 100).toFixed(0)}%`);
    
    // Calculate expected dilution cost
    // Cost = (Current Fair Value - Strike Price) × Shares × Probability
    // Only apply if warrant is "in-the-money" (theoretical > strike)
    const warrantSpread = theoreticalPriceForWarrantCheck - warrantStrikePrice;
    const warrantInTheMoney = warrantSpread > 0;
    
    if (warrantInTheMoney) {
      // Expected cost in dollars: spread × shares × probability
      // warrantSharesMillions is in millions, so multiply by 1M to get actual shares
      const expectedDilutionCostDollars = warrantSpread * (warrantSharesMillions * 1000000) * milestoneProbability;
      const expectedDilutionCostMillions = expectedDilutionCostDollars / 1000000;
      
      // Reduce pre-money valuation by expected cost
      preMoneyValuation = preMoneyValuation - expectedDilutionCostMillions;
      
      console.log(`[IPO Model] Warrant IN-THE-MONEY: $${theoreticalPriceForWarrantCheck.toFixed(2)} > $${warrantStrikePrice.toFixed(2)}`);
      console.log(`[IPO Model] Spread: $${warrantSpread.toFixed(2)} per share`);
      console.log(`[IPO Model] Expected Dilution Cost: $${warrantSpread.toFixed(2)} × ${warrantSharesMillions}M shares × ${(milestoneProbability * 100).toFixed(0)}% = $${expectedDilutionCostMillions.toFixed(2)}M`);
      console.log(`[IPO Model] Adjusted Pre-Money: $${originalPreMoneyValuation.toFixed(2)}M - $${expectedDilutionCostMillions.toFixed(2)}M = $${preMoneyValuation.toFixed(2)}M`);
      
      warnings.push(`Milestone warrant adjustment: -$${expectedDilutionCostMillions.toFixed(2)}M (${(milestoneProbability * 100).toFixed(0)}% probability × ${warrantSharesMillions}M shares at $${warrantStrikePrice.toFixed(2)} strike).`);
      
      milestoneWarrantTreatment = {
        warrantSharesMillions,
        warrantStrikePrice,
        milestoneProbability,
        theoreticalPriceBeforeAdjustment: theoreticalPriceForWarrantCheck,
        expectedDilutionCost: expectedDilutionCostMillions,
        originalPreMoneyValuation: originalPreMoneyValuation,
        adjustedPreMoneyValuation: preMoneyValuation,
        warrantInTheMoney: true,
      };
    } else {
      console.log(`[IPO Model] Warrant OUT-OF-THE-MONEY: $${theoreticalPriceForWarrantCheck.toFixed(2)} <= $${warrantStrikePrice.toFixed(2)}`);
      console.log(`[IPO Model] No valuation adjustment needed`);
      
      milestoneWarrantTreatment = {
        warrantSharesMillions,
        warrantStrikePrice,
        milestoneProbability,
        theoreticalPriceBeforeAdjustment: theoreticalPriceForWarrantCheck,
        expectedDilutionCost: 0,
        originalPreMoneyValuation: originalPreMoneyValuation,
        adjustedPreMoneyValuation: preMoneyValuation,
        warrantInTheMoney: false,
      };
    }
  }

  // ============ PHASE 2.5: Strategic Partner Block Allocation (Confidence Boost) ============
  // Guaranteed demand from strategic partner increases pricing power
  let strategicPartnerTreatment: IPOPricingResult['strategicPartnerTreatment'] = undefined;
  let confidenceMultiplier = 1.0;
  
  if (strategicPartnerSharesMillions && strategicPartnerSharesMillions > 0) {
    console.log(`[IPO Model] ============ STRATEGIC PARTNER ALLOCATION (PHASE 2.5) ============`);
    
    // Calculate base theoretical price for reference
    const baseTheoreticalPrice = preMoneyValuation / adjustedPreIpoShares;
    const baseOfferPrice = baseTheoreticalPrice * (1 - ipoDiscount);
    
    // Estimate expected float (typically ~10-25% of pre-IPO shares)
    // Use approximate primary raise / offer price to estimate new shares
    const estimatedNewShares = primaryRaiseTarget / baseOfferPrice;
    const estimatedFloat = estimatedNewShares + secondaryShares;
    
    // Partner percentage of expected float
    const partnerPercentageOfFloat = strategicPartnerSharesMillions / estimatedFloat;
    
    // Confidence multiplier: Based on partner size relative to float
    // Formula: 1.0 + (partner% × 0.3), capped at 1.15 (max 15% boost)
    confidenceMultiplier = Math.min(1.0 + (partnerPercentageOfFloat * 0.3), 1.15);
    
    // Boost theoretical price (applied before discount)
    const boostedTheoreticalPrice = baseTheoreticalPrice * confidenceMultiplier;
    const boostedOfferPrice = boostedTheoreticalPrice * (1 - ipoDiscount);
    
    const priceImpact = boostedOfferPrice - baseOfferPrice;
    const priceImpactPercent = (priceImpact / baseOfferPrice) * 100;
    
    console.log(`[IPO Model] Strategic Partner: ${strategicPartnerName || 'Unnamed Partner'}`);
    console.log(`[IPO Model] Partner Shares: ${strategicPartnerSharesMillions}M`);
    console.log(`[IPO Model] Estimated Float: ${estimatedFloat.toFixed(3)}M shares`);
    console.log(`[IPO Model] Partner % of Float: ${(partnerPercentageOfFloat * 100).toFixed(1)}%`);
    console.log(`[IPO Model] Confidence Multiplier: ${confidenceMultiplier.toFixed(4)}x`);
    console.log(`[IPO Model] Base Theoretical: $${baseTheoreticalPrice.toFixed(2)} → Boosted: $${boostedTheoreticalPrice.toFixed(2)}`);
    console.log(`[IPO Model] Base Offer: $${baseOfferPrice.toFixed(2)} → Boosted: $${boostedOfferPrice.toFixed(2)}`);
    console.log(`[IPO Model] Price Impact: +$${priceImpact.toFixed(2)} (+${priceImpactPercent.toFixed(1)}%)`);
    
    warnings.push(`Strategic partner (${strategicPartnerName || 'Partner'}) allocation of ${strategicPartnerSharesMillions}M shares: +${priceImpactPercent.toFixed(1)}% price boost due to guaranteed demand.`);
    
    strategicPartnerTreatment = {
      partnerName: strategicPartnerName || 'Strategic Partner',
      partnerSharesMillions: strategicPartnerSharesMillions,
      partnerPercentageOfFloat: partnerPercentageOfFloat,
      confidenceMultiplier: confidenceMultiplier,
      originalTheoreticalPrice: baseTheoreticalPrice,
      boostedTheoreticalPrice: boostedTheoreticalPrice,
      originalOfferPrice: baseOfferPrice,
      boostedOfferPrice: boostedOfferPrice,
      priceImpact: priceImpact,
      priceImpactPercent: priceImpactPercent,
    };
  }

  // ============ PHASE 3: Calculate Final Theoretical & Offer Price ============
  // Use adjusted share count (may be same as original if no conversion)
  // Apply confidence multiplier from strategic partner (1.0 if none)
  
  // If engine was used, apply engine adjustments to share count and valuation
  if (engineResult) {
    // Add engine-calculated share adjustments
    const engineShareAdjustment = 
      engineResult.totalExpectedConversionShares + 
      engineResult.totalExpectedContingencyShares + 
      engineResult.employeeOptionDilution;
    
    if (engineShareAdjustment > 0) {
      adjustedPreIpoShares += engineShareAdjustment;
      console.log(`[IPO Model] Engine Share Adjustment: +${engineShareAdjustment.toFixed(3)}M shares → ${adjustedPreIpoShares.toFixed(3)}M total`);
    }
    
    // Apply contingency cost adjustment
    if (engineResult.totalExpectedContingencyCost > 0) {
      preMoneyValuation -= engineResult.totalExpectedContingencyCost;
      console.log(`[IPO Model] Engine Contingency Cost: -$${engineResult.totalExpectedContingencyCost.toFixed(2)}M → $${preMoneyValuation.toFixed(2)}M valuation`);
    }
    
    // Apply demand boost if not already applied via legacy strategic partner
    if (engineResult.demandBoostMultiplier > 1.0 && confidenceMultiplier === 1.0) {
      confidenceMultiplier = engineResult.demandBoostMultiplier;
      console.log(`[IPO Model] Engine Demand Boost: ${((confidenceMultiplier - 1) * 100).toFixed(1)}%`);
    }
  }
  
  // FIX: Apply IPO discount to enterprise valuation FIRST, then convert to per-share
  // "Move your IPO discount logic so it applies to enterprise valuation before converting to equity value or dividing by share count; never apply discounts to the per-share output."
  
  // Step 1: Calculate adjusted pre-money valuation (with confidence/demand multiplier)
  const adjustedPreMoneyValuation = preMoneyValuation * confidenceMultiplier;
  console.log(`[IPO Model] Adjusted Pre-Money: $${preMoneyValuation}M${confidenceMultiplier > 1 ? ` × ${confidenceMultiplier.toFixed(4)}x = $${adjustedPreMoneyValuation.toFixed(2)}M` : ''}`);
  
  // Step 2: Apply IPO discount to ENTERPRISE VALUATION (not per-share price)
  const effectiveDiscount = Math.min(Math.max(ipoDiscount, 0), 0.5); // Cap at 50%
  const discountedPreMoneyValuation = adjustedPreMoneyValuation * (1 - effectiveDiscount);
  console.log(`[IPO Model] Discounted Pre-Money: $${adjustedPreMoneyValuation.toFixed(2)}M × (1 - ${(effectiveDiscount*100).toFixed(0)}%) = $${discountedPreMoneyValuation.toFixed(2)}M`);
  
  // Step 3: Theoretical price is UNDISCOUNTED (fair value)
  const theoreticalPrice = adjustedPreMoneyValuation / adjustedPreIpoShares;
  console.log(`[IPO Model] Theoretical Price (Undiscounted): $${adjustedPreMoneyValuation.toFixed(2)}M / ${adjustedPreIpoShares}M shares = $${theoreticalPrice.toFixed(2)}/share`);
  
  // Step 4: Offer price is derived from DISCOUNTED valuation
  let offerPrice = discountedPreMoneyValuation / adjustedPreIpoShares;
  console.log(`[IPO Model] Offer Price: $${discountedPreMoneyValuation.toFixed(2)}M / ${adjustedPreIpoShares}M shares = $${offerPrice.toFixed(2)}/share`);

  // Edge case: If conversion dilution pushed price below trigger, force just above
  if (conversionActivated && conversionTriggerPrice && offerPrice < conversionTriggerPrice) {
    console.log(`[IPO Model] EDGE CASE: Dilution pushed price ($${offerPrice.toFixed(2)}) below trigger ($${conversionTriggerPrice.toFixed(2)})`);
    offerPrice = conversionTriggerPrice + 0.01;
    console.log(`[IPO Model] Forcing price to $${offerPrice.toFixed(2)} (just above trigger)`);
    warnings.push(`Dilution pushed price below trigger. Forced to $${offerPrice.toFixed(2)}.`);
  }

  // Warning for low price
  if (offerPrice < 1.00) {
    warnings.push('Warning: Low price per share. Inputs may be unrealistic. Consider adjusting shares outstanding or multiples.');
  }
  if (offerPrice < 5.00) {
    warnings.push('Price below $5.00 may face institutional investor restrictions (penny stock concerns).');
  }

  // ============ PHASE 4: Calculate Shares to Issue ============
  // Use adjusted share count for all remaining calculations
  const preIpoShares = adjustedPreIpoShares;
  
  // New Shares Issued = Primary Raise Target / Offer Price
  const newSharesIssued = primaryRaiseTarget / offerPrice;
  console.log(`[IPO Model] New Shares: $${primaryRaiseTarget}M / $${offerPrice.toFixed(2)} = ${(newSharesIssued * 1000000).toLocaleString()} shares (${newSharesIssued.toFixed(4)}M)`);

  // ============ PHASE 5: Calculate Post-Money Valuation ============
  const impliedPreMoneyAtOffer = offerPrice * preIpoShares;
  const postMoneyValuation = impliedPreMoneyAtOffer + primaryRaiseTarget;
  console.log(`[IPO Model] Post-Money: ($${offerPrice.toFixed(2)} × ${preIpoShares}M) + $${primaryRaiseTarget}M = $${postMoneyValuation.toFixed(2)}M`);

  // ============ PHASE 6: Calculate Offering Structure ============
  const totalPrimarySecondary = newSharesIssued + secondaryShares;
  const greenshoeShares = totalPrimarySecondary * greenshoePercent;
  const totalSharesOffered = totalPrimarySecondary + greenshoeShares;
  
  // Post-IPO shares outstanding (excluding greenshoe initially)
  const postIpoSharesOutstanding = preIpoShares + newSharesIssued;
  
  // With full greenshoe exercise
  const postIpoSharesWithGreenshoe = postIpoSharesOutstanding + greenshoeShares;

  // ============ PHASE 7: Calculate Proceeds ============
  const grossPrimaryProceeds = newSharesIssued * offerPrice;
  const secondaryProceeds = secondaryShares * offerPrice;
  const greenshoeProceeds = greenshoeShares * offerPrice;
  const totalGrossProceeds = (totalPrimarySecondary * offerPrice) + greenshoeProceeds;
  
  // Underwriting fees (typically on gross proceeds)
  const underwritingFees = totalGrossProceeds * underwritingFeePercent;
  const netPrimaryProceeds = grossPrimaryProceeds - (grossPrimaryProceeds / totalGrossProceeds * underwritingFees);

  // ============ PHASE 8: Calculate Ownership & Dilution ============
  const percentageSold = (newSharesIssued / postIpoSharesOutstanding) * 100;
  const existingHoldersDilution = (1 - (preIpoShares / postIpoSharesOutstanding)) * 100;
  
  // ============ PHASE 9: Trading Metrics ============
  // Expected first day pop = inverse of discount
  // If 20% discount applied, fair value is 25% higher than offer
  const expectedFirstDayPop = (1 / (1 - ipoDiscount) - 1) * 100;
  const marketCapAtOffer = offerPrice * postIpoSharesOutstanding;

  console.log(`[IPO Model] ============ IPO PRICING SUMMARY ============`);
  console.log(`[IPO Model] Offer Price: $${offerPrice.toFixed(2)}`);
  console.log(`[IPO Model] Pre-Money (at offer): $${impliedPreMoneyAtOffer.toFixed(2)}M`);
  console.log(`[IPO Model] Post-Money: $${postMoneyValuation.toFixed(2)}M`);
  console.log(`[IPO Model] Primary Shares: ${(newSharesIssued * 1000000).toLocaleString()} (${newSharesIssued.toFixed(4)}M)`);
  console.log(`[IPO Model] Dilution: ${existingHoldersDilution.toFixed(1)}%`);
  console.log(`[IPO Model] Expected First Day Pop: ${expectedFirstDayPop.toFixed(1)}%`);
  if (conversionActivated) {
    console.log(`[IPO Model] Convertible Debt: CONVERTED (added ${conversionShares}M shares)`);
  }

  // ============ PHASE 10: Dual-Class Share Voting Control Analysis ============
  let votingControlAnalysis: IPOPricingResult['votingControlAnalysis'] = undefined;
  
  if (founderSharesMillions && founderVoteMultiplier && controlThreshold) {
    console.log(`[IPO Model] ============ DUAL-CLASS VOTING ANALYSIS ============`);
    
    // Calculate votes (all in millions for consistency)
    const founderVotes = founderSharesMillions * founderVoteMultiplier;
    const publicShares = postIpoSharesOutstanding - founderSharesMillions;
    const publicVotes = publicShares * 1;  // Public shares always get 1 vote
    const totalVotes = founderVotes + publicVotes;
    const founderVotingPower = founderVotes / totalVotes;
    const controlSecured = founderVotingPower >= controlThreshold;
    
    console.log(`[IPO Model] Founder Shares: ${founderSharesMillions}M × ${founderVoteMultiplier}x = ${founderVotes.toFixed(2)}M votes`);
    console.log(`[IPO Model] Public Shares: ${publicShares.toFixed(4)}M × 1x = ${publicVotes.toFixed(2)}M votes`);
    console.log(`[IPO Model] Total Votes: ${totalVotes.toFixed(2)}M`);
    console.log(`[IPO Model] Founder Voting Power: ${(founderVotingPower * 100).toFixed(1)}%`);
    console.log(`[IPO Model] Control Threshold: ${(controlThreshold * 100).toFixed(1)}%`);
    
    if (controlSecured) {
      console.log(`[IPO Model] ✓ VOTING CONTROL SECURED: Founders have ${(founderVotingPower * 100).toFixed(1)}% voting power`);
    } else {
      const shortfall = controlThreshold - founderVotingPower;
      console.log(`[IPO Model] ⚠️  CRITICAL WARNING: VOTING CONTROL BREACHED`);
      console.log(`[IPO Model]    Founders will have only ${(founderVotingPower * 100).toFixed(1)}% voting power`);
      console.log(`[IPO Model]    Required threshold: ${(controlThreshold * 100).toFixed(1)}%`);
      console.log(`[IPO Model]    Shortfall: ${(shortfall * 100).toFixed(1)}%`);
      warnings.push(`CRITICAL: Founders will lose voting control. Voting power (${(founderVotingPower * 100).toFixed(1)}%) is below required threshold (${(controlThreshold * 100).toFixed(1)}%).`);
    }
    
    votingControlAnalysis = {
      founderSharesMillions,
      founderVoteMultiplier,
      controlThreshold,
      founderVotes,
      publicVotes,
      totalVotes,
      founderVotingPower,
      controlSecured,
      votingPowerShortfall: controlSecured ? undefined : (controlThreshold - founderVotingPower),
    };
  }

  return {
    companyName,
    transactionDate,
    
    preMoneyValuation,
    theoreticalPrice,
    offerPrice,
    postMoneyValuation,
    impliedPreMoneyAtOffer,
    
    newSharesIssued,
    secondarySharesSold: secondaryShares,
    totalSharesOffered,
    greenshoeShares,
    
    grossPrimaryProceeds,
    netPrimaryProceeds,
    secondaryProceeds,
    totalGrossProceeds,
    underwritingFees,
    
    postIpoSharesOutstanding,
    percentageSold,
    existingHoldersDilution,
    
    expectedFirstDayPop,
    marketCapAtOffer,
    
    convertibleDebtTreatment,
    votingControlAnalysis,
    milestoneWarrantTreatment,
    strategicPartnerTreatment,
    
    // Enhanced multi-instrument analysis
    convertibleAnalysis,
    contingencyAnalysis,
    strategicDealAnalysis,
    employeeOptionAnalysis,
    blendedValuationBreakdown,
    
    warnings,
    assumptions,
  };
}

export async function generateIPOExcel(result: IPOPricingResult): Promise<Buffer> {
  const workbook = new ExcelJS.Workbook();
  workbook.creator = 'ZHI Finance';
  workbook.created = new Date();
  
  // ============ TAB 1: Executive Summary ============
  const summarySheet = workbook.addWorksheet('Executive Summary');
  
  // Header styling
  const headerStyle: Partial<ExcelJS.Style> = {
    font: { bold: true, size: 14, color: { argb: 'FFFFFFFF' } },
    fill: { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF1E3A5F' } },
    alignment: { horizontal: 'center', vertical: 'middle' },
  };
  
  const sectionStyle: Partial<ExcelJS.Style> = {
    font: { bold: true, size: 11 },
    fill: { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE8E8E8' } },
  };
  
  const numberFormat = '#,##0.00';
  const currencyFormat = '"$"#,##0.00';
  const millionsFormat = '"$"#,##0.0"M"';
  const percentFormat = '0.0%';
  
  // Title
  summarySheet.mergeCells('A1:E1');
  summarySheet.getCell('A1').value = `IPO Pricing Analysis - ${result.companyName || 'Company'}`;
  summarySheet.getCell('A1').style = headerStyle;
  summarySheet.getRow(1).height = 30;
  
  summarySheet.getCell('A2').value = `Transaction Date: ${result.transactionDate || 'N/A'}`;
  summarySheet.getCell('A2').font = { italic: true };
  
  // Key Recommendation
  summarySheet.getCell('A4').value = 'IPO PRICING RECOMMENDATION';
  summarySheet.getCell('A4').style = sectionStyle;
  summarySheet.mergeCells('A4:E4');
  
  summarySheet.getCell('A5').value = 'Recommended Offer Price:';
  summarySheet.getCell('B5').value = result.offerPrice;
  summarySheet.getCell('B5').numFmt = currencyFormat;
  summarySheet.getCell('B5').font = { bold: true, size: 16, color: { argb: 'FF006400' } };
  
  // Valuation Metrics Section
  summarySheet.getCell('A7').value = '1. KEY VALUATION METRICS';
  summarySheet.getCell('A7').style = sectionStyle;
  summarySheet.mergeCells('A7:E7');
  
  const valuationData = [
    ['Implied Pre-Money Valuation (at offer)', result.impliedPreMoneyAtOffer || 0, 'Offer Price × Pre-IPO Shares'],
    ['Implied Post-Money Valuation', result.postMoneyValuation || 0, 'Pre-Money + Primary Raise'],
    ['Theoretical Pre-Money (undiscounted)', result.preMoneyValuation || 0, 'LTM Revenue × Multiple'],
    ['Theoretical Price per Share', result.theoreticalPrice || 0, 'Theoretical Pre-Money / Pre-IPO Shares'],
    ['IPO Discount Applied', result.assumptions?.ipoDiscount || 0.15, 'Market discount for successful execution'],
    ['Final Offer Price', result.offerPrice || 0, 'Theoretical × (1 - Discount)'],
  ];
  
  let row = 8;
  valuationData.forEach(([label, value, note]) => {
    summarySheet.getCell(`A${row}`).value = label as string;
    summarySheet.getCell(`B${row}`).value = value as number;
    if (typeof value === 'number') {
      if ((label as string).includes('Discount')) {
        summarySheet.getCell(`B${row}`).numFmt = percentFormat;
      } else if ((label as string).includes('Price')) {
        summarySheet.getCell(`B${row}`).numFmt = currencyFormat;
      } else {
        summarySheet.getCell(`B${row}`).numFmt = millionsFormat;
      }
    }
    summarySheet.getCell(`C${row}`).value = note as string;
    summarySheet.getCell(`C${row}`).font = { italic: true, color: { argb: 'FF666666' } };
    row++;
  });
  
  // Offering Structure Section
  row += 1;
  summarySheet.getCell(`A${row}`).value = '2. OFFERING STRUCTURE';
  summarySheet.getCell(`A${row}`).style = sectionStyle;
  summarySheet.mergeCells(`A${row}:E${row}`);
  row++;
  
  const offeringData = [
    ['Primary Shares to Issue', (result.newSharesIssued || 0) * 1000000, `${(result.newSharesIssued || 0).toFixed(4)}M shares`],
    ['Secondary Shares (existing holders)', (result.secondarySharesSold || 0) * 1000000, `${(result.secondarySharesSold || 0).toFixed(4)}M shares`],
    ['Greenshoe (Over-allotment)', (result.greenshoeShares || 0) * 1000000, `${((result.assumptions?.greenshoePercent || 0.15) * 100).toFixed(0)}% of base offering`],
    ['Total Shares Offered', (result.totalSharesOffered || 0) * 1000000, 'Primary + Secondary + Greenshoe'],
    ['Post-IPO Shares Outstanding', (result.postIpoSharesOutstanding || 0) * 1000000, 'Pre-IPO + Primary Shares'],
    ['Percentage of Company Sold', (result.percentageSold || 0) / 100, 'New Shares / Post-IPO Outstanding'],
    ['Existing Holders Dilution', (result.existingHoldersDilution || 0) / 100, '1 - (Pre-IPO / Post-IPO)'],
  ];
  
  offeringData.forEach(([label, value, note]) => {
    summarySheet.getCell(`A${row}`).value = label as string;
    summarySheet.getCell(`B${row}`).value = value as number;
    if ((label as string).includes('Percentage') || (label as string).includes('Dilution')) {
      summarySheet.getCell(`B${row}`).numFmt = percentFormat;
    } else {
      summarySheet.getCell(`B${row}`).numFmt = '#,##0';
    }
    summarySheet.getCell(`C${row}`).value = note as string;
    summarySheet.getCell(`C${row}`).font = { italic: true, color: { argb: 'FF666666' } };
    row++;
  });
  
  // Proceeds Section
  row += 1;
  summarySheet.getCell(`A${row}`).value = '3. PROCEEDS ANALYSIS';
  summarySheet.getCell(`A${row}`).style = sectionStyle;
  summarySheet.mergeCells(`A${row}:E${row}`);
  row++;
  
  const proceedsData = [
    ['Gross Primary Proceeds', result.grossPrimaryProceeds || 0],
    ['Underwriting Fees', result.underwritingFees || 0],
    ['Net Primary Proceeds to Company', result.netPrimaryProceeds || 0],
    ['Secondary Proceeds (to sellers)', result.secondaryProceeds || 0],
    ['Total Gross Proceeds', result.totalGrossProceeds || 0],
  ];
  
  proceedsData.forEach(([label, value]) => {
    summarySheet.getCell(`A${row}`).value = label as string;
    summarySheet.getCell(`B${row}`).value = value as number;
    summarySheet.getCell(`B${row}`).numFmt = millionsFormat;
    row++;
  });
  
  // Trading Metrics Section
  row += 1;
  summarySheet.getCell(`A${row}`).value = '4. TRADING METRICS';
  summarySheet.getCell(`A${row}`).style = sectionStyle;
  summarySheet.mergeCells(`A${row}:E${row}`);
  row++;
  
  const tradingData = [
    ['Market Cap at Offer', result.marketCapAtOffer || 0],
    ['Expected First-Day Pop', (result.expectedFirstDayPop || 0) / 100],
  ];
  
  tradingData.forEach(([label, value]) => {
    summarySheet.getCell(`A${row}`).value = label as string;
    summarySheet.getCell(`B${row}`).value = value as number;
    if ((label as string).includes('Pop')) {
      summarySheet.getCell(`B${row}`).numFmt = percentFormat;
    } else {
      summarySheet.getCell(`B${row}`).numFmt = millionsFormat;
    }
    row++;
  });
  
  // Convertible Debt Section (if applicable)
  if (result.convertibleDebtTreatment) {
    row += 1;
    summarySheet.getCell(`A${row}`).value = '5. CONVERTIBLE DEBT TREATMENT';
    summarySheet.getCell(`A${row}`).style = sectionStyle;
    summarySheet.mergeCells(`A${row}:E${row}`);
    row++;
    
    const cdt = result.convertibleDebtTreatment;
    const convertibleData: [string, string | number, string?][] = [
      ['Trigger Price', cdt.triggerPrice || 0, 'per share'],
      ['Debt Amount', cdt.debtAmount || 0, 'millions'],
      ['Conversion Shares', cdt.conversionShares || 0, 'millions'],
      ['Conversion Activated', cdt.conversionActivated ? 'YES' : 'NO'],
      ['Original Pre-IPO Shares', cdt.originalPreIpoShares || 0, 'millions'],
      ['Adjusted Pre-IPO Shares', cdt.adjustedPreIpoShares || 0, 'millions'],
      ['Tentative Offer Price (pre-conversion)', cdt.tentativeOfferPrice || 0, 'per share'],
    ];
    
    convertibleData.forEach(([label, value, unit]) => {
      summarySheet.getCell(`A${row}`).value = label as string;
      summarySheet.getCell(`B${row}`).value = value;
      if (typeof value === 'number') {
        if ((unit as string)?.includes('share')) {
          summarySheet.getCell(`B${row}`).numFmt = currencyFormat;
        } else if ((unit as string)?.includes('millions')) {
          summarySheet.getCell(`B${row}`).numFmt = millionsFormat;
        }
      }
      if (value === 'YES') {
        summarySheet.getCell(`B${row}`).font = { bold: true, color: { argb: 'FF006400' } };
      } else if (value === 'NO') {
        summarySheet.getCell(`B${row}`).font = { color: { argb: 'FF666666' } };
      }
      if (unit) {
        summarySheet.getCell(`C${row}`).value = unit as string;
        summarySheet.getCell(`C${row}`).font = { italic: true, color: { argb: 'FF666666' } };
      }
      row++;
    });
  }
  
  // Voting Control Analysis Section (if applicable)
  if (result.votingControlAnalysis) {
    row += 1;
    const sectionNum = result.convertibleDebtTreatment ? '6' : '5';
    summarySheet.getCell(`A${row}`).value = `${sectionNum}. GOVERNANCE - DUAL-CLASS VOTING ANALYSIS`;
    summarySheet.getCell(`A${row}`).style = sectionStyle;
    summarySheet.mergeCells(`A${row}:E${row}`);
    row++;
    
    const vca = result.votingControlAnalysis;
    const votingData: [string, string | number, string?][] = [
      ['Founder Shares', vca.founderSharesMillions, 'millions'],
      ['Founder Vote Multiplier', `${vca.founderVoteMultiplier}x`, ''],
      ['Founder Total Votes', vca.founderVotes, 'millions'],
      ['Public Shares', (result.postIpoSharesOutstanding - vca.founderSharesMillions), 'millions'],
      ['Public Total Votes (1x)', vca.publicVotes, 'millions'],
      ['Total Votes', vca.totalVotes, 'millions'],
      ['Founder Voting Power', vca.founderVotingPower, ''],
      ['Required Control Threshold', vca.controlThreshold, ''],
      ['Voting Control Status', vca.controlSecured ? 'SECURED' : 'BREACHED', ''],
    ];
    
    if (!vca.controlSecured && vca.votingPowerShortfall) {
      votingData.push(['Voting Power Shortfall', vca.votingPowerShortfall, '']);
    }
    
    votingData.forEach(([label, value, unit]) => {
      summarySheet.getCell(`A${row}`).value = label as string;
      summarySheet.getCell(`B${row}`).value = value;
      
      if (typeof value === 'number') {
        if ((unit as string) === 'millions') {
          summarySheet.getCell(`B${row}`).numFmt = '#,##0.00"M"';
        } else if ((label as string).includes('Power') || (label as string).includes('Threshold') || (label as string).includes('Shortfall')) {
          summarySheet.getCell(`B${row}`).numFmt = '0.0%';
        }
      }
      
      // Status styling
      if (value === 'SECURED') {
        summarySheet.getCell(`B${row}`).font = { bold: true, color: { argb: 'FF006400' } };
      } else if (value === 'BREACHED') {
        summarySheet.getCell(`B${row}`).font = { bold: true, color: { argb: 'FFCC0000' } };
      }
      
      if (unit && (unit as string) !== '') {
        summarySheet.getCell(`C${row}`).value = unit as string;
        summarySheet.getCell(`C${row}`).font = { italic: true, color: { argb: 'FF666666' } };
      }
      row++;
    });
  }
  
  // Milestone Warrant Treatment Section (if applicable)
  if (result.milestoneWarrantTreatment) {
    row += 1;
    let sectionNum = '5';
    if (result.convertibleDebtTreatment) sectionNum = '6';
    if (result.convertibleDebtTreatment && result.votingControlAnalysis) sectionNum = '7';
    else if (result.votingControlAnalysis) sectionNum = '6';
    
    summarySheet.getCell(`A${row}`).value = `${sectionNum}. MILESTONE WARRANTS (CONTINGENT DILUTION)`;
    summarySheet.getCell(`A${row}`).style = sectionStyle;
    summarySheet.mergeCells(`A${row}:E${row}`);
    row++;
    
    const mwt = result.milestoneWarrantTreatment;
    const warrantData: [string, string | number, string?][] = [
      ['Warrant Shares', mwt.warrantSharesMillions, 'millions'],
      ['Strike Price', mwt.warrantStrikePrice, 'per share'],
      ['Milestone Probability', mwt.milestoneProbability, ''],
      ['', '', ''],
      ['VALUATION IMPACT:', '', ''],
      ['Original Pre-Money Valuation', mwt.originalPreMoneyValuation, 'millions'],
      ['Theoretical Price (Pre-Warrant)', mwt.theoreticalPriceBeforeAdjustment, 'per share'],
      ['Warrant Status', mwt.warrantInTheMoney ? 'IN-THE-MONEY' : 'OUT-OF-THE-MONEY', ''],
      ['Expected Dilution Cost', mwt.expectedDilutionCost, 'millions'],
      ['Adjusted Pre-Money Valuation', mwt.adjustedPreMoneyValuation, 'millions'],
    ];
    
    warrantData.forEach(([label, value, unit]) => {
      if (label === '') {
        row++;
        return;
      }
      
      // Sub-section header styling
      if ((label as string) === 'VALUATION IMPACT:') {
        summarySheet.getCell(`A${row}`).value = label as string;
        summarySheet.getCell(`A${row}`).font = { bold: true, italic: true };
        row++;
        return;
      }
      
      summarySheet.getCell(`A${row}`).value = label as string;
      summarySheet.getCell(`B${row}`).value = value;
      
      if (typeof value === 'number') {
        if ((unit as string) === 'millions') {
          summarySheet.getCell(`B${row}`).numFmt = '"$"#,##0.00"M"';
        } else if ((unit as string) === 'per share') {
          summarySheet.getCell(`B${row}`).numFmt = '"$"#,##0.00';
        } else if ((label as string).includes('Probability')) {
          summarySheet.getCell(`B${row}`).numFmt = '0.0%';
        }
      }
      
      // Status styling
      if (value === 'IN-THE-MONEY') {
        summarySheet.getCell(`B${row}`).font = { bold: true, color: { argb: 'FFCC0000' } };
      } else if (value === 'OUT-OF-THE-MONEY') {
        summarySheet.getCell(`B${row}`).font = { color: { argb: 'FF006400' } };
      }
      
      if (unit && (unit as string) !== '') {
        summarySheet.getCell(`C${row}`).value = unit as string;
        summarySheet.getCell(`C${row}`).font = { italic: true, color: { argb: 'FF666666' } };
      }
      row++;
    });
  }
  
  // Strategic Partner Block Allocation Section (if applicable)
  if (result.strategicPartnerTreatment) {
    row += 1;
    let sectionNum = '5';
    if (result.convertibleDebtTreatment) sectionNum = '6';
    if (result.votingControlAnalysis) sectionNum = String(parseInt(sectionNum) + 1);
    if (result.milestoneWarrantTreatment) sectionNum = String(parseInt(sectionNum) + 1);
    
    summarySheet.getCell(`A${row}`).value = `${sectionNum}. STRATEGIC PARTNER BLOCK ALLOCATION`;
    summarySheet.getCell(`A${row}`).style = sectionStyle;
    summarySheet.mergeCells(`A${row}:E${row}`);
    row++;
    
    const spt = result.strategicPartnerTreatment;
    const partnerData: [string, string | number, string?][] = [
      ['Partner Name', spt.partnerName, ''],
      ['Guaranteed Shares', spt.partnerSharesMillions, 'millions'],
      ['% of Expected Float', spt.partnerPercentageOfFloat, ''],
      ['', '', ''],
      ['PRICING IMPACT:', '', ''],
      ['Confidence Multiplier', `${spt.confidenceMultiplier.toFixed(4)}x`, ''],
      ['Original Theoretical Price', spt.originalTheoreticalPrice, 'per share'],
      ['Boosted Theoretical Price', spt.boostedTheoreticalPrice, 'per share'],
      ['Original Offer Price', spt.originalOfferPrice, 'per share'],
      ['Boosted Offer Price', spt.boostedOfferPrice, 'per share'],
      ['Price Impact', spt.priceImpact, 'per share'],
      ['Price Impact %', spt.priceImpactPercent / 100, ''],
    ];
    
    partnerData.forEach(([label, value, unit]) => {
      if (label === '') {
        row++;
        return;
      }
      
      // Sub-section header styling
      if ((label as string) === 'PRICING IMPACT:') {
        summarySheet.getCell(`A${row}`).value = label as string;
        summarySheet.getCell(`A${row}`).font = { bold: true, italic: true };
        row++;
        return;
      }
      
      summarySheet.getCell(`A${row}`).value = label as string;
      summarySheet.getCell(`B${row}`).value = value;
      
      if (typeof value === 'number') {
        if ((unit as string) === 'millions') {
          summarySheet.getCell(`B${row}`).numFmt = '#,##0.000"M"';
        } else if ((unit as string) === 'per share') {
          summarySheet.getCell(`B${row}`).numFmt = '"$"#,##0.00';
          // Highlight price impact in green
          if ((label as string).includes('Impact') && !((label as string).includes('%'))) {
            summarySheet.getCell(`B${row}`).font = { bold: true, color: { argb: 'FF006400' } };
            summarySheet.getCell(`B${row}`).numFmt = '"+$"#,##0.00';
          }
        } else if ((label as string).includes('%') || (label as string).includes('Float')) {
          summarySheet.getCell(`B${row}`).numFmt = '0.0%';
          if ((label as string).includes('Impact')) {
            summarySheet.getCell(`B${row}`).font = { bold: true, color: { argb: 'FF006400' } };
            summarySheet.getCell(`B${row}`).numFmt = '"+0.0%"';
          }
        }
      }
      
      if (unit && (unit as string) !== '') {
        summarySheet.getCell(`C${row}`).value = unit as string;
        summarySheet.getCell(`C${row}`).font = { italic: true, color: { argb: 'FF666666' } };
      }
      row++;
    });
  }
  
  // Warnings
  if (result.warnings && result.warnings.length > 0) {
    row += 1;
    summarySheet.getCell(`A${row}`).value = 'WARNINGS';
    summarySheet.getCell(`A${row}`).font = { bold: true, color: { argb: 'FFCC0000' } };
    row++;
    result.warnings.forEach(warning => {
      summarySheet.getCell(`A${row}`).value = warning;
      summarySheet.getCell(`A${row}`).font = { color: { argb: 'FFCC0000' } };
      row++;
    });
  }
  
  // Column widths
  summarySheet.getColumn('A').width = 35;
  summarySheet.getColumn('B').width = 20;
  summarySheet.getColumn('C').width = 45;
  
  // ============ TAB 2: Assumptions ============
  const assumptionsSheet = workbook.addWorksheet('Assumptions');
  
  assumptionsSheet.mergeCells('A1:C1');
  assumptionsSheet.getCell('A1').value = 'IPO ASSUMPTIONS';
  assumptionsSheet.getCell('A1').style = headerStyle;
  assumptionsSheet.getRow(1).height = 25;
  
  const assumptions = result.assumptions || {};
  const assumptionsData = [
    ['Company Name', assumptions.companyName || 'N/A'],
    ['Transaction Date', assumptions.transactionDate || 'N/A'],
    [''],
    ['FINANCIAL METRICS'],
    ['LTM Revenue', assumptions.ltmRevenue || 0, 'millions'],
    ['LTM EBITDA', assumptions.ltmEbitda || 'N/A', 'millions'],
    [''],
    ['VALUATION'],
    ['Valuation Method', assumptions.valuationMethod || 'revenue'],
    ['Industry Revenue Multiple', `${assumptions.industryRevenueMultiple || 0}x`],
    ['Industry EBITDA Multiple', assumptions.industryEbitdaMultiple ? `${assumptions.industryEbitdaMultiple}x` : 'N/A'],
    ['Blend Weight (if blended)', assumptions.blendWeight || 0.5],
    [''],
    ['SHARE STRUCTURE'],
    ['Pre-IPO Shares Outstanding', assumptions.preIpoShares || 0, 'millions'],
    [''],
    ['OFFERING TERMS'],
    ['Primary Raise Target', assumptions.primaryRaiseTarget || 0, 'millions'],
    ['Secondary Shares', assumptions.secondaryShares || 0, 'millions'],
    ['IPO Discount', ((assumptions.ipoDiscount || 0.15) * 100).toFixed(1) + '%'],
    ['Greenshoe (Over-allotment)', ((assumptions.greenshoePercent || 0.15) * 100).toFixed(1) + '%'],
    ['Underwriting Fee', ((assumptions.underwritingFeePercent || 0.07) * 100).toFixed(1) + '%'],
    [''],
    ['CONVERTIBLE DEBT'],
    ['Convertible Debt Amount', assumptions.convertibleDebtAmount || 'N/A', assumptions.convertibleDebtAmount ? 'millions' : ''],
    ['Conversion Trigger Price', assumptions.conversionTriggerPrice ? `$${assumptions.conversionTriggerPrice.toFixed(2)}` : 'N/A'],
    ['Conversion Shares', assumptions.conversionShares || 'N/A', assumptions.conversionShares ? 'millions' : ''],
    [''],
    ['MILESTONE WARRANTS'],
    ['Warrant Shares', assumptions.warrantSharesMillions || 'N/A', assumptions.warrantSharesMillions ? 'millions' : ''],
    ['Warrant Strike Price', assumptions.warrantStrikePrice ? `$${assumptions.warrantStrikePrice.toFixed(2)}` : 'N/A'],
    ['Milestone Probability', assumptions.milestoneProbability ? `${(assumptions.milestoneProbability * 100).toFixed(0)}%` : 'N/A'],
    [''],
    ['STRATEGIC PARTNER'],
    ['Partner Name', assumptions.strategicPartnerName || 'N/A'],
    ['Partner Shares', assumptions.strategicPartnerSharesMillions || 'N/A', assumptions.strategicPartnerSharesMillions ? 'millions' : ''],
  ];
  
  row = 3;
  assumptionsData.forEach(([label, value, unit]) => {
    if (label === '') {
      row++;
      return;
    }
    if (['FINANCIAL METRICS', 'VALUATION', 'SHARE STRUCTURE', 'OFFERING TERMS', 'CONVERTIBLE DEBT', 'MILESTONE WARRANTS', 'STRATEGIC PARTNER'].includes(label as string)) {
      assumptionsSheet.getCell(`A${row}`).style = sectionStyle;
    }
    assumptionsSheet.getCell(`A${row}`).value = label as string;
    assumptionsSheet.getCell(`B${row}`).value = value;
    if (unit) {
      assumptionsSheet.getCell(`C${row}`).value = unit as string;
      assumptionsSheet.getCell(`C${row}`).font = { italic: true, color: { argb: 'FF666666' } };
    }
    row++;
  });
  
  // Add enhanced convertibles if present
  if (result.convertibleAnalysis && result.convertibleAnalysis.instruments.length > 0) {
    row += 1;
    assumptionsSheet.getCell(`A${row}`).value = 'ENHANCED CONVERTIBLE INSTRUMENTS';
    assumptionsSheet.getCell(`A${row}`).style = sectionStyle;
    row++;
    
    result.convertibleAnalysis.instruments.forEach((inst, idx) => {
      assumptionsSheet.getCell(`A${row}`).value = `  ${idx + 1}. ${inst.name}`;
      assumptionsSheet.getCell(`A${row}`).font = { bold: true };
      row++;
      assumptionsSheet.getCell(`A${row}`).value = `     Type: ${inst.type}`;
      assumptionsSheet.getCell(`B${row}`).value = `Amount: $${inst.amountMillions.toFixed(2)}M`;
      row++;
      assumptionsSheet.getCell(`A${row}`).value = `     Status:`;
      assumptionsSheet.getCell(`B${row}`).value = inst.triggered ? 'CONVERTED' : 'Not Converted';
      assumptionsSheet.getCell(`B${row}`).font = { color: { argb: inst.triggered ? 'FF008800' : 'FF888888' } };
      row++;
      if (inst.triggered) {
        assumptionsSheet.getCell(`A${row}`).value = `     Conversion Price:`;
        assumptionsSheet.getCell(`B${row}`).value = `$${inst.conversionPrice.toFixed(2)}`;
        row++;
        assumptionsSheet.getCell(`A${row}`).value = `     Shares Issued:`;
        assumptionsSheet.getCell(`B${row}`).value = `${inst.sharesIssued.toFixed(3)}M`;
        row++;
      }
      if (inst.probability < 1.0) {
        assumptionsSheet.getCell(`A${row}`).value = `     Probability:`;
        assumptionsSheet.getCell(`B${row}`).value = `${(inst.probability * 100).toFixed(0)}%`;
        row++;
        assumptionsSheet.getCell(`A${row}`).value = `     Expected Shares:`;
        assumptionsSheet.getCell(`B${row}`).value = `${inst.expectedShares.toFixed(3)}M`;
        row++;
      }
    });
    
    row++;
    assumptionsSheet.getCell(`A${row}`).value = 'Total Expected Conversion Shares:';
    assumptionsSheet.getCell(`A${row}`).font = { bold: true };
    assumptionsSheet.getCell(`B${row}`).value = `${result.convertibleAnalysis.totalExpectedShares.toFixed(3)}M`;
    assumptionsSheet.getCell(`B${row}`).font = { bold: true };
    row++;
    assumptionsSheet.getCell(`A${row}`).value = 'Total Convertible Amount:';
    assumptionsSheet.getCell(`A${row}`).font = { bold: true };
    assumptionsSheet.getCell(`B${row}`).value = `$${result.convertibleAnalysis.totalConvertibleAmount.toFixed(2)}M`;
    row++;
  }
  
  // Add contingency analysis if present
  if (result.contingencyAnalysis && result.contingencyAnalysis.liabilities.length > 0) {
    row += 1;
    assumptionsSheet.getCell(`A${row}`).value = 'CONTINGENT INSTRUMENTS';
    assumptionsSheet.getCell(`A${row}`).style = sectionStyle;
    row++;
    
    result.contingencyAnalysis.liabilities.forEach((cont: { name: string; type: string; sharesMillions?: number; paymentMillions?: number; probability: number; expectedShares: number; expectedCostMillions: number; }, idx: number) => {
      assumptionsSheet.getCell(`A${row}`).value = `  ${idx + 1}. ${cont.name}`;
      assumptionsSheet.getCell(`A${row}`).font = { bold: true };
      row++;
      assumptionsSheet.getCell(`A${row}`).value = `     Type: ${cont.type}`;
      assumptionsSheet.getCell(`B${row}`).value = `Probability: ${(cont.probability * 100).toFixed(0)}%`;
      row++;
      if (cont.sharesMillions) {
        assumptionsSheet.getCell(`A${row}`).value = `     Expected Shares:`;
        assumptionsSheet.getCell(`B${row}`).value = `${cont.expectedShares?.toFixed(3) || 0}M`;
        row++;
      }
      if (cont.paymentMillions) {
        assumptionsSheet.getCell(`A${row}`).value = `     Expected Cost:`;
        assumptionsSheet.getCell(`B${row}`).value = `$${cont.expectedCostMillions?.toFixed(2) || 0}M`;
        row++;
      }
    });
    
    row++;
    assumptionsSheet.getCell(`A${row}`).value = 'Total Expected Contingent Shares:';
    assumptionsSheet.getCell(`A${row}`).font = { bold: true };
    assumptionsSheet.getCell(`B${row}`).value = `${result.contingencyAnalysis.totalExpectedShares.toFixed(3)}M`;
    row++;
    assumptionsSheet.getCell(`A${row}`).value = 'Total Expected Contingent Cost:';
    assumptionsSheet.getCell(`A${row}`).font = { bold: true };
    assumptionsSheet.getCell(`B${row}`).value = `$${result.contingencyAnalysis.totalExpectedCostMillions.toFixed(2)}M`;
    row++;
  }
  
  // Add blended valuation breakdown if present
  if (result.blendedValuationBreakdown && result.blendedValuationBreakdown.components.length > 0) {
    row += 1;
    assumptionsSheet.getCell(`A${row}`).value = 'BLENDED VALUATION BREAKDOWN';
    assumptionsSheet.getCell(`A${row}`).style = sectionStyle;
    row++;
    
    result.blendedValuationBreakdown.components.forEach((comp) => {
      assumptionsSheet.getCell(`A${row}`).value = `  ${comp.name}`;
      assumptionsSheet.getCell(`B${row}`).value = `${comp.multiple.toFixed(1)}x × ${(comp.weight * 100).toFixed(0)}% = ${comp.weightedMultiple?.toFixed(2) || (comp.multiple * comp.weight).toFixed(2)}x`;
      row++;
    });
    
    row++;
    assumptionsSheet.getCell(`A${row}`).value = 'Base Blended Multiple:';
    assumptionsSheet.getCell(`A${row}`).font = { bold: true };
    assumptionsSheet.getCell(`B${row}`).value = `${result.blendedValuationBreakdown.baseBlendedMultiple?.toFixed(2) || 'N/A'}x`;
    row++;
    
    if (result.blendedValuationBreakdown.growthPremiumApplied) {
      assumptionsSheet.getCell(`A${row}`).value = 'Growth Premium Applied:';
      assumptionsSheet.getCell(`B${row}`).value = `+${result.blendedValuationBreakdown.growthPremiumPercent?.toFixed(0) || 0}%`;
      assumptionsSheet.getCell(`B${row}`).font = { color: { argb: 'FF008800' } };
      row++;
    }
    
    assumptionsSheet.getCell(`A${row}`).value = 'Effective Multiple:';
    assumptionsSheet.getCell(`A${row}`).font = { bold: true };
    assumptionsSheet.getCell(`B${row}`).value = `${result.blendedValuationBreakdown.effectiveMultiple?.toFixed(2) || 'N/A'}x`;
    assumptionsSheet.getCell(`B${row}`).font = { bold: true };
    row++;
    
    assumptionsSheet.getCell(`A${row}`).value = 'Blended Valuation:';
    assumptionsSheet.getCell(`A${row}`).font = { bold: true };
    assumptionsSheet.getCell(`B${row}`).value = `$${result.blendedValuationBreakdown.totalBlendedValuation?.toFixed(2) || 0}M`;
    assumptionsSheet.getCell(`B${row}`).font = { bold: true };
    row++;
  }
  
  assumptionsSheet.getColumn('A').width = 35;
  assumptionsSheet.getColumn('B').width = 25;
  assumptionsSheet.getColumn('C').width = 15;
  
  // ============ TAB 3: Calculation Walkthrough ============
  const calcSheet = workbook.addWorksheet('Calculation Steps');
  
  calcSheet.mergeCells('A1:D1');
  calcSheet.getCell('A1').value = 'IPO PRICING CALCULATION WALKTHROUGH';
  calcSheet.getCell('A1').style = headerStyle;
  calcSheet.getRow(1).height = 25;
  
  // FIX: Corrected calculation steps to show discount applied to enterprise valuation first
  const discountPercent = ((assumptions.ipoDiscount || 0.15) * 100).toFixed(0);
  const discountedValuation = (result.preMoneyValuation || 0) * (1 - (assumptions.ipoDiscount || 0.15));
  
  const calcSteps = [
    ['Step', 'Formula', 'Calculation', 'Result'],
    ['1. Pre-Money Valuation', 'LTM Revenue × Multiple', `$${assumptions.ltmRevenue || 0}M × ${assumptions.industryRevenueMultiple || 0}x`, `$${(result.preMoneyValuation || 0).toFixed(2)}M`],
    ['2. Apply IPO Discount', 'Pre-Money × (1 - Discount)', `$${(result.preMoneyValuation || 0).toFixed(2)}M × (1 - ${discountPercent}%)`, `$${discountedValuation.toFixed(2)}M`],
    ['3. Theoretical Price', 'Pre-Money / Pre-IPO Shares', `$${(result.preMoneyValuation || 0).toFixed(2)}M / ${assumptions.preIpoShares || 0}M`, `$${(result.theoreticalPrice || 0).toFixed(2)} (undiscounted)`],
    ['4. Offer Price', 'Discounted Pre-Money / Shares', `$${discountedValuation.toFixed(2)}M / ${assumptions.preIpoShares || 0}M`, `$${(result.offerPrice || 0).toFixed(2)}`],
    ['5. New Shares to Issue', 'Primary Raise / Offer Price', `$${assumptions.primaryRaiseTarget || 0}M / $${(result.offerPrice || 0).toFixed(2)}`, `${((result.newSharesIssued || 0) * 1000000).toLocaleString()} shares`],
    ['6. Post-Money Valuation', '(Price × Pre-IPO) + Raise', `($${(result.offerPrice || 0).toFixed(2)} × ${assumptions.preIpoShares || 0}M) + $${assumptions.primaryRaiseTarget || 0}M`, `$${(result.postMoneyValuation || 0).toFixed(2)}M`],
  ];
  
  row = 3;
  calcSteps.forEach((stepRow, idx) => {
    stepRow.forEach((cell, colIdx) => {
      const cellRef = calcSheet.getCell(row, colIdx + 1);
      cellRef.value = cell;
      if (idx === 0) {
        cellRef.font = { bold: true };
        cellRef.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE8E8E8' } };
      }
    });
    row++;
  });
  
  calcSheet.getColumn('A').width = 25;
  calcSheet.getColumn('B').width = 30;
  calcSheet.getColumn('C').width = 40;
  calcSheet.getColumn('D').width = 25;
  
  // Generate buffer
  const buffer = await workbook.xlsx.writeBuffer();
  return Buffer.from(buffer);
}
