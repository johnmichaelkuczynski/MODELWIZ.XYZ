import ExcelJS from 'exceljs';
import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';

export type FinanceLLMProvider = 'zhi1' | 'zhi2' | 'zhi3' | 'zhi4' | 'zhi5';

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
  
  // Strategic Partner Block Allocation (Optional)
  strategicPartnerSharesMillions?: number;  // Shares guaranteed to strategic partner (in millions)
  strategicPartnerName?: string;            // Name of strategic partner
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
  "preIpoShares": number (Pre-IPO Fully Diluted Shares in millions),
  "primaryRaiseTarget": number (Primary Cash to Raise in millions),
  "ipoDiscount": number (as decimal, e.g., 0.20 for 20% discount),
  
  "secondaryShares": number or 0 (Secondary shares in millions, 0 if primary only),
  "greenshoePercent": number (as decimal, default 0.15 for 15% over-allotment),
  "underwritingFeePercent": number (as decimal, default 0.07 for 7% fee),
  
  "convertibleDebtAmount": number or null (Convertible debt amount in millions, null if none),
  "conversionTriggerPrice": number or null (Price per share that triggers conversion, null if no convertible debt),
  "conversionShares": number or null (Shares the debt converts into in millions, null if no convertible debt),
  
  "valuationMethod": "revenue" or "ebitda" or "blended" (default "revenue"),
  "blendWeight": number or 0.5 (weight for revenue in blended, default 0.5),
  
  "founderSharesMillions": number or null (Founder shares with super-voting rights in millions, null if no dual-class),
  "founderVoteMultiplier": number or null (Votes per founder share, e.g., 10 for 10x voting, null if no dual-class),
  "controlThreshold": number or null (Minimum voting % founders require as decimal, e.g., 0.40 for 40%, null if not specified),
  
  "warrantSharesMillions": number or null (Shares to be issued if milestone is hit in millions, null if no warrants),
  "warrantStrikePrice": number or null (Price at which warrant shares can be purchased, null if no warrants),
  "milestoneProbability": number or null (Probability of milestone being hit as decimal 0.0-1.0, null if no warrants),
  
  "strategicPartnerSharesMillions": number or null (Shares guaranteed/allocated to strategic partner in millions, null if none),
  "strategicPartnerName": string or null (Name of strategic partner/investor, null if none)
}

Default values if not specified:
- ipoDiscount: 0.15-0.25 (15-25% is standard, use 0.20 if not specified)
- greenshoePercent: 0.15 (15% standard over-allotment)
- underwritingFeePercent: 0.07 (7% standard for IPOs)
- secondaryShares: 0 (no secondary if not mentioned)
- valuationMethod: "revenue" (unless EBITDA multiple is the focus)
- convertibleDebtAmount, conversionTriggerPrice, conversionShares: null if no convertible debt mentioned
- founderSharesMillions, founderVoteMultiplier: null if no dual-class share structure mentioned
- controlThreshold: 0.50 (50%) if dual-class is mentioned but no specific threshold given
- warrantSharesMillions, warrantStrikePrice, milestoneProbability: null if no milestone warrants mentioned
- strategicPartnerSharesMillions, strategicPartnerName: null if no strategic partner allocation mentioned

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
      max_tokens: 2000,
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
      max_tokens: 2000,
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
      max_tokens: 2000,
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
        max_tokens: 2000,
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
      max_tokens: 2000,
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
    convertibleDebtAmount: normalizeToMillions(parsed.convertibleDebtAmount, 'convertibleDebtAmount'),
    conversionTriggerPrice: parsed.conversionTriggerPrice || undefined,
    conversionShares: normalizeShares(parsed.conversionShares, 'conversionShares'),
    valuationMethod: parsed.valuationMethod || 'revenue',
    blendWeight: parsed.blendWeight ?? 0.5,
    // Dual-class share structure
    founderSharesMillions: normalizeShares(parsed.founderSharesMillions, 'founderSharesMillions'),
    founderVoteMultiplier: parsed.founderVoteMultiplier || undefined,
    controlThreshold: parsed.controlThreshold ?? (parsed.founderSharesMillions ? 0.50 : undefined),
    // Milestone warrants
    warrantSharesMillions: normalizeShares(parsed.warrantSharesMillions, 'warrantSharesMillions'),
    warrantStrikePrice: parsed.warrantStrikePrice || undefined,
    milestoneProbability: parsed.milestoneProbability || undefined,
    // Strategic partner allocation
    strategicPartnerSharesMillions: normalizeShares(parsed.strategicPartnerSharesMillions, 'strategicPartnerSharesMillions'),
    strategicPartnerName: parsed.strategicPartnerName || undefined,
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

  // ============ PHASE 1.5: Handle Convertible Debt FIRST (Adjusts Share Count) ============
  // IMPORTANT: Convertible debt must be processed BEFORE milestone warrants
  // because warrants use the adjusted share count for theoretical price calculation
  
  // Calculate first-pass theoretical price to check conversion trigger
  const firstPassTheoreticalPrice = preMoneyValuation / originalPreIpoShares;
  const tentativeOfferPrice = firstPassTheoreticalPrice * (1 - ipoDiscount);
  
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
  const theoreticalPrice = (preMoneyValuation / adjustedPreIpoShares) * confidenceMultiplier;
  console.log(`[IPO Model] Theoretical Price: $${preMoneyValuation}M / ${adjustedPreIpoShares}M shares${confidenceMultiplier > 1 ? ` × ${confidenceMultiplier.toFixed(4)}x` : ''} = $${theoreticalPrice.toFixed(2)}/share`);

  // Apply Market Discount to get Offer Price
  let offerPrice = theoreticalPrice * (1 - ipoDiscount);
  console.log(`[IPO Model] Offer Price: $${theoreticalPrice.toFixed(2)} × (1 - ${(ipoDiscount*100).toFixed(0)}%) = $${offerPrice.toFixed(2)}/share`);

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
  
  assumptionsSheet.getColumn('A').width = 30;
  assumptionsSheet.getColumn('B').width = 20;
  assumptionsSheet.getColumn('C').width = 15;
  
  // ============ TAB 3: Calculation Walkthrough ============
  const calcSheet = workbook.addWorksheet('Calculation Steps');
  
  calcSheet.mergeCells('A1:D1');
  calcSheet.getCell('A1').value = 'IPO PRICING CALCULATION WALKTHROUGH';
  calcSheet.getCell('A1').style = headerStyle;
  calcSheet.getRow(1).height = 25;
  
  const calcSteps = [
    ['Step', 'Formula', 'Calculation', 'Result'],
    ['1. Pre-Money Valuation', 'LTM Revenue × Multiple', `$${assumptions.ltmRevenue || 0}M × ${assumptions.industryRevenueMultiple || 0}x`, `$${(result.preMoneyValuation || 0).toFixed(2)}M`],
    ['2. Theoretical Price', 'Pre-Money / Pre-IPO Shares', `$${(result.preMoneyValuation || 0).toFixed(2)}M / ${assumptions.preIpoShares || 0}M`, `$${(result.theoreticalPrice || 0).toFixed(2)}`],
    ['3. Apply IPO Discount', 'Theoretical × (1 - Discount)', `$${(result.theoreticalPrice || 0).toFixed(2)} × (1 - ${((assumptions.ipoDiscount || 0.15) * 100).toFixed(0)}%)`, `$${(result.offerPrice || 0).toFixed(2)}`],
    ['4. New Shares to Issue', 'Primary Raise / Offer Price', `$${assumptions.primaryRaiseTarget || 0}M / $${(result.offerPrice || 0).toFixed(2)}`, `${((result.newSharesIssued || 0) * 1000000).toLocaleString()} shares`],
    ['5. Post-Money Valuation', '(Price × Pre-IPO) + Raise', `($${(result.offerPrice || 0).toFixed(2)} × ${assumptions.preIpoShares || 0}M) + $${assumptions.primaryRaiseTarget || 0}M`, `$${(result.postMoneyValuation || 0).toFixed(2)}M`],
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
