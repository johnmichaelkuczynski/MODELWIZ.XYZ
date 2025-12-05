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
  "blendWeight": number or 0.5 (weight for revenue in blended, default 0.5)
}

Default values if not specified:
- ipoDiscount: 0.15-0.25 (15-25% is standard, use 0.20 if not specified)
- greenshoePercent: 0.15 (15% standard over-allotment)
- underwritingFeePercent: 0.07 (7% standard for IPOs)
- secondaryShares: 0 (no secondary if not mentioned)
- valuationMethod: "revenue" (unless EBITDA multiple is the focus)
- convertibleDebtAmount, conversionTriggerPrice, conversionShares: null if no convertible debt mentioned

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
  
  // Apply defaults
  return {
    companyName: parsed.companyName || 'Target Company',
    transactionDate: parsed.transactionDate || new Date().toISOString().split('T')[0],
    ltmRevenue: parsed.ltmRevenue,
    ltmEbitda: parsed.ltmEbitda || undefined,
    industryRevenueMultiple: parsed.industryRevenueMultiple,
    industryEbitdaMultiple: parsed.industryEbitdaMultiple || undefined,
    preIpoShares: parsed.preIpoShares,
    primaryRaiseTarget: parsed.primaryRaiseTarget,
    ipoDiscount: parsed.ipoDiscount || 0.20,
    secondaryShares: parsed.secondaryShares || 0,
    greenshoePercent: parsed.greenshoePercent ?? 0.15,
    underwritingFeePercent: parsed.underwritingFeePercent ?? 0.07,
    convertibleDebtAmount: parsed.convertibleDebtAmount || undefined,
    conversionTriggerPrice: parsed.conversionTriggerPrice || undefined,
    conversionShares: parsed.conversionShares || undefined,
    valuationMethod: parsed.valuationMethod || 'revenue',
    blendWeight: parsed.blendWeight ?? 0.5,
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

  // ============ PHASE 2: Handle Convertible Debt ============
  // Calculate initial theoretical price WITHOUT conversion to check trigger
  const initialTheoreticalPrice = preMoneyValuation / originalPreIpoShares;
  const tentativeOfferPrice = initialTheoreticalPrice * (1 - ipoDiscount);
  
  let adjustedPreIpoShares = originalPreIpoShares;
  let conversionActivated = false;
  
  // Check if we have convertible debt and if it triggers
  if (convertibleDebtAmount && conversionTriggerPrice && conversionShares) {
    console.log(`[IPO Model] ============ CONVERTIBLE DEBT CHECK ============`);
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

  // ============ PHASE 3: Calculate Final Theoretical & Offer Price ============
  // Use adjusted share count (may be same as original if no conversion)
  const theoreticalPrice = preMoneyValuation / adjustedPreIpoShares;
  console.log(`[IPO Model] Theoretical Price: $${preMoneyValuation}M / ${adjustedPreIpoShares}M shares = $${theoreticalPrice.toFixed(2)}/share`);

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
  ];
  
  row = 3;
  assumptionsData.forEach(([label, value, unit]) => {
    if (label === '') {
      row++;
      return;
    }
    if (['FINANCIAL METRICS', 'VALUATION', 'SHARE STRUCTURE', 'OFFERING TERMS', 'CONVERTIBLE DEBT'].includes(label as string)) {
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
