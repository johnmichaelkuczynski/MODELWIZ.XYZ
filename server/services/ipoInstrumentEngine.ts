/**
 * IPO Instrument Engine
 * 
 * Processes complex conditional instruments for IPO pricing:
 * - Multiple convertible instruments with different triggers
 * - Probability-weighted contingent liabilities
 * - Strategic deals with premiums/discounts
 * - Anchor orders for demand boost
 * - Employee option dilution (treasury stock method)
 * - Multi-proxy blended valuations
 */

import {
  ConvertibleInstrument,
  ContingentLiability,
  StrategicDeal,
  AnchorOrder,
  EmployeeOptionPool,
  ValuationMultiple,
  IPOAssumptions,
} from './ipoModelService';

// ============ ENGINE RESULT TYPES ============

export interface ConvertibleResult {
  name: string;
  type: string;
  amountMillions: number;
  conversionPrice: number;
  sharesIssued: number;
  triggered: boolean;
  probability: number;
  expectedShares: number;
}

export interface ContingencyResult {
  name: string;
  type: string;
  sharesMillions?: number;
  paymentMillions?: number;
  probability: number;
  expectedShares: number;
  expectedCostMillions: number;
}

export interface StrategicDealResult {
  partnerName: string;
  sharesMillions: number;
  priceType: string;
  effectivePrice: number;
  premiumOrDiscount: number;
  demandBoostPercent: number;
}

export interface InstrumentEngineResult {
  // Adjusted values
  adjustedPreMoneyValuation: number;
  adjustedShareCount: number;
  demandBoostMultiplier: number;
  
  // Convertibles breakdown
  convertibleResults: ConvertibleResult[];
  totalDeterministicConversionShares: number;
  totalExpectedConversionShares: number;
  
  // Contingencies breakdown
  contingencyResults: ContingencyResult[];
  totalExpectedContingencyShares: number;
  totalExpectedContingencyCost: number;
  
  // Strategic deals breakdown
  strategicDealResults: StrategicDealResult[];
  totalAnchorAmount: number;
  
  // Employee options
  employeeOptionDilution: number;
  
  // Blended valuation breakdown
  blendedValuationComponents: Array<{
    name: string;
    type: string;
    multiple: number;
    weight: number;
    contribution: number;
  }>;
  
  // Logs for debugging
  logs: string[];
}

// ============ CONDITION EVALUATOR ============

export function evaluateCondition(
  condition: string,
  ipoPrice: number,
  revenue?: number
): boolean {
  if (!condition) return true;
  
  const conditionLower = condition.toLowerCase().replace(/\s+/g, '');
  
  // Price conditions
  if (conditionLower.includes('ipo>')) {
    const threshold = parseFloat(conditionLower.split('ipo>')[1]);
    return ipoPrice > threshold;
  }
  if (conditionLower.includes('ipo>=')) {
    const threshold = parseFloat(conditionLower.split('ipo>=')[1]);
    return ipoPrice >= threshold;
  }
  if (conditionLower.includes('price>')) {
    const threshold = parseFloat(conditionLower.split('price>')[1]);
    return ipoPrice > threshold;
  }
  
  // Revenue conditions
  if (conditionLower.includes('revenue>') && revenue !== undefined) {
    const threshold = parseFloat(conditionLower.split('revenue>')[1]);
    return revenue > threshold;
  }
  
  // External triggers (use probability only)
  if (conditionLower.includes('fda') || 
      conditionLower.includes('approval') ||
      conditionLower.includes('milestone') ||
      conditionLower.includes('phase')) {
    return true; // Let probability handle it
  }
  
  return true; // Default to true, let probability handle uncertainty
}

// ============ BLENDED VALUATION CALCULATOR ============

export function calculateBlendedValuation(
  ltmRevenue: number,
  ltmEbitda: number | undefined,
  multiples: ValuationMultiple[],
  logs: string[]
): { valuation: number; components: Array<{ name: string; type: string; multiple: number; weight: number; contribution: number }> } {
  if (!multiples || multiples.length === 0) {
    logs.push('[Engine] No valuation multiples provided, returning 0');
    return { valuation: 0, components: [] };
  }
  
  let totalValuation = 0;
  const components: Array<{ name: string; type: string; multiple: number; weight: number; contribution: number }> = [];
  
  for (const mult of multiples) {
    let baseValue = 0;
    if (mult.type === 'revenue') {
      baseValue = ltmRevenue;
    } else if (mult.type === 'ebitda' && ltmEbitda !== undefined) {
      baseValue = Math.max(0, ltmEbitda); // Use 0 if negative
    }
    
    const contribution = baseValue * mult.multiple * mult.weight;
    totalValuation += contribution;
    
    components.push({
      name: mult.name,
      type: mult.type,
      multiple: mult.multiple,
      weight: mult.weight,
      contribution
    });
    
    logs.push(`[Engine] Valuation Component: ${mult.name} = $${baseValue}M × ${mult.multiple}x × ${(mult.weight * 100).toFixed(0)}% = $${contribution.toFixed(2)}M`);
  }
  
  logs.push(`[Engine] Total Blended Valuation: $${totalValuation.toFixed(2)}M`);
  return { valuation: totalValuation, components };
}

// ============ CONVERTIBLE PROCESSOR ============

export function processConvertibles(
  convertibles: ConvertibleInstrument[],
  tentativeIpoPrice: number,
  logs: string[]
): { results: ConvertibleResult[]; deterministicShares: number; expectedShares: number } {
  if (!convertibles || convertibles.length === 0) {
    return { results: [], deterministicShares: 0, expectedShares: 0 };
  }
  
  logs.push(`[Engine] ============ PROCESSING ${convertibles.length} CONVERTIBLE INSTRUMENTS ============`);
  
  const results: ConvertibleResult[] = [];
  let deterministicShares = 0;
  let expectedShares = 0;
  
  for (const conv of convertibles) {
    let conversionPrice = 0;
    let sharesIssued = 0;
    let triggered = false;
    const probability = conv.probability ?? 1.0;
    
    switch (conv.triggerType) {
      case 'lower_of':
        // Convert at lower of fixed price or % of IPO
        const price1 = conv.triggerPrice ?? Infinity;
        const price2 = conv.triggerMultiplier 
          ? tentativeIpoPrice * conv.triggerMultiplier 
          : (conv.triggerPrice2 ?? Infinity);
        conversionPrice = Math.min(price1, price2);
        sharesIssued = conv.amountMillions / conversionPrice;
        triggered = true;
        logs.push(`[Engine] ${conv.name}: lower_of($${price1.toFixed(2)}, ${conv.triggerMultiplier ? (conv.triggerMultiplier * 100).toFixed(0) + '%×IPO=' : ''}$${price2.toFixed(2)}) = $${conversionPrice.toFixed(2)} → ${sharesIssued.toFixed(3)}M shares`);
        break;
        
      case 'price_gt':
      case 'price_gte':
        // Converts if IPO price exceeds threshold
        const threshold = conv.triggerPrice ?? 0;
        const conditionMet = conv.triggerCondition 
          ? evaluateCondition(conv.triggerCondition, tentativeIpoPrice)
          : (conv.triggerType === 'price_gt' ? tentativeIpoPrice > threshold : tentativeIpoPrice >= threshold);
        
        if (conditionMet) {
          conversionPrice = conv.triggerPrice ?? tentativeIpoPrice;
          sharesIssued = conv.amountMillions / conversionPrice;
          triggered = true;
          logs.push(`[Engine] ${conv.name}: price condition MET (IPO $${tentativeIpoPrice.toFixed(2)} vs threshold $${threshold.toFixed(2)}) → ${sharesIssued.toFixed(3)}M shares at $${conversionPrice.toFixed(2)}`);
        } else {
          logs.push(`[Engine] ${conv.name}: price condition NOT MET (IPO $${tentativeIpoPrice.toFixed(2)} vs threshold $${threshold.toFixed(2)}) → NO CONVERSION`);
        }
        break;
        
      case 'at_ipo_price':
        // Converts at IPO price
        conversionPrice = tentativeIpoPrice;
        sharesIssued = conv.amountMillions / conversionPrice;
        triggered = true;
        logs.push(`[Engine] ${conv.name}: converts at IPO price $${tentativeIpoPrice.toFixed(2)} → ${sharesIssued.toFixed(3)}M shares`);
        break;
        
      case 'fixed_shares':
        // Fixed number of shares
        conversionPrice = conv.amountMillions / (conv.fixedShares ?? 1);
        sharesIssued = conv.fixedShares ?? 0;
        triggered = true;
        logs.push(`[Engine] ${conv.name}: fixed ${sharesIssued.toFixed(3)}M shares (implied price $${conversionPrice.toFixed(2)})`);
        break;
        
      case 'conditional':
        // Use probability weighting
        if (conv.fixedShares) {
          sharesIssued = conv.fixedShares;
          conversionPrice = conv.amountMillions / sharesIssued;
        } else {
          conversionPrice = conv.triggerPrice ?? tentativeIpoPrice;
          sharesIssued = conv.amountMillions / conversionPrice;
        }
        triggered = probability > 0;
        logs.push(`[Engine] ${conv.name}: conditional (${(probability * 100).toFixed(0)}% probability) → ${sharesIssued.toFixed(3)}M shares expected`);
        break;
    }
    
    const expectedSharesForThis = sharesIssued * probability;
    
    results.push({
      name: conv.name,
      type: conv.type,
      amountMillions: conv.amountMillions,
      conversionPrice,
      sharesIssued,
      triggered,
      probability,
      expectedShares: expectedSharesForThis
    });
    
    if (triggered && probability >= 1.0) {
      deterministicShares += sharesIssued;
    }
    expectedShares += expectedSharesForThis;
  }
  
  logs.push(`[Engine] Convertibles Total: ${deterministicShares.toFixed(3)}M deterministic + ${(expectedShares - deterministicShares).toFixed(3)}M expected = ${expectedShares.toFixed(3)}M total expected shares`);
  
  return { results, deterministicShares, expectedShares };
}

// ============ CONTINGENCY PROCESSOR ============

export function processContingencies(
  contingencies: ContingentLiability[],
  tentativeIpoPrice: number,
  logs: string[]
): { results: ContingencyResult[]; expectedShares: number; expectedCost: number } {
  if (!contingencies || contingencies.length === 0) {
    return { results: [], expectedShares: 0, expectedCost: 0 };
  }
  
  logs.push(`[Engine] ============ PROCESSING ${contingencies.length} CONTINGENT LIABILITIES ============`);
  
  const results: ContingencyResult[] = [];
  let totalExpectedShares = 0;
  let totalExpectedCost = 0;
  
  for (const cont of contingencies) {
    let expectedShares = 0;
    let expectedCost = 0;
    
    switch (cont.type) {
      case 'earnout':
      case 'grant':
      case 'milestone':
        // Share issuance contingency
        if (cont.sharesMillions) {
          expectedShares = cont.sharesMillions * cont.probability;
          // Cost = fair value of expected shares
          expectedCost = expectedShares * tentativeIpoPrice;
        }
        logs.push(`[Engine] ${cont.name} (${cont.type}): ${cont.sharesMillions?.toFixed(3) ?? 0}M shares × ${(cont.probability * 100).toFixed(0)}% = ${expectedShares.toFixed(3)}M expected shares`);
        break;
        
      case 'warrant':
        // Warrant with strike price
        if (cont.sharesMillions && cont.strikePrice !== undefined) {
          expectedShares = cont.sharesMillions * cont.probability;
          const spread = Math.max(0, tentativeIpoPrice - cont.strikePrice);
          expectedCost = spread * cont.sharesMillions * cont.probability;
          logs.push(`[Engine] ${cont.name} (warrant): ${cont.sharesMillions.toFixed(3)}M @ $${cont.strikePrice.toFixed(2)} strike × ${(cont.probability * 100).toFixed(0)}% = ${expectedShares.toFixed(3)}M expected, $${expectedCost.toFixed(2)}M cost`);
        }
        break;
        
      case 'litigation':
      case 'royalty':
        // Cash payment contingency
        if (cont.paymentMillions) {
          expectedCost = cont.paymentMillions * cont.probability;
          logs.push(`[Engine] ${cont.name} (${cont.type}): $${cont.paymentMillions.toFixed(2)}M × ${(cont.probability * 100).toFixed(0)}% = $${expectedCost.toFixed(2)}M expected cost`);
        }
        break;
    }
    
    results.push({
      name: cont.name,
      type: cont.type,
      sharesMillions: cont.sharesMillions,
      paymentMillions: cont.paymentMillions,
      probability: cont.probability,
      expectedShares,
      expectedCostMillions: expectedCost
    });
    
    totalExpectedShares += expectedShares;
    totalExpectedCost += expectedCost;
  }
  
  logs.push(`[Engine] Contingencies Total: ${totalExpectedShares.toFixed(3)}M expected shares, $${totalExpectedCost.toFixed(2)}M expected cost`);
  
  return { results, expectedShares: totalExpectedShares, expectedCost: totalExpectedCost };
}

// ============ STRATEGIC DEAL PROCESSOR ============

export function processStrategicDeals(
  deals: StrategicDeal[],
  anchorOrders: AnchorOrder[],
  tentativeIpoPrice: number,
  primaryRaiseTarget: number,
  logs: string[]
): { results: StrategicDealResult[]; demandBoostMultiplier: number; totalAnchorAmount: number } {
  logs.push(`[Engine] ============ PROCESSING STRATEGIC DEALS & ANCHORS ============`);
  
  const results: StrategicDealResult[] = [];
  let totalAnchorAmount = 0;
  let demandBoostMultiplier = 1.0;
  
  // Process strategic deals
  if (deals && deals.length > 0) {
    for (const deal of deals) {
      let effectivePrice = tentativeIpoPrice;
      let premiumOrDiscount = 0;
      
      switch (deal.priceType) {
        case 'ipo_premium':
          premiumOrDiscount = deal.pricePremium ?? 0;
          effectivePrice = tentativeIpoPrice * (1 + premiumOrDiscount);
          break;
        case 'discounted':
          premiumOrDiscount = -(deal.priceDiscount ?? 0);
          effectivePrice = tentativeIpoPrice * (1 - (deal.priceDiscount ?? 0));
          break;
        case 'fixed':
          effectivePrice = deal.fixedPrice ?? tentativeIpoPrice;
          premiumOrDiscount = (effectivePrice - tentativeIpoPrice) / tentativeIpoPrice;
          break;
        case 'ipo_price':
        default:
          effectivePrice = tentativeIpoPrice;
          break;
      }
      
      // Calculate demand boost from this deal
      const dealAmount = deal.sharesMillions * effectivePrice;
      let demandBoost = 0;
      
      if (deal.isAnchorOrder) {
        totalAnchorAmount += dealAmount;
        demandBoost = (dealAmount / primaryRaiseTarget) * 0.2; // Max 20% boost for full anchor
      }
      
      results.push({
        partnerName: deal.partnerName,
        sharesMillions: deal.sharesMillions,
        priceType: deal.priceType,
        effectivePrice,
        premiumOrDiscount,
        demandBoostPercent: demandBoost * 100
      });
      
      logs.push(`[Engine] ${deal.partnerName}: ${deal.sharesMillions.toFixed(3)}M shares @ $${effectivePrice.toFixed(2)} (${premiumOrDiscount >= 0 ? '+' : ''}${(premiumOrDiscount * 100).toFixed(1)}%)`);
    }
  }
  
  // Process anchor orders
  if (anchorOrders && anchorOrders.length > 0) {
    for (const anchor of anchorOrders) {
      totalAnchorAmount += anchor.amountMillions;
      
      let effectivePrice = tentativeIpoPrice;
      if (anchor.priceType === 'ipo_premium') {
        effectivePrice = tentativeIpoPrice * (1 + (anchor.pricePremium ?? 0));
      } else if (anchor.priceType === 'ipo_discount') {
        effectivePrice = tentativeIpoPrice * (1 - (anchor.priceDiscount ?? 0));
      }
      
      const demandBoost = (anchor.amountMillions / primaryRaiseTarget) * 0.2;
      
      results.push({
        partnerName: anchor.investorName,
        sharesMillions: anchor.amountMillions / effectivePrice,
        priceType: anchor.priceType,
        effectivePrice,
        premiumOrDiscount: anchor.pricePremium ?? -(anchor.priceDiscount ?? 0),
        demandBoostPercent: demandBoost * 100
      });
      
      logs.push(`[Engine] Anchor: ${anchor.investorName}: $${anchor.amountMillions.toFixed(2)}M → ${(demandBoost * 100).toFixed(1)}% demand boost`);
    }
  }
  
  // Calculate total demand boost
  if (totalAnchorAmount > 0) {
    const anchorPercent = totalAnchorAmount / primaryRaiseTarget;
    demandBoostMultiplier = 1.0 + Math.min(anchorPercent * 0.2, 0.20); // Cap at 20%
    logs.push(`[Engine] Total Anchor: $${totalAnchorAmount.toFixed(2)}M = ${(anchorPercent * 100).toFixed(1)}% of raise → ${((demandBoostMultiplier - 1) * 100).toFixed(1)}% valuation boost`);
  }
  
  return { results, demandBoostMultiplier, totalAnchorAmount };
}

// ============ EMPLOYEE OPTION PROCESSOR ============

export function processEmployeeOptions(
  options: EmployeeOptionPool | undefined,
  offerPrice: number,
  logs: string[]
): number {
  if (!options || options.sharesMillions <= 0) {
    return 0;
  }
  
  logs.push(`[Engine] ============ PROCESSING EMPLOYEE OPTIONS ============`);
  
  // Treasury Stock Method:
  // If strike < offer price, options are "in-the-money"
  // Net dilution = options - (options × strike / offer price)
  
  const vestedPercent = options.vestedPercent ?? 1.0;
  const vestedOptions = options.sharesMillions * vestedPercent;
  
  if (offerPrice <= options.avgStrikePrice) {
    logs.push(`[Engine] Options OUT-OF-THE-MONEY: Strike $${options.avgStrikePrice.toFixed(2)} >= Offer $${offerPrice.toFixed(2)} → No dilution`);
    return 0;
  }
  
  // In-the-money: calculate treasury stock method dilution
  const proceedsFromExercise = vestedOptions * options.avgStrikePrice;
  const sharesBoughtBack = proceedsFromExercise / offerPrice;
  const netDilution = vestedOptions - sharesBoughtBack;
  
  logs.push(`[Engine] Options IN-THE-MONEY: ${vestedOptions.toFixed(3)}M vested @ $${options.avgStrikePrice.toFixed(2)} strike`);
  logs.push(`[Engine] Treasury Stock: ${vestedOptions.toFixed(3)}M exercised - ${sharesBoughtBack.toFixed(3)}M bought back = ${netDilution.toFixed(3)}M net dilution`);
  
  return netDilution;
}

// ============ MAIN ENGINE FUNCTION ============

export function runInstrumentEngine(
  assumptions: IPOAssumptions,
  basePreMoneyValuation: number,
  tentativeTheoreticalPrice: number,
  tentativeOfferPrice: number
): InstrumentEngineResult {
  const logs: string[] = [];
  logs.push(`[Engine] ============ IPO INSTRUMENT ENGINE START ============`);
  logs.push(`[Engine] Base Pre-Money: $${basePreMoneyValuation.toFixed(2)}M`);
  logs.push(`[Engine] Tentative Theoretical: $${tentativeTheoreticalPrice.toFixed(2)}`);
  logs.push(`[Engine] Tentative Offer: $${tentativeOfferPrice.toFixed(2)}`);
  
  let adjustedValuation = basePreMoneyValuation;
  let adjustedShares = assumptions.preIpoShares;
  
  // Step 1: Calculate blended valuation if multiple proxies provided
  let blendedComponents: Array<{ name: string; type: string; multiple: number; weight: number; contribution: number }> = [];
  if (assumptions.valuationMultiples && assumptions.valuationMultiples.length > 0) {
    const blendedResult = calculateBlendedValuation(
      assumptions.ltmRevenue,
      assumptions.ltmEbitda,
      assumptions.valuationMultiples,
      logs
    );
    adjustedValuation = blendedResult.valuation;
    blendedComponents = blendedResult.components;
  }
  
  // Step 2: Apply anchor order demand boost
  const dealResults = processStrategicDeals(
    assumptions.strategicDeals ?? [],
    assumptions.anchorOrders ?? [],
    tentativeOfferPrice,
    assumptions.primaryRaiseTarget,
    logs
  );
  
  if (dealResults.demandBoostMultiplier > 1.0) {
    const boostedValuation = adjustedValuation * dealResults.demandBoostMultiplier;
    logs.push(`[Engine] Applying demand boost: $${adjustedValuation.toFixed(2)}M × ${dealResults.demandBoostMultiplier.toFixed(4)} = $${boostedValuation.toFixed(2)}M`);
    adjustedValuation = boostedValuation;
  }
  
  // Recalculate tentative price with new valuation
  const adjustedTheoreticalPrice = adjustedValuation / adjustedShares;
  const adjustedOfferPrice = adjustedTheoreticalPrice * (1 - assumptions.ipoDiscount);
  
  // Step 3: Process convertibles
  const convertibleResults = processConvertibles(
    assumptions.convertibles ?? [],
    adjustedOfferPrice,
    logs
  );
  
  adjustedShares += convertibleResults.expectedShares;
  
  // Step 4: Process contingencies
  const contingencyResults = processContingencies(
    assumptions.contingencies ?? [],
    adjustedOfferPrice,
    logs
  );
  
  adjustedShares += contingencyResults.expectedShares;
  adjustedValuation -= contingencyResults.expectedCost;
  
  // Step 5: Process employee options
  const employeeOptionDilution = processEmployeeOptions(
    assumptions.employeeOptions,
    adjustedOfferPrice,
    logs
  );
  
  adjustedShares += employeeOptionDilution;
  
  logs.push(`[Engine] ============ ENGINE SUMMARY ============`);
  logs.push(`[Engine] Final Adjusted Pre-Money: $${adjustedValuation.toFixed(2)}M`);
  logs.push(`[Engine] Final Adjusted Shares: ${adjustedShares.toFixed(3)}M`);
  logs.push(`[Engine] Demand Boost: ${((dealResults.demandBoostMultiplier - 1) * 100).toFixed(1)}%`);
  
  return {
    adjustedPreMoneyValuation: adjustedValuation,
    adjustedShareCount: adjustedShares,
    demandBoostMultiplier: dealResults.demandBoostMultiplier,
    
    convertibleResults: convertibleResults.results,
    totalDeterministicConversionShares: convertibleResults.deterministicShares,
    totalExpectedConversionShares: convertibleResults.expectedShares,
    
    contingencyResults: contingencyResults.results,
    totalExpectedContingencyShares: contingencyResults.expectedShares,
    totalExpectedContingencyCost: contingencyResults.expectedCost,
    
    strategicDealResults: dealResults.results,
    totalAnchorAmount: dealResults.totalAnchorAmount,
    
    employeeOptionDilution,
    
    blendedValuationComponents: blendedComponents,
    
    logs
  };
}
