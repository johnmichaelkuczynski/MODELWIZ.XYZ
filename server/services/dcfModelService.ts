import ExcelJS from 'exceljs';
import Anthropic from '@anthropic-ai/sdk';

interface DCFAssumptions {
  companyName: string;
  baseYearRevenue: number;
  revenueGrowthRates: number[];
  terminalGrowthRate: number;
  baseEBITDAMargin: number;
  targetEBITDAMargin: number;
  marginExpansionYears: number;
  daPercent: number;
  taxRate: number;
  capexPercent: number;
  nwcPercent: number;
  wacc: number;
  projectionYears: number;
  totalDebt: number;
  cashAndEquivalents: number;
  sharesOutstanding: number;
}

export async function parseFinancialDescription(description: string, customInstructions?: string): Promise<DCFAssumptions> {
  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  const systemPrompt = `You are a financial analyst expert at extracting DCF model assumptions from natural language descriptions.
Extract ALL the following variables from the user's description. If a value is not explicitly stated, use reasonable defaults based on industry standards.

Return a JSON object with EXACTLY these fields (all numbers, no strings except companyName):
{
  "companyName": "string - company name or 'Target Company' if not specified",
  "baseYearRevenue": number in millions (e.g., 850 for $850 million),
  "revenueGrowthRates": array of 5 numbers representing Y1-Y5 growth rates as decimals (e.g., [0.35, 0.35, 0.28, 0.20, 0.20]),
  "terminalGrowthRate": number as decimal (e.g., 0.04 for 4%),
  "baseEBITDAMargin": number as decimal (e.g., 0.18 for 18%),
  "targetEBITDAMargin": number as decimal (e.g., 0.32 for 32%),
  "marginExpansionYears": number of years to reach target margin (usually 5),
  "daPercent": D&A as decimal of revenue (e.g., 0.06 for 6%),
  "taxRate": number as decimal (e.g., 0.21 for 21%),
  "capexPercent": CapEx as decimal of revenue (e.g., 0.08 for 8%),
  "nwcPercent": NWC as decimal of revenue (e.g., 0.12 for 12%),
  "wacc": number as decimal (e.g., 0.115 for 11.5%),
  "projectionYears": number (usually 5),
  "totalDebt": number in millions,
  "cashAndEquivalents": number in millions,
  "sharesOutstanding": number in millions
}

Default values if not specified:
- revenueGrowthRates: [0.10, 0.10, 0.08, 0.06, 0.05] (declining growth)
- terminalGrowthRate: 0.025 (2.5%)
- baseEBITDAMargin: 0.15 (15%)
- targetEBITDAMargin: 0.20 (20%)
- marginExpansionYears: 5
- daPercent: 0.05 (5%)
- taxRate: 0.25 (25%)
- capexPercent: 0.05 (5%)
- nwcPercent: 0.10 (10%)
- wacc: 0.10 (10%)
- projectionYears: 5
- totalDebt: 0
- cashAndEquivalents: 0
- sharesOutstanding: 100

IMPORTANT: Return ONLY valid JSON, no markdown, no explanations.`;

  let userPrompt = `Extract DCF assumptions from this description:\n\n${description}`;
  
  if (customInstructions) {
    userPrompt += `\n\nAdditional instructions: ${customInstructions}`;
  }

  const response = await anthropic.messages.create({
    model: 'claude-3-5-sonnet-20241022',
    max_tokens: 2000,
    temperature: 0,
    system: systemPrompt,
    messages: [{ role: 'user', content: userPrompt }]
  });

  const content = response.content[0];
  if (content.type !== 'text') {
    throw new Error('Unexpected response type from AI');
  }

  try {
    const assumptions = JSON.parse(content.text.trim());
    return assumptions as DCFAssumptions;
  } catch (error) {
    console.error('Failed to parse AI response:', content.text);
    throw new Error('Failed to parse financial assumptions from description');
  }
}

export async function generateDCFExcel(assumptions: DCFAssumptions): Promise<Buffer> {
  const workbook = new ExcelJS.Workbook();
  workbook.creator = 'Finance Panel';
  workbook.created = new Date();

  const {
    companyName,
    baseYearRevenue,
    revenueGrowthRates,
    terminalGrowthRate,
    baseEBITDAMargin,
    targetEBITDAMargin,
    marginExpansionYears,
    daPercent,
    taxRate,
    capexPercent,
    nwcPercent,
    wacc,
    projectionYears,
    totalDebt,
    cashAndEquivalents,
    sharesOutstanding
  } = assumptions;

  const blueFont: Partial<ExcelJS.Font> = { color: { argb: 'FF0000FF' }, bold: true };
  const headerStyle: Partial<ExcelJS.Style> = {
    font: { bold: true, size: 12 },
    fill: { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } },
    alignment: { horizontal: 'center' }
  };
  const currencyFormat = '"$"#,##0';
  const percentFormat = '0.0%';
  const perShareFormat = '"$"#,##0.00';

  const summarySheet = workbook.addWorksheet('Executive Summary');
  summarySheet.columns = [
    { header: '', key: 'label', width: 35 },
    { header: '', key: 'value', width: 20 }
  ];

  summarySheet.getCell('A1').value = `${companyName} - DCF Valuation Model`;
  summarySheet.getCell('A1').font = { bold: true, size: 16 };
  summarySheet.getCell('A2').value = `Valuation Date: ${new Date().toLocaleDateString()}`;
  
  summarySheet.getCell('A4').value = 'VALUATION SUMMARY';
  summarySheet.getCell('A4').font = { bold: true, size: 14 };
  
  summarySheet.getCell('A5').value = 'Enterprise Value:';
  summarySheet.getCell('B5').value = { formula: '=DCF_Valuation!B20' };
  summarySheet.getCell('B5').numFmt = currencyFormat;
  
  summarySheet.getCell('A6').value = 'Less: Net Debt:';
  summarySheet.getCell('B6').value = { formula: '=Assumptions!B32-Assumptions!B33' };
  summarySheet.getCell('B6').numFmt = currencyFormat;
  
  summarySheet.getCell('A7').value = 'Equity Value:';
  summarySheet.getCell('B7').value = { formula: '=DCF_Valuation!B24' };
  summarySheet.getCell('B7').numFmt = currencyFormat;
  summarySheet.getCell('B7').font = { bold: true };
  
  summarySheet.getCell('A8').value = 'Shares Outstanding:';
  summarySheet.getCell('B8').value = { formula: '=Assumptions!B34' };
  summarySheet.getCell('B8').numFmt = '#,##0';
  
  summarySheet.getCell('A9').value = 'Value per Share:';
  summarySheet.getCell('B9').value = { formula: '=DCF_Valuation!B26' };
  summarySheet.getCell('B9').numFmt = perShareFormat;
  summarySheet.getCell('B9').font = { bold: true };
  
  summarySheet.getCell('A11').value = 'KEY ASSUMPTIONS';
  summarySheet.getCell('A11').font = { bold: true, size: 14 };
  
  summarySheet.getCell('A12').value = 'Base Year Revenue:';
  summarySheet.getCell('B12').value = { formula: '=Assumptions!B5' };
  summarySheet.getCell('B12').numFmt = currencyFormat;
  
  summarySheet.getCell('A13').value = 'WACC:';
  summarySheet.getCell('B13').value = { formula: '=Assumptions!B26' };
  summarySheet.getCell('B13').numFmt = percentFormat;
  
  summarySheet.getCell('A14').value = 'Terminal Growth Rate:';
  summarySheet.getCell('B14').value = { formula: '=Assumptions!B12' };
  summarySheet.getCell('B14').numFmt = percentFormat;
  
  summarySheet.getCell('A15').value = 'Terminal EBITDA Margin:';
  summarySheet.getCell('B15').value = { formula: '=Assumptions!B16' };
  summarySheet.getCell('B15').numFmt = percentFormat;

  const assumptionsSheet = workbook.addWorksheet('Assumptions');
  assumptionsSheet.columns = [
    { header: '', key: 'label', width: 40 },
    { header: '', key: 'value', width: 15 }
  ];

  assumptionsSheet.getCell('A1').value = 'REVENUE ASSUMPTIONS';
  assumptionsSheet.getCell('A1').font = { bold: true, size: 14 };
  
  assumptionsSheet.getCell('A5').value = 'Base Year Revenue (Year 0) ($M):';
  assumptionsSheet.getCell('B5').value = baseYearRevenue;
  assumptionsSheet.getCell('B5').font = blueFont;
  assumptionsSheet.getCell('B5').numFmt = currencyFormat;
  
  for (let i = 0; i < 5; i++) {
    assumptionsSheet.getCell(`A${6 + i}`).value = `Year ${i + 1} Growth Rate:`;
    assumptionsSheet.getCell(`B${6 + i}`).value = revenueGrowthRates[i] || 0.10;
    assumptionsSheet.getCell(`B${6 + i}`).font = blueFont;
    assumptionsSheet.getCell(`B${6 + i}`).numFmt = percentFormat;
  }
  
  assumptionsSheet.getCell('A12').value = 'Terminal Growth Rate:';
  assumptionsSheet.getCell('B12').value = terminalGrowthRate;
  assumptionsSheet.getCell('B12').font = blueFont;
  assumptionsSheet.getCell('B12').numFmt = percentFormat;

  assumptionsSheet.getCell('A14').value = 'PROFITABILITY ASSUMPTIONS';
  assumptionsSheet.getCell('A14').font = { bold: true, size: 14 };
  
  assumptionsSheet.getCell('A15').value = 'Base Year EBITDA Margin:';
  assumptionsSheet.getCell('B15').value = baseEBITDAMargin;
  assumptionsSheet.getCell('B15').font = blueFont;
  assumptionsSheet.getCell('B15').numFmt = percentFormat;
  
  assumptionsSheet.getCell('A16').value = 'Target EBITDA Margin:';
  assumptionsSheet.getCell('B16').value = targetEBITDAMargin;
  assumptionsSheet.getCell('B16').font = blueFont;
  assumptionsSheet.getCell('B16').numFmt = percentFormat;
  
  assumptionsSheet.getCell('A17').value = 'Years to Target Margin:';
  assumptionsSheet.getCell('B17').value = marginExpansionYears;
  assumptionsSheet.getCell('B17').font = blueFont;
  
  assumptionsSheet.getCell('A18').value = 'D&A as % of Revenue:';
  assumptionsSheet.getCell('B18').value = daPercent;
  assumptionsSheet.getCell('B18').font = blueFont;
  assumptionsSheet.getCell('B18').numFmt = percentFormat;
  
  assumptionsSheet.getCell('A19').value = 'Tax Rate:';
  assumptionsSheet.getCell('B19').value = taxRate;
  assumptionsSheet.getCell('B19').font = blueFont;
  assumptionsSheet.getCell('B19').numFmt = percentFormat;

  assumptionsSheet.getCell('A21').value = 'OPERATING ASSUMPTIONS';
  assumptionsSheet.getCell('A21').font = { bold: true, size: 14 };
  
  assumptionsSheet.getCell('A22').value = 'CapEx as % of Revenue:';
  assumptionsSheet.getCell('B22').value = capexPercent;
  assumptionsSheet.getCell('B22').font = blueFont;
  assumptionsSheet.getCell('B22').numFmt = percentFormat;
  
  assumptionsSheet.getCell('A23').value = 'NWC as % of Revenue:';
  assumptionsSheet.getCell('B23').value = nwcPercent;
  assumptionsSheet.getCell('B23').font = blueFont;
  assumptionsSheet.getCell('B23').numFmt = percentFormat;

  assumptionsSheet.getCell('A25').value = 'VALUATION PARAMETERS';
  assumptionsSheet.getCell('A25').font = { bold: true, size: 14 };
  
  assumptionsSheet.getCell('A26').value = 'WACC:';
  assumptionsSheet.getCell('B26').value = wacc;
  assumptionsSheet.getCell('B26').font = blueFont;
  assumptionsSheet.getCell('B26').numFmt = percentFormat;
  
  assumptionsSheet.getCell('A27').value = 'Projection Period (Years):';
  assumptionsSheet.getCell('B27').value = projectionYears;
  assumptionsSheet.getCell('B27').font = blueFont;

  assumptionsSheet.getCell('A30').value = 'CAPITAL STRUCTURE';
  assumptionsSheet.getCell('A30').font = { bold: true, size: 14 };
  
  assumptionsSheet.getCell('A32').value = 'Total Debt ($M):';
  assumptionsSheet.getCell('B32').value = totalDebt;
  assumptionsSheet.getCell('B32').font = blueFont;
  assumptionsSheet.getCell('B32').numFmt = currencyFormat;
  
  assumptionsSheet.getCell('A33').value = 'Cash & Equivalents ($M):';
  assumptionsSheet.getCell('B33').value = cashAndEquivalents;
  assumptionsSheet.getCell('B33').font = blueFont;
  assumptionsSheet.getCell('B33').numFmt = currencyFormat;
  
  assumptionsSheet.getCell('A34').value = 'Shares Outstanding (M):';
  assumptionsSheet.getCell('B34').value = sharesOutstanding;
  assumptionsSheet.getCell('B34').font = blueFont;
  assumptionsSheet.getCell('B34').numFmt = '#,##0.0';

  const revenueSheet = workbook.addWorksheet('Revenue_EBITDA');
  const revHeaders = ['', 'Year 0', 'Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'];
  revenueSheet.addRow(revHeaders);
  revenueSheet.getRow(1).font = { bold: true };
  revenueSheet.getRow(1).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  
  revenueSheet.columns = [
    { width: 25 },
    { width: 15 },
    { width: 15 },
    { width: 15 },
    { width: 15 },
    { width: 15 },
    { width: 15 }
  ];

  revenueSheet.addRow(['Revenue Growth Rate', '', 
    { formula: '=Assumptions!B6' },
    { formula: '=Assumptions!B7' },
    { formula: '=Assumptions!B8' },
    { formula: '=Assumptions!B9' },
    { formula: '=Assumptions!B10' }
  ]);
  for (let i = 2; i <= 7; i++) {
    revenueSheet.getCell(2, i).numFmt = percentFormat;
  }

  revenueSheet.addRow(['Revenue ($M)',
    { formula: '=Assumptions!B5' },
    { formula: '=B3*(1+C2)' },
    { formula: '=C3*(1+D2)' },
    { formula: '=D3*(1+E2)' },
    { formula: '=E3*(1+F2)' },
    { formula: '=F3*(1+G2)' }
  ]);
  for (let i = 2; i <= 7; i++) {
    revenueSheet.getCell(3, i).numFmt = currencyFormat;
  }

  revenueSheet.addRow(['EBITDA Margin',
    { formula: '=Assumptions!B15' },
    { formula: '=Assumptions!B15+(Assumptions!B16-Assumptions!B15)*(1/Assumptions!B17)' },
    { formula: '=Assumptions!B15+(Assumptions!B16-Assumptions!B15)*(2/Assumptions!B17)' },
    { formula: '=Assumptions!B15+(Assumptions!B16-Assumptions!B15)*(3/Assumptions!B17)' },
    { formula: '=Assumptions!B15+(Assumptions!B16-Assumptions!B15)*(4/Assumptions!B17)' },
    { formula: '=Assumptions!B16' }
  ]);
  for (let i = 2; i <= 7; i++) {
    revenueSheet.getCell(4, i).numFmt = percentFormat;
  }

  revenueSheet.addRow(['EBITDA ($M)',
    { formula: '=B3*B4' },
    { formula: '=C3*C4' },
    { formula: '=D3*D4' },
    { formula: '=E3*E4' },
    { formula: '=F3*F4' },
    { formula: '=G3*G4' }
  ]);
  for (let i = 2; i <= 7; i++) {
    revenueSheet.getCell(5, i).numFmt = currencyFormat;
  }

  const fcfSheet = workbook.addWorksheet('Free_Cash_Flow');
  const fcfHeaders = ['', 'Year 0', 'Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'];
  fcfSheet.addRow(fcfHeaders);
  fcfSheet.getRow(1).font = { bold: true };
  fcfSheet.getRow(1).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };
  
  fcfSheet.columns = [
    { width: 30 },
    { width: 15 },
    { width: 15 },
    { width: 15 },
    { width: 15 },
    { width: 15 },
    { width: 15 }
  ];

  fcfSheet.addRow(['Revenue ($M)',
    { formula: '=Revenue_EBITDA!B3' },
    { formula: '=Revenue_EBITDA!C3' },
    { formula: '=Revenue_EBITDA!D3' },
    { formula: '=Revenue_EBITDA!E3' },
    { formula: '=Revenue_EBITDA!F3' },
    { formula: '=Revenue_EBITDA!G3' }
  ]);

  fcfSheet.addRow(['EBITDA ($M)',
    { formula: '=Revenue_EBITDA!B5' },
    { formula: '=Revenue_EBITDA!C5' },
    { formula: '=Revenue_EBITDA!D5' },
    { formula: '=Revenue_EBITDA!E5' },
    { formula: '=Revenue_EBITDA!F5' },
    { formula: '=Revenue_EBITDA!G5' }
  ]);

  fcfSheet.addRow(['Less: D&A ($M)',
    { formula: '=-B2*Assumptions!$B$18' },
    { formula: '=-C2*Assumptions!$B$18' },
    { formula: '=-D2*Assumptions!$B$18' },
    { formula: '=-E2*Assumptions!$B$18' },
    { formula: '=-F2*Assumptions!$B$18' },
    { formula: '=-G2*Assumptions!$B$18' }
  ]);

  fcfSheet.addRow(['EBIT ($M)',
    { formula: '=B3+B4' },
    { formula: '=C3+C4' },
    { formula: '=D3+D4' },
    { formula: '=E3+E4' },
    { formula: '=F3+F4' },
    { formula: '=G3+G4' }
  ]);

  fcfSheet.addRow(['Less: Taxes ($M)',
    { formula: '=-B5*Assumptions!$B$19' },
    { formula: '=-C5*Assumptions!$B$19' },
    { formula: '=-D5*Assumptions!$B$19' },
    { formula: '=-E5*Assumptions!$B$19' },
    { formula: '=-F5*Assumptions!$B$19' },
    { formula: '=-G5*Assumptions!$B$19' }
  ]);

  fcfSheet.addRow(['NOPAT ($M)',
    { formula: '=B5+B6' },
    { formula: '=C5+C6' },
    { formula: '=D5+D6' },
    { formula: '=E5+E6' },
    { formula: '=F5+F6' },
    { formula: '=G5+G6' }
  ]);

  fcfSheet.addRow(['Add back: D&A ($M)',
    { formula: '=-B4' },
    { formula: '=-C4' },
    { formula: '=-D4' },
    { formula: '=-E4' },
    { formula: '=-F4' },
    { formula: '=-G4' }
  ]);

  fcfSheet.addRow(['Less: CapEx ($M)',
    { formula: '=-B2*Assumptions!$B$22' },
    { formula: '=-C2*Assumptions!$B$22' },
    { formula: '=-D2*Assumptions!$B$22' },
    { formula: '=-E2*Assumptions!$B$22' },
    { formula: '=-F2*Assumptions!$B$22' },
    { formula: '=-G2*Assumptions!$B$22' }
  ]);

  fcfSheet.addRow(['NWC Balance ($M)',
    { formula: '=B2*Assumptions!$B$23' },
    { formula: '=C2*Assumptions!$B$23' },
    { formula: '=D2*Assumptions!$B$23' },
    { formula: '=E2*Assumptions!$B$23' },
    { formula: '=F2*Assumptions!$B$23' },
    { formula: '=G2*Assumptions!$B$23' }
  ]);

  fcfSheet.addRow(['Less: Change in NWC ($M)',
    '',
    { formula: '=-(C10-B10)' },
    { formula: '=-(D10-C10)' },
    { formula: '=-(E10-D10)' },
    { formula: '=-(F10-E10)' },
    { formula: '=-(G10-F10)' }
  ]);

  fcfSheet.addRow(['Unlevered FCF ($M)',
    { formula: '=B7+B8+B9+B11' },
    { formula: '=C7+C8+C9+C11' },
    { formula: '=D7+D8+D9+D11' },
    { formula: '=E7+E8+E9+E11' },
    { formula: '=F7+F8+F9+F11' },
    { formula: '=G7+G8+G9+G11' }
  ]);
  fcfSheet.getRow(12).font = { bold: true };

  for (let row = 2; row <= 12; row++) {
    for (let col = 2; col <= 7; col++) {
      fcfSheet.getCell(row, col).numFmt = currencyFormat;
    }
  }

  const dcfSheet = workbook.addWorksheet('DCF_Valuation');
  dcfSheet.columns = [
    { width: 35 },
    { width: 18 },
    { width: 18 },
    { width: 18 }
  ];

  dcfSheet.getCell('A1').value = 'PRESENT VALUE OF EXPLICIT PERIOD CASH FLOWS';
  dcfSheet.getCell('A1').font = { bold: true, size: 14 };
  
  dcfSheet.addRow(['Year', 'FCF ($M)', 'Discount Factor', 'Present Value ($M)']);
  dcfSheet.getRow(2).font = { bold: true };
  dcfSheet.getRow(2).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };

  for (let year = 1; year <= 5; year++) {
    const row = dcfSheet.addRow([
      year,
      { formula: `=Free_Cash_Flow!${String.fromCharCode(66 + year)}12` },
      { formula: `=1/(1+Assumptions!$B$26)^${year}` },
      { formula: `=B${2 + year}*C${2 + year}` }
    ]);
    row.getCell(2).numFmt = currencyFormat;
    row.getCell(3).numFmt = '0.0000';
    row.getCell(4).numFmt = currencyFormat;
  }

  dcfSheet.addRow(['Total PV of Years 1-5:', '', '', { formula: '=SUM(D3:D7)' }]);
  dcfSheet.getRow(8).font = { bold: true };
  dcfSheet.getCell('D8').numFmt = currencyFormat;

  dcfSheet.getCell('A10').value = 'TERMINAL VALUE CALCULATION';
  dcfSheet.getCell('A10').font = { bold: true, size: 14 };
  
  dcfSheet.addRow(['Year 6 FCF ($M):', { formula: '=Free_Cash_Flow!G12*(1+Assumptions!B12)' }]);
  dcfSheet.getCell('B11').numFmt = currencyFormat;
  
  dcfSheet.addRow(['Terminal Value ($M):', { formula: '=B11/(Assumptions!B26-Assumptions!B12)' }]);
  dcfSheet.getCell('B12').numFmt = currencyFormat;
  
  dcfSheet.addRow(['Discount Factor (Year 5):', { formula: '=1/(1+Assumptions!B26)^5' }]);
  dcfSheet.getCell('B13').numFmt = '0.0000';
  
  dcfSheet.addRow(['PV of Terminal Value ($M):', { formula: '=B12*B13' }]);
  dcfSheet.getCell('B14').numFmt = currencyFormat;
  dcfSheet.getRow(14).font = { bold: true };

  dcfSheet.getCell('A16').value = 'ENTERPRISE VALUE BRIDGE';
  dcfSheet.getCell('A16').font = { bold: true, size: 14 };
  
  dcfSheet.addRow(['PV of Explicit Period FCFs ($M):', { formula: '=D8' }]);
  dcfSheet.getCell('B17').numFmt = currencyFormat;
  
  dcfSheet.addRow(['Plus: PV of Terminal Value ($M):', { formula: '=B14' }]);
  dcfSheet.getCell('B18').numFmt = currencyFormat;
  
  dcfSheet.addRow(['']);
  
  dcfSheet.addRow(['Enterprise Value ($M):', { formula: '=B17+B18' }]);
  dcfSheet.getCell('B20').numFmt = currencyFormat;
  dcfSheet.getRow(20).font = { bold: true };
  
  dcfSheet.addRow(['Less: Total Debt ($M):', { formula: '=-Assumptions!B32' }]);
  dcfSheet.getCell('B21').numFmt = currencyFormat;
  
  dcfSheet.addRow(['Plus: Cash ($M):', { formula: '=Assumptions!B33' }]);
  dcfSheet.getCell('B22').numFmt = currencyFormat;
  
  dcfSheet.addRow(['']);
  
  dcfSheet.addRow(['Equity Value ($M):', { formula: '=B20+B21+B22' }]);
  dcfSheet.getCell('B24').numFmt = currencyFormat;
  dcfSheet.getRow(24).font = { bold: true };
  
  dcfSheet.addRow(['Shares Outstanding (M):', { formula: '=Assumptions!B34' }]);
  dcfSheet.getCell('B25').numFmt = '#,##0.0';
  
  dcfSheet.addRow(['Value per Share:', { formula: '=B24/B25' }]);
  dcfSheet.getCell('B26').numFmt = perShareFormat;
  dcfSheet.getRow(26).font = { bold: true, size: 14 };
  dcfSheet.getCell('B26').fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFFF00' } };

  const sensSheet = workbook.addWorksheet('Sensitivity_WACC_Growth');
  sensSheet.columns = [
    { width: 15 },
    { width: 12 },
    { width: 12 },
    { width: 12 },
    { width: 12 },
    { width: 12 },
    { width: 12 },
    { width: 12 }
  ];

  sensSheet.getCell('A1').value = 'EQUITY VALUE SENSITIVITY ($M)';
  sensSheet.getCell('A1').font = { bold: true, size: 14 };
  sensSheet.getCell('A2').value = 'WACC vs Terminal Growth Rate';
  
  sensSheet.getCell('B3').value = 'Terminal Growth Rate';
  sensSheet.mergeCells('B3:H3');
  sensSheet.getCell('B3').alignment = { horizontal: 'center' };
  sensSheet.getCell('B3').font = { bold: true };

  const baseWacc = wacc;
  const baseTermGrowth = terminalGrowthRate;
  
  const waccValues = [
    baseWacc - 0.02,
    baseWacc - 0.015,
    baseWacc - 0.01,
    baseWacc - 0.005,
    baseWacc,
    baseWacc + 0.005,
    baseWacc + 0.01,
    baseWacc + 0.015,
    baseWacc + 0.02
  ];
  
  const termGrowthValues = [
    baseTermGrowth - 0.01,
    baseTermGrowth - 0.0075,
    baseTermGrowth - 0.005,
    baseTermGrowth - 0.0025,
    baseTermGrowth,
    baseTermGrowth + 0.0025,
    baseTermGrowth + 0.005
  ];

  sensSheet.addRow(['WACC \\ TG', ...termGrowthValues.map(v => v)]);
  sensSheet.getRow(4).font = { bold: true };
  for (let i = 2; i <= 8; i++) {
    sensSheet.getCell(4, i).numFmt = percentFormat;
  }

  for (let wIdx = 0; wIdx < waccValues.length; wIdx++) {
    const waccVal = waccValues[wIdx];
    const row: any[] = [waccVal];
    
    for (let tIdx = 0; tIdx < termGrowthValues.length; tIdx++) {
      const tgVal = termGrowthValues[tIdx];
      
      const year5Fcf = calculateYear5FCF(assumptions);
      const terminalFcf = year5Fcf * (1 + tgVal);
      const terminalValue = terminalFcf / (waccVal - tgVal);
      const pvTerminal = terminalValue / Math.pow(1 + waccVal, 5);
      
      const pvFcfs = calculatePVOfFCFs(assumptions, waccVal);
      const ev = pvFcfs + pvTerminal;
      const equityValue = ev - totalDebt + cashAndEquivalents;
      
      row.push(equityValue);
    }
    
    const excelRow = sensSheet.addRow(row);
    excelRow.getCell(1).numFmt = percentFormat;
    for (let i = 2; i <= 8; i++) {
      excelRow.getCell(i).numFmt = currencyFormat;
      
      if (wIdx === 4 && i === 6) {
        excelRow.getCell(i).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFFF00' } };
      }
    }
  }

  const scenarioSheet = workbook.addWorksheet('Scenarios');
  scenarioSheet.columns = [
    { width: 30 },
    { width: 18 },
    { width: 18 },
    { width: 18 }
  ];

  scenarioSheet.getCell('A1').value = 'SCENARIO ANALYSIS';
  scenarioSheet.getCell('A1').font = { bold: true, size: 14 };

  scenarioSheet.addRow(['', 'Bear Case', 'Base Case', 'Bull Case']);
  scenarioSheet.getRow(2).font = { bold: true };
  scenarioSheet.getRow(2).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } };

  const bearMultiplier = 0.85;
  const bullMultiplier = 1.15;

  scenarioSheet.addRow(['Revenue Growth (Avg)', 
    (revenueGrowthRates.reduce((a,b) => a+b, 0) / 5) * bearMultiplier,
    revenueGrowthRates.reduce((a,b) => a+b, 0) / 5,
    (revenueGrowthRates.reduce((a,b) => a+b, 0) / 5) * bullMultiplier
  ]);
  for (let i = 2; i <= 4; i++) {
    scenarioSheet.getCell(3, i).numFmt = percentFormat;
  }

  scenarioSheet.addRow(['Terminal EBITDA Margin',
    targetEBITDAMargin * bearMultiplier,
    targetEBITDAMargin,
    Math.min(targetEBITDAMargin * bullMultiplier, 0.45)
  ]);
  for (let i = 2; i <= 4; i++) {
    scenarioSheet.getCell(4, i).numFmt = percentFormat;
  }

  scenarioSheet.addRow(['WACC',
    wacc + 0.015,
    wacc,
    wacc - 0.015
  ]);
  for (let i = 2; i <= 4; i++) {
    scenarioSheet.getCell(5, i).numFmt = percentFormat;
  }

  scenarioSheet.addRow(['Terminal Growth',
    terminalGrowthRate - 0.01,
    terminalGrowthRate,
    terminalGrowthRate + 0.01
  ]);
  for (let i = 2; i <= 4; i++) {
    scenarioSheet.getCell(6, i).numFmt = percentFormat;
  }

  scenarioSheet.addRow(['']);

  const bearEquity = calculateScenarioEquity(assumptions, bearMultiplier, wacc + 0.015, terminalGrowthRate - 0.01);
  const baseEquity = calculateScenarioEquity(assumptions, 1, wacc, terminalGrowthRate);
  const bullEquity = calculateScenarioEquity(assumptions, bullMultiplier, wacc - 0.015, terminalGrowthRate + 0.01);

  scenarioSheet.addRow(['Equity Value ($M)', bearEquity, baseEquity, bullEquity]);
  scenarioSheet.getRow(8).font = { bold: true };
  for (let i = 2; i <= 4; i++) {
    scenarioSheet.getCell(8, i).numFmt = currencyFormat;
  }

  scenarioSheet.addRow(['Value per Share', 
    bearEquity / sharesOutstanding, 
    baseEquity / sharesOutstanding, 
    bullEquity / sharesOutstanding
  ]);
  scenarioSheet.getRow(9).font = { bold: true };
  for (let i = 2; i <= 4; i++) {
    scenarioSheet.getCell(9, i).numFmt = perShareFormat;
  }

  scenarioSheet.getCell('B8').fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFF9999' } };
  scenarioSheet.getCell('C8').fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFFF00' } };
  scenarioSheet.getCell('D8').fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF99FF99' } };
  
  scenarioSheet.getCell('B9').fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFF9999' } };
  scenarioSheet.getCell('C9').fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFFF00' } };
  scenarioSheet.getCell('D9').fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF99FF99' } };

  const buffer = await workbook.xlsx.writeBuffer();
  return Buffer.from(buffer);
}

function calculateYear5FCF(assumptions: DCFAssumptions): number {
  const { baseYearRevenue, revenueGrowthRates, baseEBITDAMargin, targetEBITDAMargin, 
          daPercent, taxRate, capexPercent, nwcPercent } = assumptions;
  
  let revenue = baseYearRevenue;
  let prevNwc = revenue * nwcPercent;
  
  for (let year = 1; year <= 5; year++) {
    revenue = revenue * (1 + revenueGrowthRates[year - 1]);
  }
  
  const margin = targetEBITDAMargin;
  const ebitda = revenue * margin;
  const da = revenue * daPercent;
  const ebit = ebitda - da;
  const taxes = ebit * taxRate;
  const nopat = ebit - taxes;
  const capex = revenue * capexPercent;
  const currentNwc = revenue * nwcPercent;
  
  let prevRevenue = baseYearRevenue;
  for (let year = 1; year <= 4; year++) {
    prevRevenue = prevRevenue * (1 + revenueGrowthRates[year - 1]);
  }
  prevNwc = prevRevenue * nwcPercent;
  
  const changeNwc = currentNwc - prevNwc;
  const fcf = nopat + da - capex - changeNwc;
  
  return fcf;
}

function calculatePVOfFCFs(assumptions: DCFAssumptions, waccVal: number): number {
  const { baseYearRevenue, revenueGrowthRates, baseEBITDAMargin, targetEBITDAMargin,
          marginExpansionYears, daPercent, taxRate, capexPercent, nwcPercent } = assumptions;
  
  let totalPV = 0;
  let prevRevenue = baseYearRevenue;
  let prevNwc = baseYearRevenue * nwcPercent;
  
  for (let year = 1; year <= 5; year++) {
    const revenue = prevRevenue * (1 + revenueGrowthRates[year - 1]);
    const marginProgress = Math.min(year / marginExpansionYears, 1);
    const margin = baseEBITDAMargin + (targetEBITDAMargin - baseEBITDAMargin) * marginProgress;
    
    const ebitda = revenue * margin;
    const da = revenue * daPercent;
    const ebit = ebitda - da;
    const taxes = ebit * taxRate;
    const nopat = ebit - taxes;
    const capex = revenue * capexPercent;
    const currentNwc = revenue * nwcPercent;
    const changeNwc = currentNwc - prevNwc;
    const fcf = nopat + da - capex - changeNwc;
    
    const discountFactor = 1 / Math.pow(1 + waccVal, year);
    totalPV += fcf * discountFactor;
    
    prevRevenue = revenue;
    prevNwc = currentNwc;
  }
  
  return totalPV;
}

function calculateScenarioEquity(
  assumptions: DCFAssumptions, 
  growthMultiplier: number, 
  waccVal: number, 
  termGrowth: number
): number {
  const modifiedAssumptions = {
    ...assumptions,
    revenueGrowthRates: assumptions.revenueGrowthRates.map(r => r * growthMultiplier),
    targetEBITDAMargin: Math.min(assumptions.targetEBITDAMargin * growthMultiplier, 0.45)
  };
  
  const pvFcfs = calculatePVOfFCFs(modifiedAssumptions, waccVal);
  
  let revenue = modifiedAssumptions.baseYearRevenue;
  for (let year = 1; year <= 5; year++) {
    revenue = revenue * (1 + modifiedAssumptions.revenueGrowthRates[year - 1]);
  }
  
  const margin = modifiedAssumptions.targetEBITDAMargin;
  const ebitda = revenue * margin;
  const da = revenue * modifiedAssumptions.daPercent;
  const ebit = ebitda - da;
  const taxes = ebit * modifiedAssumptions.taxRate;
  const nopat = ebit - taxes;
  
  let prevRevenue = modifiedAssumptions.baseYearRevenue;
  for (let year = 1; year <= 4; year++) {
    prevRevenue = prevRevenue * (1 + modifiedAssumptions.revenueGrowthRates[year - 1]);
  }
  
  const capex = revenue * modifiedAssumptions.capexPercent;
  const currentNwc = revenue * modifiedAssumptions.nwcPercent;
  const prevNwc = prevRevenue * modifiedAssumptions.nwcPercent;
  const changeNwc = currentNwc - prevNwc;
  const year5Fcf = nopat + da - capex - changeNwc;
  
  const terminalFcf = year5Fcf * (1 + termGrowth);
  const terminalValue = terminalFcf / (waccVal - termGrowth);
  const pvTerminal = terminalValue / Math.pow(1 + waccVal, 5);
  
  const ev = pvFcfs + pvTerminal;
  const equityValue = ev - assumptions.totalDebt + assumptions.cashAndEquivalents;
  
  return equityValue;
}
