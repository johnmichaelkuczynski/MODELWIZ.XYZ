# Cognitive Analysis Platform

## Overview
This platform analyzes written text to assess the intelligence and cognitive fingerprint of authors using multi-model AI evaluation. It provides document analysis, AI detection, translation, comprehensive cognitive profiling, and intelligent text rewriting capabilities. The project's vision is to offer deep insights into cognitive abilities and thought processes from written content, with advanced features for maximizing intelligence scores through iterative rewriting, and also offers professional financial model generation.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture
The application uses a monorepo structure, separating client and server.
- **Frontend**: React with TypeScript, TailwindCSS, shadcn/ui, wouter for routing, React Query for server state, and Chart.js for data visualization.
- **Backend**: Express.js with TypeScript, integrating multiple LLMs, document processing via Mathpix OCR, speech-to-text with AssemblyAI, and email services via SendGrid.
- **Database**: PostgreSQL with Drizzle ORM, storing user, document, analysis, and cognitive profile data.
- **Core Services**: Includes multi-model intelligence evaluation, document comparison, multi-language translation, OCR for mathematical notation, and intelligent text rewriting with custom instructions support.
    - **Intelligence Evaluation**: A 4-phase system with initial assessment, deep analytical questioning across 17 cognitive dimensions (including Conceptual Depth, Inferential Control, Semantic Compression, Novel Abstraction, Cognitive Risk, Authenticity, Symbolic Manipulation), revision, and final pushback. Supports genre-aware assessment.
    - **Intelligent Rewrite Function (MAXINTEL)**: Recursively optimizes text to maximize intelligence scores, supporting custom instructions and optional external knowledge integration.
    - **GPT Bypass Humanizer**: Transforms 100% AI-generated text to 0% AI detection.
    - **Coherence Meter**: Offers two processing strategies (simple chunking, outline-guided) with adjustable aggressiveness, and a dedicated `math-proof-validity` mode for mathematical correctness verification.
    - **Text Model Validator**: Includes "Truth Select" for truth-value isomorphism and "Math Truth Select" for rigorous first-order model theory and truth-grounded model searching, both supporting literal truth verification. Users can select which AI model powers validation (ZHI 1-5), defaulting to Grok (ZHI 5).
    - **AI Chat Assistant**: Features full conversation history tracking and Zhi Database integration for contextual philosophical content.
    - **Conservative Reconstruction**: "Charitable Interpretation" mode produces self-contained, coherent essays articulating a text's unified argument.
    - **Finance Panel**: Generates professional Excel financial models from natural language input, producing value-based workbooks with executive summaries, assumptions, projections, valuations, and sensitivity analysis.
      - **DCF Model**: Fully implemented with 3 valuation scenarios (Bear/Base/Bull), 5-year projections, and sensitivity analysis.
      - **LBO Model**: Fully implemented with sources/uses, debt schedules, sponsor returns (IRR/MOIC), exit valuation, and 5-year projections.
      - **M&A Model**: Fully implemented with comprehensive fixes (December 2024):
        - **Accretion/Dilution Analysis**: Uses acquirerExplicitEPS if user provides it directly (e.g., "earns $3.20 per share") instead of recalculating.
        - **Sources & Uses**: Now balances exactly. Includes explicit or calculated transaction fees. Cash from balance sheet auto-adjusts to balance.
        - **Goodwill Calculation**: Correct formula: Purchase Price - Fair Value Net Assets - Identified Intangibles.
        - **Purchase Price Allocation (PPA) Tab**: New Excel tab with breakdown of customer relationships, developed technology, other intangibles, and amortization schedules.
        - **Debt Schedule Tab**: New Excel tab with beginning/ending balance, mandatory amortization, and interest expense by year.
        - **Revenue Synergy Margin**: Applies flow-through margin (e.g., 50%) to revenue synergies for EBITDA impact. Default 100% if not specified.
        - **Interest Expense**: Properly calculated from debt schedule and included in pro forma projections.
        - Features separate phase-in schedules for revenue synergies (default: 0/50/100/100/100%) and cost synergies (default: 20/60/100/100/100%).
      - **3-Statement Model**: Fully implemented with comprehensive 11-tab Excel generation:
        - **Income Statement**: Revenue, COGS, Gross Profit, Operating Expenses, EBITDA, D&A, EBIT, Interest, Taxes, Net Income, EPS
        - **Balance Sheet**: Assets (Cash, A/R, Inventory, PP&E), Liabilities (A/P, Debt), Shareholders' Equity with balance check (Assets = L + E)
        - **Cash Flow Statement**: Operating (CFO), Investing (CFI), Financing (CFF) activities with Free Cash Flow
        - **Circular Reference Handling**: Iterative calculation for revolver draws/paydowns and cash balancing
        - **Supporting Schedules**: Debt Schedule, Working Capital, PP&E, Shareholders' Equity
        - **Ratio Analysis**: 50+ ratios across profitability, liquidity, leverage, efficiency, growth, per-share metrics
        - **Charts Data**: Pre-formatted tables for Revenue/EBITDA, Leverage, EPS visualization
        - **Balance Sheet Reconciliation (December 2024 Fix)**: Implements standard financial modeling "plug" mechanism using cash as balancing item. Ensures Assets = Liabilities + Equity across all periods (0-5) by: (1) anchoring historical assets to user input, (2) adjusting retained earnings to force period 0 balance, (3) propagating retained earnings adjustment through forward periods, (4) applying cash plug for any remaining imbalance per period.
      - **IPO Pricing Model**: Fully implemented with comprehensive mechanical pricing (December 2024):
        - **Core Mechanics**: EV = MarketCap + Debt - Cash; Proceeds = OfferPrice × SharesSold
        - **Shares Calculation**: Computed from dollarRaise/offerPrice at each price point
        - **Founder Ownership**: Recomputed at each price point (fixed founder shares / updated FD shares)
        - **Price Range Derivation**: Uses user-provided range, OR computes from (1) fair value per share ±15%, (2) dollar raise / shares, (3) peer EV/EBITDA multiple (for restaurant/consumer), (4) peer EV/Rev multiple, (5) EV/raNPV multiple
        - **POP Calculation**: Purely mechanical from user-provided order book and sector benchmarks
        - **Book Quality**: Uses ln(oversubscription) with no fabricated extrapolation or decay rates
        - **All Defaults Neutral**: downRoundIpoPenalty=0, dualClassDiscount=0, underwritingFeePercent=0, monthsToCatalyst=undefined
        - **No Hardcoded Thresholds**: Removed all judgment thresholds (oversubscription 3/5/10, POP -0.10)
        - **No CEO Text Inference**: pricingAggressiveness and managementPriority must be explicitly provided
        - **Recommendation Logic**: Based purely on user's pricingAggressiveness and managementPriority
        - **Peer-First Pricing with DCF Ceiling (December 2024)**:
          - Peer multiples are the BASE CASE anchor (not DCF)
          - DCF serves as CEILING only (caps peer if lower)
          - Post-money equity calculation with iterative convergence (5 iterations, delta < $0.01)
          - IPO discount: 87.5% of anchor for ~14% expected Day-1 pop
        - **Greenshoe Exercise Flag (December 2024)**:
          - `greenshoeAssumedExercised` boolean (default: true)
          - When true: greenshoe included in dilution, proceeds, cash, EV
          - When false: base primary only, no greenshoe in any metric
          - Ensures internal consistency across all calculations
        - **Greenshoe Proceeds Separation**:
          - basePrimaryProceedsM: Exact target (e.g., $550M)
          - greenshoeProceedsM: Additive on top (e.g., ~$82.5M for 15%)
          - grossProceedsM = base + greenshoe + secondary
        - **Restaurant/Consumer Sector Support (December 2024)**: 
          - EV/EBITDA as primary valuation multiple (not EV/Revenue)
          - Hard guard prevents EV/Revenue fallback for restaurant/consumer sectors
          - ntmEBITDA auto-calculated from revenue × margin when not provided directly
          - Sector-specific WACC defaults: restaurant 8.5%, consumer staples 8%, SaaS 9.5%
          - Down-round logic only applies to biotech or when explicitly flagged
        - **Flexible Natural Language Parsing (December 2024)**:
          - Recognizes "comparable companies: A at 7x, B at 4.5x" → calculates median multiple
          - Parses cap tables: "founders hold X, investors hold Y, options Z" → sums to sharesOutstandingPreIPO
          - Handles "$600M with $480M primary and $120M secondary" → extracts all proceeds correctly
          - Recognizes "WACC 11.8%", "CapEx at 7%", "terminal growth 3.5%" in natural language
        - **DCF Independence**: Fair value computed independently using user WACC/terminal growth, not reverse-engineered from offer price
      - Default LLM: Zhi 5 (Grok) using grok-3 model for finance models.
    - **Data Science Panel**: Generates production-ready Python code for machine learning and statistical analysis from natural language descriptions.
      - **Regression Models**: Fully implemented with 7 regression types:
        - Simple Linear Regression, Multiple Linear Regression, Polynomial Regression
        - Ridge Regression (L2 regularization), Lasso Regression (L1 regularization), Elastic Net
        - Logistic Regression (classification)
        - Output includes: data loading, EDA, preprocessing, model training, cross-validation, evaluation metrics, coefficient interpretation, visualizations, prediction function, model persistence
        - Downloadable as .py file with complete, executable code
      - **Machine Learning Models**: Coming soon (Random Forest, XGBoost, Neural Networks, clustering)
      - **Statistical Forecasting**: Coming soon (ARIMA, SARIMA, Prophet, exponential smoothing)
      - **Predictive Analytics**: Coming soon (end-to-end pipelines, feature engineering, model comparison)
      - Default LLM: Zhi 5 (Grok) for data science code generation.
- **UI/UX**: Utilizes shadcn/ui and TailwindCSS for styling, offering detailed card-based layouts for analysis reports and supporting PDF/text downloads, document upload, and output download.

## External Dependencies
- **AI Service Providers**: OpenAI API (GPT-4) (ZHI 1), Anthropic API (Claude) (ZHI 2), DeepSeek API (ZHI 3), Perplexity AI (ZHI 4), Grok API (xAI) (ZHI 5).
- **Supporting Services**: Mathpix OCR, AssemblyAI, SendGrid, Google Custom Search, Stripe (for credit purchases), AnalyticPhilosophy.net Zhi API (for external knowledge queries).
- **Database & Infrastructure**: Neon/PostgreSQL, Drizzle ORM, Replit.