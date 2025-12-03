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
    - **Text Model Validator**: Includes "Truth Select" for truth-value isomorphism and "Math Truth Select" for rigorous first-order model theory and truth-grounded model searching, both supporting literal truth verification.
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
      - **3-Statement Model**: Coming soon.
      - Default LLM: Zhi 5 (Grok) using grok-3 model for finance models.
- **UI/UX**: Utilizes shadcn/ui and TailwindCSS for styling, offering detailed card-based layouts for analysis reports and supporting PDF/text downloads, document upload, and output download.

## External Dependencies
- **AI Service Providers**: OpenAI API (GPT-4) (ZHI 1), Anthropic API (Claude) (ZHI 2), DeepSeek API (ZHI 3), Perplexity AI (ZHI 4), Grok API (xAI) (ZHI 5).
- **Supporting Services**: Mathpix OCR, AssemblyAI, SendGrid, Google Custom Search, Stripe (for credit purchases), AnalyticPhilosophy.net Zhi API (for external knowledge queries).
- **Database & Infrastructure**: Neon/PostgreSQL, Drizzle ORM, Replit.