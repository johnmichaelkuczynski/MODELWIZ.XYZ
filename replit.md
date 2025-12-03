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
    - **Finance Panel**: Generates professional Excel financial models (DCF, LBO, M&A, 3-Statement) from natural language input, producing formula-based workbooks with executive summaries, assumptions, projections, valuations, and sensitivity analysis.
- **UI/UX**: Utilizes shadcn/ui and TailwindCSS for styling, offering detailed card-based layouts for analysis reports and supporting PDF/text downloads, document upload, and output download.

## External Dependencies
- **AI Service Providers**: OpenAI API (GPT-4) (ZHI 1), Anthropic API (Claude) (ZHI 2), DeepSeek API (ZHI 3), Perplexity AI (ZHI 4), Grok API (xAI) (ZHI 5).
- **Supporting Services**: Mathpix OCR, AssemblyAI, SendGrid, Google Custom Search, Stripe (for credit purchases), AnalyticPhilosophy.net Zhi API (for external knowledge queries).
- **Database & Infrastructure**: Neon/PostgreSQL, Drizzle ORM, Replit.