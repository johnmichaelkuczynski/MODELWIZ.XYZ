import React, { useState, useEffect } from "react";
import ModeToggle from "@/components/ModeToggle";
import DocumentInput from "@/components/DocumentInput";
import DocumentResults from "@/components/DocumentResults";
import ComparativeResults from "@/components/ComparativeResults";
import AIDetectionModal from "@/components/AIDetectionModal";
import ProviderSelector, { LLMProvider } from "@/components/ProviderSelector";

import ChatDialog from "@/components/ChatDialog";
import SemanticDensityAnalyzer from "@/components/SemanticDensityAnalyzer";
import CaseAssessmentModal from "@/components/CaseAssessmentModal";
import { DocumentComparisonModal } from "@/components/DocumentComparisonModal";
import { FictionAssessmentModal } from "@/components/FictionAssessmentModal";
import { FictionAssessmentPopup } from "@/components/FictionAssessmentPopup";
import { FictionComparisonModal } from "@/components/FictionComparisonModal";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Brain, Trash2, FileEdit, Loader2, Zap, Clock, Sparkles, Download, Shield, RefreshCw, Upload, FileText, BookOpen, BarChart3, AlertCircle, TrendingUp, Calculator, DollarSign, FileSpreadsheet, PiggyBank, LineChart, FileCode, Search } from "lucide-react";
import { analyzeDocument, compareDocuments, checkForAI } from "@/lib/analysis";
import { AnalysisMode, DocumentInput as DocumentInputType, AIDetectionResult, DocumentAnalysis, DocumentComparison } from "@/lib/types";
import { useToast } from "@/hooks/use-toast";
import CopyButton from "@/components/CopyButton";
import SendToButton from "@/components/SendToButton";
import { MathRenderer } from "@/components/MathRenderer";

const HomePage: React.FC = () => {
  const { toast } = useToast();
  
  // State for analysis mode
  const [mode, setMode] = useState<AnalysisMode>("single");
  
  // State for analysis type (quick vs comprehensive)
  const [analysisType, setAnalysisType] = useState<"quick" | "comprehensive">("quick");

  // State for document inputs
  const [documentA, setDocumentA] = useState<DocumentInputType>({ content: "" });
  const [documentB, setDocumentB] = useState<DocumentInputType>({ content: "" });

  // State for analysis results
  const [analysisA, setAnalysisA] = useState<DocumentAnalysis | null>(null);
  const [analysisB, setAnalysisB] = useState<DocumentAnalysis | null>(null);
  const [comparison, setComparison] = useState<DocumentComparison | null>(null);



  // State for loading indicators
  const [isAnalysisLoading, setIsAnalysisLoading] = useState(false);
  const [isAICheckLoading, setIsAICheckLoading] = useState(false);

  // State for showing results section
  const [showResults, setShowResults] = useState(false);

  // State for AI detection
  const [aiDetectionModalOpen, setAIDetectionModalOpen] = useState(false);
  const [currentAICheckDocument, setCurrentAICheckDocument] = useState<"A" | "B">("A");
  const [aiDetectionResult, setAIDetectionResult] = useState<AIDetectionResult | undefined>(undefined);


  
  // State for case assessment
  const [caseAssessmentModalOpen, setCaseAssessmentModalOpen] = useState(false);
  const [caseAssessmentResult, setCaseAssessmentResult] = useState<any>(null);
  const [isCaseAssessmentLoading, setIsCaseAssessmentLoading] = useState(false);
  
  // State for document comparison
  const [comparisonModalOpen, setComparisonModalOpen] = useState(false);
  const [comparisonResult, setComparisonResult] = useState<any>(null);
  const [isComparisonLoading, setIsComparisonLoading] = useState(false);
  
  // State for fiction assessment
  const [fictionAssessmentModalOpen, setFictionAssessmentModalOpen] = useState(false);
  const [fictionComparisonModalOpen, setFictionComparisonModalOpen] = useState(false);
  const [currentFictionDocument, setCurrentFictionDocument] = useState<"A" | "B">("A");
  const [isFictionAssessmentLoading, setIsFictionAssessmentLoading] = useState(false);
  const [fictionAssessmentResult, setFictionAssessmentResult] = useState<any>(null);
  
  // Standalone Fiction Assessment Popup State
  const [fictionPopupOpen, setFictionPopupOpen] = useState(false);

  // State for maximize intelligence feature
  const [maximizeIntelligenceModalOpen, setMaximizeIntelligenceModalOpen] = useState(false);
  const [customInstructions, setCustomInstructions] = useState("");
  const [useExternalKnowledge, setUseExternalKnowledge] = useState(false);
  const [isMaximizeIntelligenceLoading, setIsMaximizeIntelligenceLoading] = useState(false);
  const [rewriteResult, setRewriteResult] = useState<string>("");
  const [rewriteResultsModalOpen, setRewriteResultsModalOpen] = useState(false);
  const [rewriteResultData, setRewriteResultData] = useState<any>(null);
  
  
  // Streaming state for real-time analysis
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');

  // Default instructions for maximize intelligence
  const defaultInstructions = `REWRITE IN SUCH THAT THE RESULTING DOCUMENT SCORES MAXIMALLY HIGH ON EACH OF THE FOLLOWING QUESTIONS (SO FAR AS THAT IS POSSIBLE WITHOUT TOTALLY CHANGING THE CONTENT), THE QUESTIONS IN QUESTION BEING:

IS IT INSIGHTFUL?
DOES IT DEVELOP POINTS? (OR, IF IT IS A SHORT EXCERPT, IS THERE EVIDENCE THAT IT WOULD DEVELOP POINTS IF EXTENDED)?
IS THE ORGANIZATION MERELY SEQUENTIAL (JUST ONE POINT AFTER ANOTHER, LITTLE OR NO LOGICAL SCAFFOLDING)? OR ARE THE IDEAS ARRANGED, NOT JUST SEQUENTIALLY BUT HIERARCHICALLY?
IF THE POINTS IT MAKES ARE NOT INSIGHTFUL, DOES IT OPERATE SKILLFULLY WITH CANONS OF LOGIC/REASONING.
ARE THE POINTS CLICHES? OR ARE THEY "FRESH"?
DOES IT USE TECHNICAL JARGON TO OBFUSCATE OR TO RENDER MORE PRECISE?
IS IT ORGANIC? DO POINTS DEVELOP IN AN ORGANIC, NATURAL WAY? DO THEY 'UNFOLD'? OR ARE THEY FORCED AND ARTIFICIAL?
DOES IT OPEN UP NEW DOMAINS? OR, ON THE CONTRARY, DOES IT SHUT OFF INQUIRY (BY CONDITIONALIZING FURTHER DISCUSSION OF THE MATTERS ON ACCEPTANCE OF ITS INTERNAL AND POSSIBLY VERY FAULTY LOGIC)?
IS IT ACTUALLY INTELLIGENT OR JUST THE WORK OF SOMEBODY WHO, JUDGING BY THE SUBJECT-MATTER, IS PRESUMED TO BE INTELLIGENT (BUT MAY NOT BE)?
IS IT REAL OR IS IT PHONY?
DO THE SENTENCES EXHIBIT COMPLEX AND COHERENT INTERNAL LOGIC?
IS THE PASSAGE GOVERNED BY A STRONG CONCEPT? OR IS THE ONLY ORGANIZATION DRIVEN PURELY BY EXPOSITORY (AS OPPOSED TO EPISTEMIC) NORMS?
IS THERE SYSTEM-LEVEL CONTROL OVER IDEAS? IN OTHER WORDS, DOES THE AUTHOR SEEM TO RECALL WHAT HE SAID EARLIER AND TO BE IN A POSITION TO INTEGRATE IT INTO POINTS HE HAS MADE SINCE THEN?
ARE THE POINTS 'REAL'? ARE THEY FRESH? OR IS SOME INSTITUTION OR SOME ACCEPTED VEIN OF PROPAGANDA OR ORTHODOXY JUST USING THE AUTHOR AS A MOUTH PIECE?
IS THE WRITING EVASIVE OR DIRECT?
ARE THE STATEMENTS AMBIGUOUS?
DOES THE PROGRESSION OF THE TEXT DEVELOP ACCORDING TO WHO SAID WHAT OR ACCORDING TO WHAT ENTAILS OR CONFIRMS WHAT?
DOES THE AUTHOR USE OTHER AUTHORS TO DEVELOP HIS IDEAS OR TO CLOAK HIS OWN LACK OF IDEAS?`;
  
  // State for LLM provider
  const [selectedProvider, setSelectedProvider] = useState<LLMProvider>("zhi1");

  // GPT Bypass Humanizer State - Following Exact Protocol
  const [boxA, setBoxA] = useState(""); // AI text to humanize
  const [boxB, setBoxB] = useState(""); // Human style sample  
  const [boxC, setBoxC] = useState(""); // Humanized output
  const [boxAScore, setBoxAScore] = useState<number | null>(null);
  const [boxBScore, setBoxBScore] = useState<number | null>(null);
  const [boxCScore, setBoxCScore] = useState<number | null>(null);
  const [humanizerCustomInstructions, setHumanizerCustomInstructions] = useState("");
  const [selectedStylePresets, setSelectedStylePresets] = useState<string[]>([]);
  const [selectedWritingSample, setSelectedWritingSample] = useState("Content-Neutral|Formal and Functional Relationships");
  const [humanizerProvider, setHumanizerProvider] = useState<LLMProvider>("zhi2"); // Anthropic default
  const [isHumanizerLoading, setIsHumanizerLoading] = useState(false);
  const [isReRewriteLoading, setIsReRewriteLoading] = useState(false);
  const [writingSamples, setWritingSamples] = useState<any>({});
  const [stylePresets, setStylePresets] = useState<any>({});
  const [chunks, setChunks] = useState<any[]>([]);
  const [selectedChunkIds, setSelectedChunkIds] = useState<string[]>([]);
  const [showChunkSelector, setShowChunkSelector] = useState(false);
  
  // Text Model Validator State
  const [validatorInputText, setValidatorInputText] = useState("");
  const [validatorMode, setValidatorMode] = useState<"reconstruction" | "isomorphism" | "mathmodel" | "autodecide" | "truth-isomorphism" | "math-truth-select" | null>(null);
  const [validatorOutput, setValidatorOutput] = useState<string>("");
  const [validatorLoading, setValidatorLoading] = useState(false);
  const [validatorTargetDomain, setValidatorTargetDomain] = useState("");
  const [validatorFidelityLevel, setValidatorFidelityLevel] = useState<"conservative" | "aggressive">("conservative");
  const [validatorMathFramework, setValidatorMathFramework] = useState("variational-inference");
  const [validatorConstraintType, setValidatorConstraintType] = useState<"pure-swap" | "true-statements" | "historical">("pure-swap");
  const [validatorRigorLevel, setValidatorRigorLevel] = useState<"sketch" | "semi-formal" | "proof-ready">("semi-formal");
  const [showValidatorCustomization, setShowValidatorCustomization] = useState(false);
  const [validatorCustomInstructions, setValidatorCustomInstructions] = useState("");
  const [showRedoModal, setShowRedoModal] = useState(false);
  const [redoCustomInstructions, setRedoCustomInstructions] = useState("");
  const [validatorTruthMapping, setValidatorTruthMapping] = useState<"false-to-true" | "true-to-true" | "true-to-false">("false-to-true");
  const [validatorMathTruthMapping, setValidatorMathTruthMapping] = useState<"make-true" | "keep-true" | "make-false">("make-true");
  const [validatorLiteralTruth, setValidatorLiteralTruth] = useState(false);
  const [validatorLLMProvider, setValidatorLLMProvider] = useState<string>("zhi5"); // Default to Grok
  
  // Coherence Meter State
  const [coherenceInputText, setCoherenceInputText] = useState("");
  const [coherenceType, setCoherenceType] = useState<"logical-consistency" | "logical-cohesiveness" | "scientific-explanatory" | "thematic-psychological" | "instructional" | "motivational" | "mathematical" | "philosophical" | "auto-detect">("auto-detect");
  const [coherenceAnalysis, setCoherenceAnalysis] = useState<string>("");
  const [coherenceRewrite, setCoherenceRewrite] = useState<string>("");
  const [coherenceChanges, setCoherenceChanges] = useState<string>("");
  const [coherenceLoading, setCoherenceLoading] = useState(false);
  const [coherenceMode, setCoherenceMode] = useState<"analyze" | "rewrite" | null>(null);
  const [coherenceScore, setCoherenceScore] = useState<number | null>(null);
  const [coherenceAssessment, setCoherenceAssessment] = useState<"PASS" | "WEAK" | "FAIL" | null>(null);
  const [coherenceAggressiveness, setCoherenceAggressiveness] = useState<"conservative" | "moderate" | "aggressive">("aggressive");
  const [coherenceProcessingMode, setCoherenceProcessingMode] = useState<"simple" | "outline-guided">("simple");
  const [coherenceChunks, setCoherenceChunks] = useState<Array<{id: string, text: string, preview: string}>>([]);
  const [selectedCoherenceChunks, setSelectedCoherenceChunks] = useState<string[]>([]);
  const [showCoherenceChunkSelector, setShowCoherenceChunkSelector] = useState(false);
  const [coherenceStageProgress, setCoherenceStageProgress] = useState<string>("");
  
  // Finance Panel State
  const [financeModelType, setFinanceModelType] = useState<"dcf" | "lbo" | "ma" | "threestatement" | null>(null);
  const [financeInputText, setFinanceInputText] = useState("");
  const [financeCustomInstructions, setFinanceCustomInstructions] = useState("");
  const [financeLoading, setFinanceLoading] = useState(false);
  const [financeLoadingPhase, setFinanceLoadingPhase] = useState<string>("");
  const [financeResult, setFinanceResult] = useState<any>(null);
  const [showFinanceCustomization, setShowFinanceCustomization] = useState(false);
  const [financeLLMProvider, setFinanceLLMProvider] = useState<"zhi1" | "zhi2" | "zhi3" | "zhi4" | "zhi5">("zhi5");
  const [financeDownloadLoading, setFinanceDownloadLoading] = useState(false);
  
  // Data Science Panel State
  const [dataScienceModelType, setDataScienceModelType] = useState<"regression" | "ml" | "forecasting" | "predictive" | null>(null);
  const [dataScienceInputText, setDataScienceInputText] = useState("");
  const [dataScienceCustomInstructions, setDataScienceCustomInstructions] = useState("");
  const [dataScienceLoading, setDataScienceLoading] = useState(false);
  const [dataScienceResult, setDataScienceResult] = useState<any>(null);
  const [dataScienceLLMProvider, setDataScienceLLMProvider] = useState<"zhi1" | "zhi2" | "zhi3" | "zhi4" | "zhi5">("zhi5");
  const [dataScienceDownloadLoading, setDataScienceDownloadLoading] = useState(false);
  
  // Load writing samples and style presets on component mount
  useEffect(() => {
    const loadWritingSamples = async () => {
      try {
        const response = await fetch('/api/writing-samples');
        if (response.ok) {
          const data = await response.json();
          setWritingSamples(data.samples);
          // Set default to "Formal and Functional Relationships" (CONTENT-NEUTRAL default)
          if (data.samples["CONTENT-NEUTRAL"] && data.samples["CONTENT-NEUTRAL"]["Formal and Functional Relationships"]) {
            setBoxB(data.samples["CONTENT-NEUTRAL"]["Formal and Functional Relationships"]);
          }
        }
      } catch (error) {
        console.error('Failed to load writing samples:', error);
      }
    };

    const loadStylePresets = async () => {
      try {
        const response = await fetch('/api/style-presets');
        if (response.ok) {
          const data = await response.json();
          setStylePresets(data.presets);
        }
      } catch (error) {
        console.error('Failed to load style presets:', error);
      }
    };

    loadWritingSamples();
    loadStylePresets();
  }, []);

  // GPT Bypass Humanizer Functions - Following Exact Protocol
  
  // Debounce function for delayed execution
  const debounce = (func: Function, delay: number) => {
    let timeoutId: NodeJS.Timeout;
    return (...args: any[]) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func(...args), delay);
    };
  };

  // Automatic GPTZero evaluation (no button push needed)
  const evaluateTextAI = async (text: string, setScore: (score: number) => void) => {
    if (!text.trim()) return;

    try {
      const response = await fetch('/api/evaluate-ai', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          // Ensure we display the percentage correctly (not negative values)
          const humanPercentage = Math.max(0, Math.min(100, data.humanPercentage));
          setScore(humanPercentage);
        }
      }
    } catch (error) {
      console.error('AI evaluation error:', error);
    }
  };

  // File upload handler for PDF/Word/Doc
  const handleFileUpload = async (file: File, setter: (content: string) => void) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/extract-text', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to extract text from document');
      }

      const data = await response.json();
      setter(data.content);
      toast({
        title: "File Uploaded",
        description: `Successfully loaded ${file.name}`,
      });
    } catch (error) {
      console.error('File upload error:', error);
      toast({
        title: "Upload Failed", 
        description: "Could not read the file. Please try a different format.",
        variant: "destructive",
      });
    }
  };

  // Download text as file
  const handleDownloadText = (text: string, filename: string) => {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast({
      title: "Download Started",
      description: `Downloading ${filename}`,
    });
  };

  // Text chunking for large documents (500+ words)
  const handleChunkText = async (text: string) => {
    try {
      const response = await fetch('/api/chunk-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, maxWords: 500 }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setChunks(data.chunks);
          setShowChunkSelector(true);
          toast({
            title: "Text Chunked",
            description: `Document divided into ${data.chunks.length} chunks of ~500 words each.`,
          });
        }
      }
    } catch (error) {
      console.error('Text chunking error:', error);
    }
  };

  // Main humanization function with surgical precision
  const handleHumanize = async () => {
    if (!boxA.trim() || !boxB.trim()) {
      toast({
        title: "Missing Input",
        description: "Both Box A (AI text) and Box B (human style sample) are required.",
        variant: "destructive",
      });
      return;
    }

    setIsHumanizerLoading(true);
    setBoxC("");
    setBoxCScore(null);

    try {
      const response = await fetch('/api/gpt-bypass-humanizer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          boxA,
          boxB,
          stylePresets: selectedStylePresets,
          provider: humanizerProvider,
          customInstructions: humanizerCustomInstructions,
          selectedChunkIds: selectedChunkIds.length > 0 ? selectedChunkIds : undefined,
          chunks: chunks.length > 0 ? chunks : undefined
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Humanization failed');
      }

      const data = await response.json();
      if (data.success && data.result) {
        setBoxC(data.result.humanizedText);
        
        // Automatically evaluate humanized text
        setTimeout(() => {
          evaluateTextAI(data.result.humanizedText, setBoxCScore);
        }, 1000);
        
        toast({
          title: "Humanization Complete!",
          description: `Text humanized with surgical precision. Original: ${data.result.originalScore || 'N/A'}% → Humanized: ${data.result.humanizedScore || 'Evaluating...'}% Human.`,
        });
      }
    } catch (error: any) {
      console.error('Humanization error:', error);
      toast({
        title: "Humanization Failed",
        description: error.message || "An error occurred during humanization.",
        variant: "destructive",
      });
    } finally {
      setIsHumanizerLoading(false);
    }
  };

  // Re-rewrite function for recursive rewriting
  const handleReRewrite = async () => {
    if (!boxC.trim() || !boxB.trim()) {
      toast({
        title: "Missing Input",
        description: "Both output text and style sample are required for re-rewrite.",
        variant: "destructive",
      });
      return;
    }

    setIsReRewriteLoading(true);

    try {
      const response = await fetch('/api/gpt-bypass-humanizer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          boxA: boxC, // Use current output as new input
          boxB,
          stylePresets: selectedStylePresets,
          provider: humanizerProvider,
          customInstructions: humanizerCustomInstructions + " [RECURSIVE REWRITE] Further improve human-like qualities and reduce AI detection."
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Re-rewrite failed');
      }

      const data = await response.json();
      if (data.success && data.result) {
        setBoxC(data.result.humanizedText);
        
        // Automatically evaluate re-rewritten text
        setTimeout(() => {
          evaluateTextAI(data.result.humanizedText, setBoxCScore);
        }, 1000);
        
        toast({
          title: "Re-rewrite Complete!",
          description: `Text re-rewritten recursively. New score: ${data.result.humanizedScore || 'Evaluating...'}% Human.`,
        });
      }
    } catch (error: any) {
      console.error('Re-rewrite error:', error);
      toast({
        title: "Re-rewrite Failed",
        description: error.message || "An error occurred during re-rewrite.",
        variant: "destructive",
      });
    } finally {
      setIsReRewriteLoading(false);
    }
  };

  // Download function for PDF/TXT/Word
  const downloadHumanizerResult = (format: 'pdf' | 'txt' | 'docx') => {
    if (!boxC.trim()) return;

    const filename = `humanized-text.${format}`;
    const blob = new Blob([boxC], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    
    toast({
      title: "Download Complete",
      description: `Humanized text saved as ${filename}`,
    });
  };

  // Send functionality handlers
  const handleSendToHumanizer = (text: string) => {
    setBoxA(text);
    toast({
      title: "Text sent to Humanizer",
      description: "Text has been placed in Box A for humanization"
    });
  };

  const handleSendToIntelligence = (text: string) => {
    setDocumentA({ ...documentA, content: text });
    toast({
      title: "Text sent to Intelligence Analysis",
      description: "Text has been placed in the intelligence analysis input"
    });
  };

  const handleSendToChat = (text: string) => {
    // This will be handled by the ChatDialog component
    // For now, we can show a notification that the text will be available to chat
    toast({
      title: "Text available to Chat",
      description: "The text is now available as context for AI chat"
    });
  };

  // Text Model Validator Handler
  const handleValidatorProcess = async (mode: "reconstruction" | "isomorphism" | "mathmodel" | "autodecide" | "truth-isomorphism" | "math-truth-select") => {
    if (!validatorInputText.trim()) {
      toast({
        title: "No Input Text",
        description: "Please enter text to validate",
        variant: "destructive"
      });
      return;
    }

    setValidatorMode(mode);
    setValidatorLoading(true);
    setValidatorOutput("");

    try {
      const response = await fetch('/api/text-model-validator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: validatorInputText,
          mode,
          targetDomain: validatorTargetDomain,
          fidelityLevel: validatorFidelityLevel,
          mathFramework: validatorMathFramework,
          constraintType: validatorConstraintType,
          rigorLevel: validatorRigorLevel,
          customInstructions: validatorCustomInstructions,
          truthMapping: validatorTruthMapping,
          mathTruthMapping: validatorMathTruthMapping,
          literalTruth: validatorLiteralTruth,
          llmProvider: validatorLLMProvider,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Validation failed');
      }

      const data = await response.json();
      if (data.success && data.output) {
        setValidatorOutput(data.output);
        toast({
          title: "Validation Complete!",
          description: `Text validated using ${mode} mode`,
        });
      }
    } catch (error: any) {
      console.error('Validator error:', error);
      toast({
        title: "Validation Failed",
        description: error.message || "An error occurred during validation.",
        variant: "destructive",
      });
    } finally {
      setValidatorLoading(false);
    }
  };

  const handleValidatorClear = () => {
    setValidatorInputText("");
    setValidatorOutput("");
    setValidatorMode(null);
    setShowValidatorCustomization(false);
    setValidatorCustomInstructions("");
  };

  // Finance Panel Handler - Two-step flow: Parse & Preview, then Download
  const handleFinanceModelGenerate = async (modelType: "dcf" | "lbo" | "ma" | "threestatement") => {
    if (!financeInputText.trim()) {
      toast({
        title: "No Input",
        description: "Please describe the company financials and projections",
        variant: "destructive"
      });
      return;
    }

    setFinanceModelType(modelType);
    setFinanceLoading(true);
    setFinanceResult(null);

    const providerNames: Record<string, string> = {
      'zhi1': 'OpenAI GPT-4',
      'zhi2': 'Anthropic Claude',
      'zhi3': 'DeepSeek',
      'zhi4': 'Perplexity',
      'zhi5': 'Grok'
    };

    try {
      if (modelType === "dcf") {
        setFinanceLoadingPhase(`Parsing with ${providerNames[financeLLMProvider]}...`);
        
        const response = await fetch('/api/finance/parse-dcf', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            description: financeInputText,
            customInstructions: financeCustomInstructions,
            llmProvider: financeLLMProvider,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || 'Parsing failed');
        }

        const result = await response.json();
        setFinanceResult(result);
        
        toast({
          title: "Analysis Complete",
          description: `DCF valuation computed for ${result.assumptions?.companyName || 'company'}`,
        });
      } else if (modelType === "lbo") {
        setFinanceLoadingPhase(`Parsing LBO with ${providerNames[financeLLMProvider]}...`);
        
        const response = await fetch('/api/finance/parse-lbo', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            description: financeInputText,
            customInstructions: financeCustomInstructions,
            llmProvider: financeLLMProvider,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || 'LBO parsing failed');
        }

        const result = await response.json();
        setFinanceResult(result);
        
        toast({
          title: "LBO Analysis Complete",
          description: `Returns calculated for ${result.assumptions?.companyName || 'target company'}`,
        });
      } else if (modelType === "ma") {
        setFinanceLoadingPhase(`Parsing M&A with ${providerNames[financeLLMProvider]}...`);
        
        const response = await fetch('/api/finance/parse-ma', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            description: financeInputText,
            customInstructions: financeCustomInstructions,
            llmProvider: financeLLMProvider,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || 'M&A parsing failed');
        }

        const result = await response.json();
        setFinanceResult(result);
        
        toast({
          title: "M&A Analysis Complete",
          description: `Accretion/dilution analysis for ${result.assumptions?.acquirerName || 'acquirer'} acquiring ${result.assumptions?.targetName || 'target'}`,
        });
      } else if (modelType === "threestatement") {
        setFinanceLoadingPhase(`Parsing 3-Statement with ${providerNames[financeLLMProvider]}...`);
        
        const response = await fetch('/api/finance/parse-3statement', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            description: financeInputText,
            customInstructions: financeCustomInstructions,
            llmProvider: financeLLMProvider,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || '3-Statement parsing failed');
        }

        const result = await response.json();
        setFinanceResult(result);
        
        toast({
          title: "3-Statement Model Complete",
          description: `Financial projections for ${result.assumptions?.companyName || 'company'} with ${result.summary?.isBalanced ? 'balanced' : 'unbalanced'} balance sheet`,
        });
      } else {
        toast({
          title: "Coming Soon",
          description: `This model type is not yet available.`,
          variant: "destructive"
        });
      }
    } catch (error: any) {
      console.error('Finance model error:', error);
      toast({
        title: "Analysis Failed",
        description: error.message || "An error occurred during analysis.",
        variant: "destructive",
      });
    } finally {
      setFinanceLoading(false);
      setFinanceLoadingPhase("");
    }
  };

  // Download Excel from parsed results
  const handleFinanceDownloadExcel = async () => {
    if (!financeResult?.assumptions) {
      toast({
        title: "No Results",
        description: "Please generate a valuation first",
        variant: "destructive"
      });
      return;
    }

    setFinanceDownloadLoading(true);

    try {
      // Choose the appropriate endpoint based on model type
      let endpoint = '/api/finance/download-dcf';
      let modelLabel = 'DCF';
      if (financeModelType === "lbo") {
        endpoint = '/api/finance/download-lbo';
        modelLabel = 'LBO';
      } else if (financeModelType === "ma") {
        endpoint = '/api/finance/download-ma';
        modelLabel = 'M&A';
      } else if (financeModelType === "threestatement") {
        endpoint = '/api/finance/download-3statement';
        modelLabel = '3-Statement';
      }
      
      const requestBody = financeModelType === "threestatement" 
        ? { result: financeResult }
        : { assumptions: financeResult.assumptions };
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Download failed');
      }

      // Handle file download - get array buffer and create blob with correct type
      const arrayBuffer = await response.arrayBuffer();
      const blob = new Blob([arrayBuffer], { 
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' 
      });
      
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `${modelLabel}_Model_${new Date().toISOString().split('T')[0]}.xlsx`;
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
        if (filenameMatch) {
          filename = filenameMatch[1].replace(/"/g, '');
        }
      }

      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast({
        title: "Excel Downloaded",
        description: `Model saved as ${filename}`,
      });
    } catch (error: any) {
      console.error('Excel download error:', error);
      toast({
        title: "Download Failed",
        description: error.message || "Failed to download Excel file.",
        variant: "destructive",
      });
    } finally {
      setFinanceDownloadLoading(false);
    }
  };

  const handleFinanceClear = () => {
    setFinanceInputText("");
    setFinanceCustomInstructions("");
    setFinanceModelType(null);
    setFinanceResult(null);
    setShowFinanceCustomization(false);
    setFinanceLoadingPhase("");
  };

  // Data Science Panel Handlers
  const handleDataScienceProcess = async (modelType: "regression" | "ml" | "forecasting" | "predictive") => {
    if (!dataScienceInputText.trim()) {
      toast({
        title: "No Input",
        description: "Please enter a problem description",
        variant: "destructive"
      });
      return;
    }

    setDataScienceModelType(modelType);
    setDataScienceLoading(true);
    setDataScienceResult(null);

    try {
      let endpoint = "";
      switch (modelType) {
        case "regression":
          endpoint = "/api/datascience/parse-regression";
          break;
        case "ml":
          endpoint = "/api/datascience/parse-ml";
          break;
        case "forecasting":
          endpoint = "/api/datascience/parse-forecasting";
          break;
        case "predictive":
          endpoint = "/api/datascience/parse-predictive";
          break;
      }

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          description: dataScienceInputText,
          customInstructions: dataScienceCustomInstructions,
          llmProvider: dataScienceLLMProvider
        })
      });

      const result = await response.json();

      if (result.success) {
        setDataScienceResult(result);
        toast({
          title: "Code Generated",
          description: `${modelType.charAt(0).toUpperCase() + modelType.slice(1)} model code ready - Analysis by ${result.providerUsed}`
        });
      } else {
        throw new Error(result.message || "Failed to generate code");
      }
    } catch (error: any) {
      console.error("Data science error:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to generate code",
        variant: "destructive"
      });
    } finally {
      setDataScienceLoading(false);
    }
  };

  const handleDataScienceDownload = async () => {
    if (!dataScienceResult?.pythonCode) {
      toast({
        title: "No Code",
        description: "Generate code first before downloading",
        variant: "destructive"
      });
      return;
    }

    setDataScienceDownloadLoading(true);

    try {
      const isML = dataScienceResult.parameters?.problemType && ['classification', 'clustering', 'dimensionality_reduction'].includes(dataScienceResult.parameters.problemType);
      const isForecasting = dataScienceResult.parameters?.modelType && ['arima', 'sarima', 'sarimax', 'holtwinters', 'ets', 'prophet', 'auto'].includes(dataScienceResult.parameters.modelType.toLowerCase());
      const isPredictive = dataScienceModelType === 'predictive' || 
        dataScienceResult.parameters?.featureEngineeringLevel !== undefined ||
        dataScienceResult.parameters?.ensembleMethods !== undefined ||
        dataScienceResult.parameters?.interpretabilityLevel !== undefined;
      
      let downloadEndpoint = "/api/datascience/download-regression";
      let defaultFilename = "regression_model.py";
      
      if (isPredictive) {
        downloadEndpoint = "/api/datascience/download-predictive";
        defaultFilename = "predictive_pipeline.py";
      } else if (isForecasting) {
        downloadEndpoint = "/api/datascience/download-forecasting";
        defaultFilename = "forecasting_model.py";
      } else if (isML) {
        downloadEndpoint = "/api/datascience/download-ml";
        defaultFilename = "ml_model.py";
      }
      
      const response = await fetch(downloadEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pythonCode: dataScienceResult.pythonCode,
          parameters: dataScienceResult.parameters
        })
      });

      if (!response.ok) {
        throw new Error("Failed to download file");
      }

      const blob = await response.blob();
      const contentDisposition = response.headers.get("Content-Disposition");
      const filenameMatch = contentDisposition?.match(/filename="(.+)"/);
      const filename = filenameMatch ? filenameMatch[1] : defaultFilename;

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      toast({
        title: "Download Complete",
        description: `Saved as ${filename}`
      });
    } catch (error: any) {
      console.error("Download error:", error);
      toast({
        title: "Download Failed",
        description: error.message || "Could not download file",
        variant: "destructive"
      });
    } finally {
      setDataScienceDownloadLoading(false);
    }
  };

  const handleDataScienceClear = () => {
    setDataScienceInputText("");
    setDataScienceCustomInstructions("");
    setDataScienceModelType(null);
    setDataScienceResult(null);
  };

  // Global Clear All Handler - clears Finance and Data Science panels
  const handleGlobalClearAll = () => {
    // Clear Finance Panel
    setFinanceInputText("");
    setFinanceCustomInstructions("");
    setFinanceModelType(null);
    setFinanceResult(null);
    setShowFinanceCustomization(false);
    setFinanceLoadingPhase("");
    
    // Clear Data Science Panel
    setDataScienceInputText("");
    setDataScienceCustomInstructions("");
    setDataScienceModelType(null);
    setDataScienceResult(null);
    
    toast({
      title: "All Cleared",
      description: "Finance and Data Science panels have been reset",
    });
  };

  // Coherence Meter Handlers
  const createCoherenceChunks = (text: string) => {
    const words = text.trim().split(/\s+/);
    const chunkSize = 400; // ~400 words per chunk
    const chunks: Array<{id: string, text: string, preview: string}> = [];
    
    for (let i = 0; i < words.length; i += chunkSize) {
      const chunkWords = words.slice(i, i + chunkSize);
      const chunkText = chunkWords.join(' ');
      const preview = chunkWords.slice(0, 20).join(' ') + (chunkWords.length > 20 ? '...' : '');
      
      chunks.push({
        id: `chunk-${i / chunkSize + 1}`,
        text: chunkText,
        preview: preview
      });
    }
    
    return chunks;
  };

  const handleCoherenceAnalyze = async () => {
    const wordCount = coherenceInputText.trim().split(/\s+/).length;
    
    if (!coherenceInputText.trim()) {
      toast({
        title: "No Input Text",
        description: "Please enter text to analyze for coherence",
        variant: "destructive"
      });
      return;
    }
    
    // If text is longer than 500 words, create chunks and show selector
    if (wordCount > 500) {
      const chunks = createCoherenceChunks(coherenceInputText);
      setCoherenceChunks(chunks);
      setSelectedCoherenceChunks(chunks.map(c => c.id)); // Select all by default
      setShowCoherenceChunkSelector(true);
      toast({
        title: "Text Too Long for Single Analysis",
        description: `Your text has ${wordCount} words. It has been divided into ${chunks.length} sections. Select which sections to analyze.`,
      });
      return;
    }

    setCoherenceLoading(true);
    setCoherenceMode("analyze");
    setCoherenceAnalysis("");
    setCoherenceScore(null);
    setCoherenceAssessment(null);

    try {
      const response = await fetch('/api/coherence-meter', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: coherenceInputText,
          coherenceType,
          mode: "analyze"
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Analysis failed');
      }

      const data = await response.json();
      if (data.success) {
        setCoherenceAnalysis(data.analysis);
        setCoherenceScore(data.score);
        setCoherenceAssessment(data.assessment);
        toast({
          title: "Coherence Analysis Complete!",
          description: `Score: ${data.score}/10 - ${data.assessment}`,
        });
      }
    } catch (error: any) {
      console.error('Coherence analysis error:', error);
      toast({
        title: "Analysis Failed",
        description: error.message || "An error occurred during coherence analysis.",
        variant: "destructive",
      });
    } finally {
      setCoherenceLoading(false);
    }
  };

  const handleCoherenceRewrite = async () => {
    const wordCount = coherenceInputText.trim().split(/\s+/).length;
    
    if (!coherenceInputText.trim()) {
      toast({
        title: "No Input Text",
        description: "Please enter text to rewrite for coherence",
        variant: "destructive"
      });
      return;
    }
    
    // If text is longer than 500 words, create chunks and show selector
    if (wordCount > 500) {
      const chunks = createCoherenceChunks(coherenceInputText);
      setCoherenceChunks(chunks);
      setSelectedCoherenceChunks(chunks.map(c => c.id)); // Select all by default
      setShowCoherenceChunkSelector(true);
      toast({
        title: "Text Too Long for Single Rewrite",
        description: `Your text has ${wordCount} words. It has been divided into ${chunks.length} sections. Select which sections to rewrite.`,
      });
      return;
    }

    setCoherenceLoading(true);
    setCoherenceMode("rewrite");
    setCoherenceRewrite("");
    setCoherenceChanges("");

    try {
      const response = await fetch('/api/coherence-meter', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: coherenceInputText,
          coherenceType,
          mode: "rewrite",
          aggressiveness: coherenceAggressiveness
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Rewrite failed');
      }

      const data = await response.json();
      if (data.success) {
        setCoherenceRewrite(data.rewrite);
        setCoherenceChanges(data.changes);
        toast({
          title: "Coherence Rewrite Complete!",
          description: `Text rewritten to maximize ${coherenceType} coherence`,
        });
      }
    } catch (error: any) {
      console.error('Coherence rewrite error:', error);
      toast({
        title: "Rewrite Failed",
        description: error.message || "An error occurred during coherence rewriting.",
        variant: "destructive",
      });
    } finally {
      setCoherenceLoading(false);
    }
  };

  const handleProcessSelectedChunks = async (mode: "analyze" | "rewrite") => {
    if (selectedCoherenceChunks.length === 0) {
      toast({
        title: "No Sections Selected",
        description: "Please select at least one section to process",
        variant: "destructive"
      });
      return;
    }

    const selectedChunkObjects = coherenceChunks.filter(c => selectedCoherenceChunks.includes(c.id));
    
    setCoherenceLoading(true);
    setCoherenceMode(mode);
    setShowCoherenceChunkSelector(false);
    
    if (mode === "analyze") {
      setCoherenceAnalysis("");
      setCoherenceScore(null);
      setCoherenceAssessment(null);
    } else {
      setCoherenceRewrite("");
      setCoherenceChanges("");
    }

    // Check if outline-guided mode is selected
    if (coherenceProcessingMode === "outline-guided") {
      // Use outline-guided endpoint for full text
      const fullText = selectedChunkObjects.map(c => c.text).join('\n\n');
      
      setCoherenceStageProgress("STAGE 1: Generating document outline...\nThis may take a moment...");
      
      try {
        const response = await fetch('/api/coherence-outline-guided', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: fullText,
            coherenceType,
            mode,
            aggressiveness: coherenceAggressiveness
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || `Outline-guided ${mode} failed`);
        }

        const data = await response.json();
        if (data.success) {
          if (mode === "analyze") {
            setCoherenceAnalysis(data.analysis);
          } else {
            setCoherenceRewrite(data.rewrite);
            setCoherenceChanges(data.changes);
          }
          
          toast({
            title: "Outline-Guided Processing Complete!",
            description: `Successfully processed using two-stage approach`,
          });
        }
      } catch (error: any) {
        console.error(`Outline-guided ${mode} error:`, error);
        toast({
          title: `${mode.charAt(0).toUpperCase() + mode.slice(1)} Failed`,
          description: error.message || `An error occurred during outline-guided ${mode}.`,
          variant: "destructive",
        });
      } finally {
        setCoherenceLoading(false);
        setCoherenceStageProgress("");
      }
    } else {
      // Use simple chunking mode - process each chunk independently
      let combinedAnalysis = "";
      let combinedRewrite = "";
      let combinedChanges = "";

      try {
        for (let i = 0; i < selectedChunkObjects.length; i++) {
          const chunk = selectedChunkObjects[i];
          
          toast({
            title: `Processing Section ${i + 1}/${selectedChunkObjects.length}`,
            description: `Analyzing: "${chunk.preview}"`,
          });

          const response = await fetch('/api/coherence-meter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: chunk.text,
              coherenceType,
              mode,
              aggressiveness: coherenceAggressiveness
            }),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `${mode} failed for section ${i + 1}`);
          }

          const data = await response.json();
          if (data.success) {
            if (mode === "analyze") {
              combinedAnalysis += `\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nSECTION ${i + 1} of ${selectedChunkObjects.length}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n${data.analysis}`;
            } else {
              combinedRewrite += `\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nSECTION ${i + 1} of ${selectedChunkObjects.length}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n${data.rewrite}`;
              combinedChanges += `\n\n━━━━ SECTION ${i + 1} ━━━━\n${data.changes}`;
            }
          }
        }

        if (mode === "analyze") {
          setCoherenceAnalysis(combinedAnalysis.trim());
          toast({
            title: "All Sections Analyzed!",
            description: `Processed ${selectedChunkObjects.length} sections successfully`,
          });
        } else {
          setCoherenceRewrite(combinedRewrite.trim());
          setCoherenceChanges(combinedChanges.trim());
          toast({
            title: "All Sections Rewritten!",
            description: `Processed ${selectedChunkObjects.length} sections successfully`,
          });
        }
      } catch (error: any) {
        console.error(`Coherence ${mode} error:`, error);
        toast({
          title: `${mode.charAt(0).toUpperCase() + mode.slice(1)} Failed`,
          description: error.message || `An error occurred during coherence ${mode}.`,
          variant: "destructive",
        });
      } finally {
        setCoherenceLoading(false);
      }
    }
  };

  const handleCoherenceClear = () => {
    setCoherenceInputText("");
    setCoherenceAnalysis("");
    setCoherenceRewrite("");
    setCoherenceChanges("");
    setCoherenceMode(null);
    setCoherenceScore(null);
    setCoherenceAssessment(null);
    setCoherenceChunks([]);
    setSelectedCoherenceChunks([]);
    setShowCoherenceChunkSelector(false);
  };


  // FIXED streaming function
  const startStreaming = async (text: string, provider: string) => {
    console.log('startStreaming called with:', { text: text.slice(0, 50), provider });
    
    try {
      console.log('Making fetch request to /api/stream-analysis...');
      
      const response = await fetch('/api/stream-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text, provider }),
      });

      console.log('Response received:', response.status, response.statusText);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      console.log('Starting to read stream...');
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          console.log('Stream ended');
          setIsStreaming(false);
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        console.log('Received chunk:', chunk);
        
        if (chunk) {
          setStreamingContent(prev => {
            const newContent = prev + chunk;
            console.log('Updated content length:', newContent.length);
            return newContent;
          });
        }
      }
      
    } catch (error) {
      console.error('Streaming error:', error);
      setStreamingContent('ERROR: ' + (error instanceof Error ? error.message : 'Unknown error'));
      setIsStreaming(false);
    }
  };
  const [apiStatus, setApiStatus] = useState<{
    openai: boolean;
    anthropic: boolean;
    perplexity: boolean;
    deepseek: boolean;
    grok: boolean;
  }>({
    openai: false,
    anthropic: false,
    perplexity: false,
    deepseek: false,
    grok: false
  });
  
  // Check API status when component mounts
  useEffect(() => {
    async function checkApiStatus() {
      try {
        const response = await fetch("/api/check-api");
        const data = await response.json();
        
        if (data.api_keys) {
          setApiStatus({
            openai: data.api_keys.openai === "configured",
            anthropic: data.api_keys.anthropic === "configured",
            perplexity: data.api_keys.perplexity === "configured",
            deepseek: data.api_keys.deepseek === "configured",
            grok: data.api_keys.grok === "configured"
          });
          
          console.log("API Status:", data.api_keys);
        }
      } catch (error) {
        console.error("Error checking API status:", error);
      }
    }
    
    checkApiStatus();
  }, []);

  // Handler for checking if a document is AI-generated
  const handleCheckAI = async (documentId: "A" | "B") => {
    const document = documentId === "A" ? documentA : documentB;
    
    if (!document.content.trim()) {
      alert("Please enter some text before checking for AI.");
      return;
    }

    setCurrentAICheckDocument(documentId);
    setAIDetectionModalOpen(true);
    setIsAICheckLoading(true);
    setAIDetectionResult(undefined);

    try {
      const result = await checkForAI(document);
      setAIDetectionResult(result);
      
      // Update the document analysis with AI detection results if it exists
      if (documentId === "A" && analysisA) {
        setAnalysisA({
          ...analysisA,
          aiDetection: result
        });
      } else if (documentId === "B" && analysisB) {
        setAnalysisB({
          ...analysisB,
          aiDetection: result
        });
      }
    } catch (error) {
      console.error("Error checking for AI:", error);
    } finally {
      setIsAICheckLoading(false);
    }
  };

  // Handler for case assessment - REAL-TIME STREAMING
  const handleCaseAssessment = async () => {
    if (!documentA.content.trim()) {
      alert("Please enter some text to assess how well it makes its case.");
      return;
    }

    // Check if the selected provider is available (map ZHI names to original API names)
    const apiKeyMapping: Record<string, string> = {
      'zhi1': 'openai',
      'zhi2': 'anthropic', 
      'zhi3': 'deepseek',
      'zhi4': 'perplexity',
      'zhi5': 'grok'
    };
    const actualApiKey = apiKeyMapping[selectedProvider] || selectedProvider;
    if (selectedProvider !== "all" && !apiStatus[actualApiKey as keyof typeof apiStatus]) {
      alert(`The ${selectedProvider} API key is not configured or is invalid. Please select a different provider or ensure the API key is properly set.`);
      return;
    }

    // Reset any previous streaming state and clear previous analysis results
    setIsStreaming(false);
    setStreamingContent('');
    setAnalysisA(null); // Clear previous intelligence analysis
    setShowResults(true); // Ensure results section is visible
    
    // Start REAL-TIME streaming for case assessment
    setIsStreaming(true);
    setIsCaseAssessmentLoading(true);
    setCaseAssessmentResult(null);

    try {
      const provider = selectedProvider === "all" ? "zhi1" : selectedProvider;
      
      const response = await fetch('/api/case-assessment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: documentA.content,
          provider: provider,
          context: documentA.context
        }),
      });

      if (!response.ok) {
        throw new Error(`Case assessment failed: ${response.statusText}`);
      }

      // REAL-TIME STREAMING: Read response token by token
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let fullResponse = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        fullResponse += chunk;
        setStreamingContent(fullResponse); // Show each token as it arrives
      }

      // Parse the case assessment response to extract scores
      const parseScores = (text: string) => {
        const extractScore = (pattern: string): number => {
          // Try multiple patterns to extract scores
          const patterns = [
            new RegExp(`${pattern}[:\\s]*(\\d+)(?:/100)?`, 'i'),
            new RegExp(`${pattern}.*?(\\d+)/100`, 'i'),
            new RegExp(`${pattern}.*?Score[:\\s]*(\\d+)`, 'i'),
            new RegExp(`${pattern}.*?(\\d+)`, 'i')
          ];
          
          for (const regex of patterns) {
            const match = text.match(regex);
            if (match && match[1]) {
              const score = parseInt(match[1]);
              if (score >= 0 && score <= 100) {
                return score;
              }
            }
          }
          
          // Fallback: compute score based on text analysis
          return computeFallbackScore(pattern, text);
        };

        // Fallback scoring based on text analysis
        const computeFallbackScore = (category: string, fullText: string): number => {
          const text = fullText.toLowerCase();
          let score = 50; // Base score
          
          // Look for positive indicators
          const positiveWords = ['strong', 'effective', 'clear', 'compelling', 'convincing', 'well-structured', 'logical', 'coherent'];
          const negativeWords = ['weak', 'unclear', 'confusing', 'illogical', 'lacks', 'missing', 'problematic'];
          
          positiveWords.forEach(word => {
            if (text.includes(word)) score += 8;
          });
          
          negativeWords.forEach(word => {
            if (text.includes(word)) score -= 8;
          });
          
          // Category-specific adjustments
          if (category.includes('PROOF') && text.includes('evidence')) score += 10;
          if (category.includes('CREDIBILITY') && text.includes('reliable')) score += 10;
          if (category.includes('WRITING') && text.includes('readable')) score += 10;
          
          return Math.max(0, Math.min(100, score));
        };

        return {
          proofEffectiveness: extractScore('PROOF EFFECTIVENESS'),
          claimCredibility: extractScore('CLAIM CREDIBILITY'),
          nonTriviality: extractScore('NON-TRIVIALITY'),
          proofQuality: extractScore('PROOF QUALITY'),
          functionalWriting: extractScore('FUNCTIONAL WRITING'),
          overallCaseScore: extractScore('OVERALL CASE SCORE'),
          detailedAssessment: fullResponse
        };
      };

      console.log('FULL AI RESPONSE FOR DEBUGGING:', fullResponse);
      const caseAssessmentData = parseScores(fullResponse);
      console.log('PARSED SCORES:', caseAssessmentData);
      setCaseAssessmentResult(caseAssessmentData);
      
      // CREATE CASE ASSESSMENT ONLY RESULT - NOT INTELLIGENCE ASSESSMENT  
      setAnalysisA({
        id: Date.now(),
        formattedReport: "", // Empty so it doesn't show in intelligence section
        overallScore: undefined, // No intelligence score
        provider: provider,
        analysis: "",
        summary: "",
        caseAssessment: caseAssessmentData,
        analysisType: "case_assessment", // Flag to identify this as case assessment
      });
      
      // NO POPUP - Results are now in main report only
      
    } catch (error) {
      console.error("Error performing case assessment:", error);
      alert("Failed to assess document case. Please try again.");
    } finally {
      setIsCaseAssessmentLoading(false);
      setIsStreaming(false);
    }
  };

  // Handler for document comparison
  const handleDocumentComparison = async () => {
    if (!documentA.content.trim() || !documentB.content.trim()) {
      alert("Please enter text in both documents to compare them.");
      return;
    }

    // Check if the selected provider is available (map ZHI names to original API names)
    const apiKeyMapping: Record<string, string> = {
      'zhi1': 'openai',
      'zhi2': 'anthropic', 
      'zhi3': 'deepseek',
      'zhi4': 'perplexity',
      'zhi5': 'grok'
    };
    const actualApiKey = apiKeyMapping[selectedProvider] || selectedProvider;
    if (selectedProvider !== "all" && !apiStatus[actualApiKey as keyof typeof apiStatus]) {
      alert(`The ${selectedProvider} API key is not configured or is invalid. Please select a different provider or ensure the API key is properly set.`);
      return;
    }

    setIsComparisonLoading(true);
    setComparisonResult(null);

    try {
      const provider = selectedProvider === "all" ? "zhi1" : selectedProvider;
      
      const response = await fetch('/api/compare', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          documentA: documentA.content,
          documentB: documentB.content,
          provider: provider
        }),
      });

      if (!response.ok) {
        throw new Error(`Document comparison failed: ${response.statusText}`);
      }

      const data = await response.json();
      setComparisonResult(data);
      setComparisonModalOpen(true);
      
    } catch (error) {
      console.error("Error comparing documents:", error);
      alert(`Document comparison failed: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setIsComparisonLoading(false);
    }
  };

  // Handler for fiction assessment - REAL-TIME STREAMING
  const handleFictionAssessment = async (documentId: "A" | "B") => {
    const document = documentId === "A" ? documentA : documentB;
    if (!document.content.trim()) {
      alert(`Please enter some text in Document ${documentId}.`);
      return;
    }

    // Check if the selected provider is available (map ZHI names to original API names)
    const apiKeyMapping: Record<string, string> = {
      'zhi1': 'openai',
      'zhi2': 'anthropic', 
      'zhi3': 'deepseek',
      'zhi4': 'perplexity',
      'zhi5': 'grok'
    };
    const actualApiKey = apiKeyMapping[selectedProvider] || selectedProvider;
    if (selectedProvider !== "all" && !apiStatus[actualApiKey as keyof typeof apiStatus]) {
      alert(`The ${selectedProvider} API key is not configured or is invalid. Please select a different provider or ensure the API key is properly set.`);
      return;
    }

    // Reset any previous streaming state
    setIsStreaming(false);
    setStreamingContent('');
    
    // Start REAL-TIME streaming for fiction assessment
    setIsStreaming(true);
    setIsFictionAssessmentLoading(true);
    setFictionAssessmentResult(null);

    try {
      const provider = selectedProvider === "all" ? "zhi1" : selectedProvider;
      
      const response = await fetch('/api/fiction-assessment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: document.content,
          provider: provider
        }),
      });

      if (!response.ok) {
        throw new Error(`Fiction assessment failed: ${response.statusText}`);
      }

      // REAL-TIME STREAMING: Read response token by token
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let fullResponse = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        fullResponse += chunk;
        setStreamingContent(fullResponse); // Show each token as it arrives
      }

      // Parse the fiction assessment response to extract scores
      const parseFictionScores = (text: string) => {
        const extractScore = (pattern: string): number => {
          const regex = new RegExp(`${pattern}[:\\s]*(\\d+)(?:/100)?`, 'i');
          const match = text.match(regex);
          return match ? parseInt(match[1]) : 0;
        };

        return {
          worldCoherence: extractScore('WORLD COHERENCE'),
          emotionalPlausibility: extractScore('EMOTIONAL PLAUSIBILITY'),
          thematicDepth: extractScore('THEMATIC DEPTH'),
          narrativeStructure: extractScore('NARRATIVE STRUCTURE'),
          proseControl: extractScore('PROSE CONTROL'),
          overallFictionScore: extractScore('OVERALL FICTION SCORE'),
          detailedAssessment: fullResponse
        };
      };

      const fictionAssessmentData = parseFictionScores(fullResponse);
      setFictionAssessmentResult(fictionAssessmentData);
      setCurrentFictionDocument(documentId);
      
      // CREATE FICTION ASSESSMENT ONLY RESULT - NOT INTELLIGENCE ASSESSMENT  
      setAnalysisA({
        id: Date.now(),
        formattedReport: "", // Empty so it doesn't show in intelligence section
        overallScore: undefined, // No intelligence score
        provider: provider,
        analysis: "",
        summary: "",
        fictionAssessment: fictionAssessmentData,
        analysisType: "fiction_assessment", // Flag to identify this as fiction assessment
      });
      
      // NO POPUP - Results are now in main report only
      
    } catch (error) {
      console.error("Error performing fiction assessment:", error);
      alert(`Fiction assessment with ${selectedProvider} failed: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setIsFictionAssessmentLoading(false);
      setIsStreaming(false);
      setStreamingContent(''); // Clean up streaming content
    }
  };

  // Handler for fiction comparison
  const handleFictionComparison = () => {
    if (!documentA.content.trim() || !documentB.content.trim()) {
      alert("Please enter text in both documents to compare them.");
      return;
    }

    setFictionComparisonModalOpen(true);
  };

  // Handler for maximize intelligence
  const handleMaximizeIntelligence = async () => {
    if (!documentA.content.trim()) {
      alert("Please provide document content first.");
      return;
    }

    setIsMaximizeIntelligenceLoading(true);
    try {
      const instructionsToUse = customInstructions.trim() || defaultInstructions;
      
      const response = await fetch('/api/intelligent-rewrite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          originalText: documentA.content,
          customInstructions: instructionsToUse,
          provider: selectedProvider === "all" ? "zhi1" : selectedProvider,
          useExternalKnowledge: useExternalKnowledge
        }),
      });

      if (!response.ok) {
        throw new Error(`Rewrite failed: ${response.statusText}`);
      }

      const data = await response.json();
      setRewriteResult(data.result?.rewrittenText || data.rewrittenText || "No rewrite result returned");
      
      // Store the complete result data and show results modal
      setRewriteResultData(data.result);
      setRewriteResultsModalOpen(true);
      
    } catch (error) {
      console.error('Maximize intelligence error:', error);
      alert(error instanceof Error ? error.message : "Failed to maximize intelligence. Please try again.");
    } finally {
      setIsMaximizeIntelligenceLoading(false);
      setMaximizeIntelligenceModalOpen(false);
    }
  };


  // Handler for downloading rewrite results

  const handleDownloadRewrite = () => {
    if (!rewriteResultData) return;
    
    const content = `INTELLIGENT REWRITE RESULTS
${"=".repeat(50)}

ORIGINAL TEXT:
${rewriteResultData.originalText}

REWRITTEN TEXT:
${rewriteResultData.rewrittenText}

SCORE IMPROVEMENT:
Original Score: ${rewriteResultData.originalScore}/100
Rewritten Score: ${rewriteResultData.rewrittenScore}/100
Improvement: ${rewriteResultData.rewrittenScore - rewriteResultData.originalScore} points

REWRITE REPORT:
${rewriteResultData.rewriteReport || "No detailed report available"}

Provider: ${rewriteResultData.provider}
Instructions: ${rewriteResultData.instructions}

Generated on: ${new Date().toLocaleString()}`;
    
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `intelligent-rewrite-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleUseRewrittenText = () => {
    if (rewriteResultData?.rewrittenText) {
      setDocumentA(prev => ({ ...prev, content: rewriteResultData.rewrittenText }));
      setRewriteResultsModalOpen(false);
    }
  };

  const handleKeepOriginalText = () => {
    setRewriteResultsModalOpen(false);
  };

  // Handler for sending rewritten text to intelligence analysis
  const handleSendToIntelligenceAnalysis = () => {
    if (rewriteResultData?.rewrittenText) {
      setDocumentA(prev => ({ ...prev, content: rewriteResultData.rewrittenText }));
      setRewriteResultsModalOpen(false);
      // Optional: Auto-trigger intelligence analysis
      // setTimeout(() => handleCognitiveQuick(), 100);
    }
  };

  // Handler for analyzing documents - FIXED MAIN ANALYSIS
  // Helper function to get content for analysis based on chunk selection
  const getContentForAnalysis = (document: DocumentInputType): string => {
    // If no chunks or no chunks selected, use full content
    if (!document.chunks || !document.selectedChunkIds || document.selectedChunkIds.length === 0) {
      return document.content;
    }
    
    // Combine selected chunks
    const selectedChunks = document.chunks.filter(chunk => 
      document.selectedChunkIds!.includes(chunk.id)
    );
    
    return selectedChunks.map(chunk => chunk.content).join('\n\n');
  };

  const handleAnalyze = async () => {
    const contentA = getContentForAnalysis(documentA);
    const contentB = getContentForAnalysis(documentB);
    
    if (!contentA.trim()) {
      const message = documentA.chunks && documentA.chunks.length > 1 
        ? "Please select at least one chunk to analyze from Document A."
        : "Please enter some text in Document A.";
      alert(message);
      return;
    }

    if (mode === "compare" && !contentB.trim()) {
      const message = documentB.chunks && documentB.chunks.length > 1 
        ? "Please select at least one chunk to analyze from Document B."
        : "Please enter some text in Document B for comparison.";
      alert(message);
      return;
    }
    
    // Check if the selected provider is available (map ZHI names to original API names)
    const apiKeyMapping: Record<string, string> = {
      'zhi1': 'openai',
      'zhi2': 'anthropic', 
      'zhi3': 'deepseek',
      'zhi4': 'perplexity',
      'zhi5': 'grok'
    };
    const actualApiKey = apiKeyMapping[selectedProvider] || selectedProvider;
    if (selectedProvider !== "all" && !apiStatus[actualApiKey as keyof typeof apiStatus]) {
      alert(`The ${selectedProvider} API key is not configured or is invalid. Please select a different provider or ensure the API key is properly set.`);
      return;
    }

    // FIXED: Use proper analysis for single document mode
    if (mode === "single") {
      setShowResults(true);
      setIsAnalysisLoading(true);
      
      try {
        const provider = selectedProvider === "all" ? "zhi1" : selectedProvider;
        if (analysisType === "quick") {
          // Quick analysis - regular API call
          const response = await fetch('/api/cognitive-quick', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: contentA, provider: provider }),
          });

          if (!response.ok) {
            throw new Error(`Analysis failed: ${response.statusText}`);
          }

          const data = await response.json();
          setAnalysisA(data.analysis || data.result);
        } else {
          // Reset any previous streaming state
          setIsStreaming(false);
          setStreamingContent('');
          
          // Comprehensive analysis - streaming
          setIsStreaming(true);
          
          const response = await fetch('/api/stream-comprehensive', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: contentA, provider: provider }),
          });

          if (!response.ok) {
            throw new Error(`Streaming failed: ${response.statusText}`);
          }

          const reader = response.body?.getReader();
          const decoder = new TextDecoder();
          let fullContent = '';

          if (reader) {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              const chunk = decoder.decode(value, { stream: true });
              fullContent += chunk;
              setStreamingContent(fullContent);
            }
            
            // Extract actual score from streamed content
            const scoreMatch = fullContent.match(/FINAL SCORE:\s*(\d+)\/100/i) || 
                              fullContent.match(/Final Score:\s*(\d+)\/100/i) ||
                              fullContent.match(/Score:\s*(\d+)\/100/i);
            const actualScore = scoreMatch ? parseInt(scoreMatch[1]) : 0;
            
            // Convert streaming content to analysis format
            setAnalysisA({
              id: Date.now(),
              formattedReport: fullContent,
              overallScore: actualScore, // Use actual AI-generated score
              provider: provider
            });
          }
          
          setIsStreaming(false);
          setStreamingContent(''); // Clean up streaming content
        }
        
      } catch (error) {
        console.error("Error analyzing document:", error);
        alert(`Analysis with ${selectedProvider} failed: ${error instanceof Error ? error.message : "Unknown error"}`);
      } finally {
        setIsAnalysisLoading(false);
      }
      return;
    }
    
    // Regular analysis logic for comparison mode
    setShowResults(true);
    setIsAnalysisLoading(true);
    
    try {
      // Two-document mode: use existing comparison logic for now
      if (analysisType === "quick") {
        const provider = selectedProvider === "all" ? "zhi1" : selectedProvider;
        
        const response = await fetch('/api/quick-compare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            documentA: contentA,
            documentB: contentB,
            provider: provider
          }),
        });

        if (!response.ok) {
          throw new Error(`Quick comparison failed: ${response.statusText}`);
        }

        const data = await response.json();
        setAnalysisA(data.analysisA);
        setAnalysisB(data.analysisB);
        setComparison(data.comparison);
      } else {
        // Use the comprehensive comparison (existing logic)
        console.log(`Comparing with ${selectedProvider}...`);
        // Create temporary documents with the selected content for comparison
        const tempDocA = { ...documentA, content: contentA };
        const tempDocB = { ...documentB, content: contentB };
        const results = await compareDocuments(tempDocA, tempDocB, selectedProvider);
        setAnalysisA(results.analysisA);
        setAnalysisB(results.analysisB);
        setComparison(results.comparison);
      }
    } catch (error) {
      console.error("Error comparing documents:", error);
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      alert(`Comparison with ${selectedProvider} failed: ${errorMessage}\n\nPlease verify that the ${selectedProvider} API key is correctly configured.`);
    } finally {
      setIsAnalysisLoading(false);
    }
  };
  

  
  // Handler for resetting the entire analysis
  const handleReset = () => {
    // Clear document inputs
    setDocumentA({ content: "" });
    setDocumentB({ content: "" });
    
    // Clear analysis results
    setAnalysisA(null);
    setAnalysisB(null);
    setComparison(null);
    
    // Clear streaming content
    setIsStreaming(false);
    setStreamingContent('');
    
    // Reset UI states
    setShowResults(false);
    setIsAnalysisLoading(false);
    setIsAICheckLoading(false);
    setAIDetectionResult(undefined);
    
    // Reset to single mode
    setMode("single");
    
    // Scroll to top
    window.scrollTo(0, 0);
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      {/* External Knowledge Toggle - KEPT VISIBLE PER USER REQUEST */}
      <div className="flex justify-end mb-4">
        <div className="flex items-center gap-3 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border-2 border-blue-300 dark:border-blue-700 shadow-md min-w-[320px]">
          <div className="flex-1">
            <div className="text-sm font-bold text-blue-900 dark:text-blue-100">
              USE ZHI DATABASE
            </div>
            <div className="text-xs text-blue-700 dark:text-blue-300 mt-0.5">
              AnalyticPhilosophy.net
            </div>
          </div>
          <Switch
            id="global-external-knowledge"
            checked={useExternalKnowledge}
            onCheckedChange={setUseExternalKnowledge}
            className="data-[state=checked]:bg-blue-600"
            data-testid="toggle-external-knowledge-global"
          />
        </div>
      </div>

      {/* INTELLIGENCE ANALYSIS TOOL - HIDDEN BY USER REQUEST */}
      <div className="hidden">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <h1 className="text-3xl font-bold text-gray-800 mb-2">Intelligence Analysis Tool</h1>
            <p className="text-gray-600">Analyze, compare, and enhance writing samples with AI-powered intelligence evaluation</p>
          </div>
        </div>
      </header>

      {/* Analysis Mode Selector */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Analysis Settings</h2>
        <div className="flex flex-wrap gap-8 items-center">
          <ModeToggle mode={mode} setMode={setMode} />
          
          {/* Fiction Assessment Button */}
          <div className="border p-4 rounded-lg bg-white shadow-sm">
            <h3 className="text-lg font-medium text-gray-800 mb-3">Fiction Analysis</h3>
            <Button
              onClick={() => setFictionPopupOpen(true)}
              variant="outline"
              className="flex items-center gap-2 bg-purple-50 hover:bg-purple-100 border-purple-200 text-purple-700"
              data-testid="button-fiction-assessment"
            >
              <BookOpen className="w-4 h-4" />
              Assess Fiction
            </Button>
            <p className="text-xs text-gray-500 mt-2">
              Analyze literary fiction with specialized assessment criteria
            </p>
          </div>
          
          {/* Analysis Mode Toggle */}
          <div className="border p-4 rounded-lg bg-white shadow-sm">
            <h3 className="text-lg font-medium text-gray-800 mb-3">Analysis Mode</h3>
            <div className="flex gap-3">
              <Button
                onClick={() => setAnalysisType("quick")}
                variant={analysisType === "quick" ? "default" : "outline"}
                className="flex items-center gap-2"
              >
                <Zap className="h-4 w-4" />
                Quick Analysis
              </Button>
              <Button
                onClick={() => setAnalysisType("comprehensive")}
                variant={analysisType === "comprehensive" ? "default" : "outline"}
                className="flex items-center gap-2"
              >
                <Clock className="h-4 w-4" />
                Comprehensive
                <Badge variant="secondary" className="ml-1 text-xs">
                  ~3 min
                </Badge>
              </Button>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              {analysisType === "quick" 
                ? "Fast assessment focusing on core intelligence indicators"
                : "In-depth 4-phase evaluation protocol (takes up to 3 minutes)"
              }
            </p>
          </div>
          
          <div className="border p-4 rounded-lg bg-white shadow-sm mt-2 md:mt-0">
            <h3 className="text-lg font-medium text-gray-800 mb-3">Choose Your AI Provider</h3>
            <ProviderSelector 
              selectedProvider={selectedProvider}
              onProviderChange={setSelectedProvider}
              label="AI Provider"
              apiStatus={apiStatus}
              className="mb-3"
            />
            
            {/* API Status Indicators */}
            <div className="mt-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Provider Status:</h4>
              <div className="flex flex-wrap gap-2">
                <div className={`px-3 py-1 rounded-full text-xs font-medium flex items-center ${apiStatus.openai ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                  <span className={`h-2 w-2 rounded-full mr-1.5 ${apiStatus.openai ? 'bg-green-500' : 'bg-red-500'}`}></span>
                  ZHI 1: {apiStatus.openai ? 'Active' : 'Inactive'}
                </div>
                <div className={`px-3 py-1 rounded-full text-xs font-medium flex items-center ${apiStatus.anthropic ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                  <span className={`h-2 w-2 rounded-full mr-1.5 ${apiStatus.anthropic ? 'bg-green-500' : 'bg-red-500'}`}></span>
                  ZHI 2: {apiStatus.anthropic ? 'Active' : 'Inactive'}
                </div>
                <div className={`px-3 py-1 rounded-full text-xs font-medium flex items-center ${apiStatus.perplexity ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                  <span className={`h-2 w-2 rounded-full mr-1.5 ${apiStatus.perplexity ? 'bg-green-500' : 'bg-red-500'}`}></span>
                  ZHI 3: {apiStatus.perplexity ? 'Active' : 'Inactive'}
                </div>
              </div>
              <p className="mt-2 text-xs text-gray-500">All API providers are active and ready to use. Each offers different analysis capabilities.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Document Input Section */}
      <div className="mb-8">
        {/* Document A */}
        <DocumentInput
          id="A"
          document={documentA}
          setDocument={setDocumentA}
          onCheckAI={() => handleCheckAI("A")}
        />

        {/* Document B (shown only in compare mode) */}
        {mode === "compare" && (
          <DocumentInput
            id="B"
            document={documentB}
            setDocument={setDocumentB}
            onCheckAI={() => handleCheckAI("B")}
          />
        )}

        {/* Analysis Options */}
        {mode === "single" ? (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-3 text-center">Choose Analysis Type</h3>
            <p className="text-sm text-gray-600 mb-4 text-center">Run any or all analyses on your document - no need to re-upload text</p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Intelligence Analysis */}
              <div className="text-center">
                <Button
                  onClick={handleAnalyze}
                  className="w-full px-4 py-6 bg-blue-600 text-white rounded-md font-semibold hover:bg-blue-700 flex flex-col items-center min-h-[100px]"
                  disabled={isAnalysisLoading || !documentA.content.trim()}
                >
                  <Brain className="h-6 w-6 mb-2" />
                  <span className="text-sm">
                    {isAnalysisLoading ? "Analyzing..." : "Intelligence Analysis"}
                  </span>
                </Button>
                <p className="text-xs text-gray-500 mt-2">Assess cognitive abilities and intelligence</p>
              </div>

              {/* Case Assessment */}
              <div className="text-center">
                <Button
                  onClick={handleCaseAssessment}
                  className="w-full px-4 py-6 bg-purple-600 text-white rounded-md font-semibold hover:bg-purple-700 flex flex-col items-center min-h-[100px]"
                  disabled={isCaseAssessmentLoading || !documentA.content.trim()}
                >
                  <FileEdit className="h-6 w-6 mb-2" />
                  <span className="text-sm text-center leading-tight">
                    {isCaseAssessmentLoading ? "Assessing..." : "Case Assessment"}
                  </span>
                </Button>
                <p className="text-xs text-gray-500 mt-2">How well does it make its case?</p>
              </div>

              {/* Fiction Assessment */}
              <div className="text-center">
                <Button
                  onClick={() => handleFictionAssessment("A")}
                  className="w-full px-4 py-6 bg-orange-600 text-white rounded-md font-semibold hover:bg-orange-700 flex flex-col items-center min-h-[100px]"
                  disabled={!documentA.content.trim() || isFictionAssessmentLoading}
                >
                  {isFictionAssessmentLoading ? (
                    <Loader2 className="h-6 w-6 mb-2 animate-spin" />
                  ) : (
                    <FileEdit className="h-6 w-6 mb-2" />
                  )}
                  <span className="text-sm">
                    {isFictionAssessmentLoading ? "Assessing..." : "Fiction Analysis"}
                  </span>
                </Button>
                <p className="text-xs text-gray-500 mt-2">Evaluate creative writing quality</p>
              </div>

              {/* Maximize Intelligence */}
              <div className="text-center">
                <Button
                  onClick={() => setMaximizeIntelligenceModalOpen(true)}
                  className="w-full px-4 py-6 bg-emerald-600 text-white rounded-md font-semibold hover:bg-emerald-700 flex flex-col items-center min-h-[100px]"
                  disabled={!documentA.content.trim()}
                  data-testid="button-maximize-intelligence"
                >
                  <Sparkles className="h-6 w-6 mb-2" />
                  <span className="text-sm">Maximize Intelligence</span>
                </Button>
                <p className="text-xs text-gray-500 mt-2">Rewrite to boost intelligence score</p>
              </div>
            </div>
            
            {/* Clear All Button */}
            <div className="mt-6 text-center">
              <Button
                onClick={handleReset}
                variant="outline"
                className="px-6 py-2 border-red-300 text-red-600 hover:bg-red-50 hover:border-red-400 flex items-center mx-auto"
                disabled={isAnalysisLoading}
              >
                <Trash2 className="h-4 w-4 mr-2" />
                <span>New Analysis / Clear All</span>
              </Button>
            </div>
          </div>
        ) : (
          /* Comparison Mode Buttons */
          <div className="flex justify-center gap-4 flex-wrap">
            <Button
              onClick={handleAnalyze}
              className="px-6 py-3 bg-blue-600 text-white rounded-md font-semibold hover:bg-blue-700 flex items-center"
              disabled={isAnalysisLoading}
            >
              <Brain className="h-5 w-5 mr-2" />
              <span>
                {isAnalysisLoading ? "Analyzing..." : "Analyze Both Documents"}
              </span>
            </Button>
            
            <Button
              onClick={handleDocumentComparison}
              className="px-6 py-3 bg-purple-600 text-white rounded-md font-semibold hover:bg-purple-700 flex items-center"
              disabled={!documentA.content.trim() || !documentB.content.trim() || isComparisonLoading}
            >
              <FileEdit className="h-5 w-5 mr-2" />
              <span>
                {isComparisonLoading ? "Comparing..." : "Which One Makes Its Case Better?"}
              </span>
            </Button>
            
            <Button
              onClick={handleFictionComparison}
              className="px-6 py-3 bg-amber-600 text-white rounded-md font-semibold hover:bg-amber-700 flex items-center"
              disabled={!documentA.content.trim() || !documentB.content.trim()}
            >
              <FileEdit className="h-5 w-5 mr-2" />
              <span>Compare Fiction</span>
            </Button>            
            <Button
              onClick={handleReset}
              className="px-6 py-3 bg-red-600 text-white rounded-md font-semibold hover:bg-red-700 flex items-center"
              disabled={isAnalysisLoading}
            >
              <Trash2 className="h-5 w-5 mr-2" />
              <span>New Analysis / Clear All</span>
            </Button>
          </div>
        )}
      </div>

      {/* AI Detection Modal */}
      <AIDetectionModal
        isOpen={aiDetectionModalOpen}
        onClose={() => setAIDetectionModalOpen(false)}
        result={aiDetectionResult}
        isLoading={isAICheckLoading}
      />

      {/* Results Section */}
      {showResults && (
        <div id="resultsSection">
          {/* Loading Indicator */}
          {isAnalysisLoading ? (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent mb-4"></div>
              <p className="text-gray-600">Analyzing document content...</p>
            </div>
          ) : (
            <div>
              {/* Document A Results */}
              {analysisA && <DocumentResults id="A" analysis={analysisA} originalDocument={documentA} analysisMode={analysisType} onSendToHumanizer={handleSendToHumanizer} onSendToIntelligence={handleSendToIntelligence} onSendToChat={handleSendToChat} />}

              {/* Document B Results (only in compare mode) */}
              {mode === "compare" && analysisB && (
                <DocumentResults id="B" analysis={analysisB} originalDocument={documentB} analysisMode={analysisType} onSendToHumanizer={handleSendToHumanizer} onSendToIntelligence={handleSendToIntelligence} onSendToChat={handleSendToChat} />
              )}

              {/* Comparative Results (only in compare mode) */}
              {mode === "compare" && comparison && analysisA && analysisB && (
                <ComparativeResults
                  analysisA={analysisA}
                  analysisB={analysisB}
                  comparison={comparison}
                  onSendToHumanizer={handleSendToHumanizer}
                  onSendToIntelligence={handleSendToIntelligence}
                  onSendToChat={handleSendToChat}
                  documentAText={documentA?.content}
                  documentBText={documentB?.content}
                />
              )}
              

              
              {/* Semantic Density Analysis - always shown when there's text */}
              {mode === "single" && documentA.content.trim() && (
                <div className="bg-white rounded-lg shadow-md p-6 mb-8 mt-8">
                  <SemanticDensityAnalyzer text={documentA.content} />
                </div>
              )}
            </div>
          )}
        </div>
      )}



      {/* Case Assessment Modal - REMOVED: Results now show in main report only */}

      {/* Document Comparison Modal */}
      <DocumentComparisonModal
        isOpen={comparisonModalOpen}
        onClose={() => setComparisonModalOpen(false)}
        result={comparisonResult}
        isLoading={isComparisonLoading}
      />

      {/* AI Detection Modal */}
      <AIDetectionModal
        isOpen={aiDetectionModalOpen}
        onClose={() => setAIDetectionModalOpen(false)}
        result={aiDetectionResult}
        isLoading={isAICheckLoading}
      />

      {/* Fiction Assessment Modal - REMOVED: Results now show in main report only */}

      {/* Fiction Comparison Modal */}
      <FictionComparisonModal
        isOpen={fictionComparisonModalOpen}
        onClose={() => setFictionComparisonModalOpen(false)}
        documentA={{
          content: documentA.content,
          title: documentA.filename || "Document A"
        }}
        documentB={{
          content: documentB.content,
          title: documentB.filename || "Document B"
        }}
      />



      {/* Inline Streaming Results Area */}
      {(isStreaming || streamingContent) && (
        <div className="mx-4 mb-6">
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-6">
            <div className="flex items-center gap-2 mb-4">
              <Brain className="h-5 w-5 text-blue-600" />
              <h3 className="text-lg font-semibold text-blue-900">
                🎯 Intelligence Analysis
                {isStreaming && <span className="ml-2 text-sm font-normal text-blue-600">Streaming...</span>}
              </h3>
            </div>
            <div className="bg-white rounded-md p-4 border border-blue-100 min-h-[200px]">
              <div className="prose prose-sm max-w-none text-gray-800 whitespace-pre-wrap font-mono text-sm leading-relaxed">
                {streamingContent}
                {isStreaming && <span className="inline-block w-2 h-4 bg-blue-500 animate-pulse ml-1">|</span>}
              </div>
            </div>
            {streamingContent && !isStreaming && (
              <div className="mt-4 flex justify-end">
                <Button 
                  onClick={() => setStreamingContent('')}
                  variant="outline"
                  size="sm"
                  className="text-gray-600 hover:text-gray-800"
                >
                  New Analysis
                </Button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Maximize Intelligence Modal */}
      <Dialog open={maximizeIntelligenceModalOpen} onOpenChange={setMaximizeIntelligenceModalOpen}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-emerald-600" />
              Maximize Intelligence
            </DialogTitle>
            <DialogDescription>
              Customize rewrite instructions to maximize intelligence scores, or use our default optimization criteria.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            {/* External Knowledge Toggle */}
            <div className="flex items-center justify-between p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="flex-1">
                <Label htmlFor="external-knowledge-main" className="text-sm font-medium text-blue-900 dark:text-blue-100">
                  USE ZHI DATABASE (AnalyticPhilosophy.net)
                </Label>
                <p className="text-xs text-blue-700 dark:text-blue-300 mt-1">
                  When enabled, MAXINTEL fetches research passages and citations from the Zhi knowledge base
                </p>
              </div>
              <Switch
                id="external-knowledge-main"
                checked={useExternalKnowledge}
                onCheckedChange={setUseExternalKnowledge}
                disabled={isMaximizeIntelligenceLoading}
                data-testid="toggle-external-knowledge-main"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 mb-2 block">
                Custom Instructions (optional)
              </label>
              <Textarea
                value={customInstructions}
                onChange={(e) => setCustomInstructions(e.target.value)}
                placeholder="Enter custom rewrite instructions here. If left empty, default optimization criteria will be used."
                className="min-h-[120px]"
                data-testid="textarea-custom-instructions"
              />
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Default Instructions (used if custom field is empty):</h4>
              <div className="text-xs text-gray-600 max-h-40 overflow-y-auto whitespace-pre-wrap">
                {defaultInstructions}
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setMaximizeIntelligenceModalOpen(false)}
              data-testid="button-cancel-maximize"
            >
              Cancel
            </Button>
            <Button
              onClick={handleMaximizeIntelligence}
              disabled={isMaximizeIntelligenceLoading}
              className="bg-emerald-600 hover:bg-emerald-700"
              data-testid="button-confirm-maximize"
            >
              {isMaximizeIntelligenceLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Rewriting...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Maximize Intelligence
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Intelligent Rewrite Results Modal */}
      <Dialog open={rewriteResultsModalOpen} onOpenChange={setRewriteResultsModalOpen}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="w-6 h-6 text-emerald-600" />
              Intelligent Rewrite Results
            </DialogTitle>
            <DialogDescription>
              Your text has been optimized for maximum intelligence scoring. Review the results below.
            </DialogDescription>
          </DialogHeader>
          
          {rewriteResultData && (
            <div className="space-y-6">
              {/* Score Improvement */}
              <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg">
                <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-2">Score Improvement</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-gray-700 dark:text-gray-300">{rewriteResultData.originalScore}/100</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Original</div>
                  </div>
                  <div className="text-center">
                    <div className="text-emerald-600 dark:text-emerald-400">
                      {rewriteResultData.rewrittenScore > rewriteResultData.originalScore ? "+" : ""}
                      {rewriteResultData.rewrittenScore - rewriteResultData.originalScore}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Change</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">{rewriteResultData.rewrittenScore}/100</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Rewritten</div>
                  </div>
                </div>
              </div>

              {/* Rewritten Text */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold">Rewritten Text</h3>
                  <SendToButton
                    text={rewriteResultData.rewrittenText}
                    onSendToValidator={(text) => setValidatorInputText(text)}
                    size="sm"
                  />
                </div>
                <div className="border rounded-lg p-4 bg-gray-50 dark:bg-gray-800 max-h-60 overflow-y-auto">
                  <p className="whitespace-pre-wrap">{rewriteResultData.rewrittenText}</p>
                </div>
              </div>

              {/* Original Text for comparison */}
              <div>
                <h3 className="font-semibold mb-2">Original Text</h3>
                <div className="border rounded-lg p-4 bg-gray-50 dark:bg-gray-800 max-h-40 overflow-y-auto">
                  <p className="whitespace-pre-wrap text-sm text-gray-600 dark:text-gray-400">{rewriteResultData.originalText}</p>
                </div>
              </div>

              {/* Rewrite Report if available */}
              {rewriteResultData.rewriteReport && (
                <div>
                  <h3 className="font-semibold mb-2">Rewrite Analysis Report</h3>
                  <div className="border rounded-lg p-4 bg-gray-50 dark:bg-gray-800 max-h-40 overflow-y-auto">
                    <p className="whitespace-pre-wrap text-sm">{rewriteResultData.rewriteReport}</p>
                  </div>
                </div>
              )}
            </div>
          )}
          
          <DialogFooter className="flex flex-col sm:flex-row gap-3">
            <Button 
              variant="outline" 
              onClick={handleDownloadRewrite}
              className="flex items-center gap-2"
              data-testid="button-download-rewrite"
            >
              <Download className="w-4 h-4" />
              Download Results
            </Button>
            <Button 
              variant="outline" 
              onClick={handleKeepOriginalText}
              data-testid="button-keep-original"
            >
              Keep Original
            </Button>
            <Button 
              onClick={handleSendToIntelligenceAnalysis}
              className="bg-blue-600 hover:bg-blue-700 text-white"
              data-testid="button-send-to-intelligence"
            >
              <Brain className="w-4 h-4 mr-2" />
              Send to Intelligence Analysis
            </Button>
            <Button 
              onClick={() => {
                if (rewriteResultData?.rewrittenText) {
                  setBoxA(rewriteResultData.rewrittenText);
                  setRewriteResultsModalOpen(false);
                  toast({
                    title: "Text sent to Humanizer",
                    description: "Rewritten text has been sent to the Humanizer input box"
                  });
                }
              }}
              className="bg-purple-600 hover:bg-purple-700 text-white"
              data-testid="button-send-to-humanizer"
            >
              <Shield className="w-4 h-4 mr-2" />
              Send to Humanizer
            </Button>
            <Button 
              onClick={handleUseRewrittenText}
              className="bg-emerald-600 hover:bg-emerald-700 text-white"
              data-testid="button-use-rewritten"
            >
              Use Rewritten Text
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      </div>
      {/* END OF HIDDEN INTELLIGENCE ANALYSIS TOOL */}

      {/* FINANCE PANEL - Financial Modeling Tools */}
      <div className="mt-8 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/10 dark:to-indigo-900/10 p-8 rounded-lg border-2 border-blue-200 dark:border-blue-700">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-blue-900 dark:text-blue-100 mb-3 flex items-center justify-center gap-3">
              <TrendingUp className="w-8 h-8 text-blue-600" />
              Finance Panel
            </h2>
            <p className="text-lg text-gray-700 dark:text-gray-300 mb-2">
              Generate professional financial models with working Excel formulas
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              DCF valuation, LBO analysis, M&A modeling, and integrated 3-statement models
            </p>
            {/* Clear All Button */}
            <div className="mt-4">
              <Button
                onClick={handleFinanceClear}
                variant="outline"
                size="sm"
                className="border-red-300 text-red-600 hover:bg-red-50 dark:border-red-700 dark:text-red-400 dark:hover:bg-red-900/20"
                disabled={financeLoading}
                data-testid="button-finance-clear"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear Finance Panel
              </Button>
            </div>
          </div>

          {/* Input Area */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-semibold text-blue-800 dark:text-blue-200">
                Describe Company Financials & Projections
              </label>
              <label className="cursor-pointer">
                <input
                  type="file"
                  accept=".pdf,.doc,.docx,.txt"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFileUpload(file, setFinanceInputText);
                  }}
                  data-testid="input-finance-upload"
                />
                <Button
                  variant="outline"
                  size="sm"
                  className="border-blue-300 hover:bg-blue-50 dark:hover:bg-blue-900/20"
                  onClick={(e) => {
                    e.preventDefault();
                    (e.currentTarget.previousElementSibling as HTMLInputElement)?.click();
                  }}
                  data-testid="button-finance-upload"
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Upload Document
                </Button>
              </label>
            </div>
            <Textarea
              value={financeInputText}
              onChange={(e) => setFinanceInputText(e.target.value)}
              placeholder="Describe the company financials in natural language. Example: 'Build me a DCF model for CloudTech Inc. Current revenue is $850 million, growing at 35% for years 1-2, 28% in year 3, then 20% in years 4-5. Terminal growth rate of 4%. Current EBITDA margin is 18%, expanding to 32% by year 5. D&A at 6% of revenue, CapEx at 8%, NWC at 12%. Tax rate 21%. WACC is 11.5%. The company has $420M debt and $130M cash with 75M shares outstanding.'"
              className="min-h-[200px] font-mono text-sm"
              data-testid="textarea-finance-input"
            />
          </div>

          {/* AI Model Selection - Always Visible */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-blue-200 dark:border-blue-700 mb-6">
            <label className="block text-sm font-semibold text-blue-800 dark:text-blue-200 mb-2">
              AI Model Selection
            </label>
            <Select value={financeLLMProvider} onValueChange={(value: "zhi1" | "zhi2" | "zhi3" | "zhi4" | "zhi5") => setFinanceLLMProvider(value)}>
              <SelectTrigger className="w-full max-w-xs" data-testid="select-finance-llm-main">
                <SelectValue placeholder="Select AI Model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="zhi5">ZHI 5 - Default</SelectItem>
                <SelectItem value="zhi1">ZHI 1</SelectItem>
                <SelectItem value="zhi2">ZHI 2</SelectItem>
                <SelectItem value="zhi3">ZHI 3</SelectItem>
                <SelectItem value="zhi4">ZHI 4</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Choose which AI model will analyze your financial data and generate the Excel model.
            </p>
          </div>

          {/* Custom Instructions */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-blue-200 dark:border-blue-700 mb-6">
            <label className="block text-sm font-semibold text-blue-800 dark:text-blue-200 mb-2">
              Custom Instructions (Optional)
            </label>
            <Textarea
              value={financeCustomInstructions}
              onChange={(e) => setFinanceCustomInstructions(e.target.value)}
              placeholder="Add any special requirements for the financial model...

Examples:
- Use a 7-year projection period instead of 5
- Add a sensitivity analysis on revenue growth
- Include quarterly breakdown for Year 1
- Use different WACC for each scenario
- Add specific industry comparables"
              className="min-h-[100px] text-sm"
              disabled={financeLoading}
              data-testid="textarea-finance-custom-instructions"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Specify additional analysis, formatting preferences, or special calculations
            </p>
          </div>

          {/* Four Model Type Buttons */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <Button
              onClick={() => {
                setShowFinanceCustomization(prev => financeModelType === "dcf" ? !prev : true);
                setFinanceModelType("dcf");
              }}
              className={`flex flex-col items-center justify-center p-6 h-auto ${
                financeModelType === "dcf" 
                  ? "bg-blue-600 hover:bg-blue-700 text-white" 
                  : "bg-white dark:bg-gray-800 hover:bg-blue-50 dark:hover:bg-blue-900/20 text-blue-700 dark:text-blue-300 border-2 border-blue-300"
              }`}
              disabled={financeLoading}
              data-testid="button-dcf-model"
            >
              <Calculator className="w-6 h-6 mb-2" />
              <span className="font-bold text-lg">DCF Model</span>
              <span className="text-xs mt-1 text-center opacity-80">Discounted Cash Flow</span>
            </Button>

            <Button
              onClick={() => {
                setShowFinanceCustomization(prev => financeModelType === "lbo" ? !prev : true);
                setFinanceModelType("lbo");
              }}
              className={`flex flex-col items-center justify-center p-6 h-auto ${
                financeModelType === "lbo" 
                  ? "bg-purple-600 hover:bg-purple-700 text-white" 
                  : "bg-white dark:bg-gray-800 hover:bg-purple-50 dark:hover:bg-purple-900/20 text-purple-700 dark:text-purple-300 border-2 border-purple-300"
              }`}
              disabled={financeLoading}
              data-testid="button-lbo-model"
            >
              <DollarSign className="w-6 h-6 mb-2" />
              <span className="font-bold text-lg">LBO Model</span>
              <span className="text-xs mt-1 text-center opacity-80">Leveraged Buyout</span>
            </Button>

            <Button
              onClick={() => {
                setShowFinanceCustomization(prev => financeModelType === "ma" ? !prev : true);
                setFinanceModelType("ma");
              }}
              className={`flex flex-col items-center justify-center p-6 h-auto ${
                financeModelType === "ma" 
                  ? "bg-green-600 hover:bg-green-700 text-white" 
                  : "bg-white dark:bg-gray-800 hover:bg-green-50 dark:hover:bg-green-900/20 text-green-700 dark:text-green-300 border-2 border-green-300"
              }`}
              disabled={financeLoading}
              data-testid="button-ma-model"
            >
              <TrendingUp className="w-6 h-6 mb-2" />
              <span className="font-bold text-lg">M&A Model</span>
              <span className="text-xs mt-1 text-center opacity-80">Merger & Acquisition</span>
            </Button>

            <Button
              onClick={() => {
                setShowFinanceCustomization(prev => financeModelType === "threestatement" ? !prev : true);
                setFinanceModelType("threestatement");
              }}
              className={`flex flex-col items-center justify-center p-6 h-auto ${
                financeModelType === "threestatement" 
                  ? "bg-orange-600 hover:bg-orange-700 text-white" 
                  : "bg-white dark:bg-gray-800 hover:bg-orange-50 dark:hover:bg-orange-900/20 text-orange-700 dark:text-orange-300 border-2 border-orange-300"
              }`}
              disabled={financeLoading}
              data-testid="button-3statement-model"
            >
              <FileSpreadsheet className="w-6 h-6 mb-2" />
              <span className="font-bold text-lg">3-Statement</span>
              <span className="text-xs mt-1 text-center opacity-80">Integrated Financials</span>
            </Button>
          </div>

          {/* Customization Panel - Appears when model type is selected */}
          {showFinanceCustomization && financeModelType && (
            <div className="mb-6 p-4 bg-white dark:bg-gray-800 rounded-lg border border-blue-200 dark:border-blue-700">
              <h3 className="text-lg font-semibold text-blue-800 dark:text-blue-200 mb-3">
                {financeModelType === "dcf" && "DCF Model Customization"}
                {financeModelType === "lbo" && "LBO Model Customization"}
                {financeModelType === "ma" && "M&A Model Customization"}
                {financeModelType === "threestatement" && "3-Statement Model Customization"}
              </h3>

              {/* LLM Provider Selector */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Select AI Model for Analysis
                </label>
                <Select value={financeLLMProvider} onValueChange={(value: "zhi1" | "zhi2" | "zhi3" | "zhi4" | "zhi5") => setFinanceLLMProvider(value)}>
                  <SelectTrigger className="w-full max-w-xs" data-testid="select-finance-llm">
                    <SelectValue placeholder="Select AI Model" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="zhi5" data-testid="select-item-zhi5">ZHI 5 - Default</SelectItem>
                    <SelectItem value="zhi1" data-testid="select-item-zhi1">ZHI 1</SelectItem>
                    <SelectItem value="zhi2" data-testid="select-item-zhi2">ZHI 2</SelectItem>
                    <SelectItem value="zhi3" data-testid="select-item-zhi3">ZHI 3</SelectItem>
                    <SelectItem value="zhi4" data-testid="select-item-zhi4">ZHI 4</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Choose which AI model will analyze your financial data and generate the model
                </p>
              </div>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Custom Instructions (Optional)
                </label>
                <Textarea
                  value={financeCustomInstructions}
                  onChange={(e) => setFinanceCustomInstructions(e.target.value)}
                  placeholder={
                    financeModelType === "dcf" 
                      ? "Add specific instructions for DCF model... (e.g., 'Include sensitivity analysis for WACC range 8-14%', 'Add football field chart', 'Use specific terminal multiple instead of perpetuity growth')"
                      : financeModelType === "lbo"
                      ? "Add specific instructions for LBO model... (e.g., 'Model 5-year hold period', 'Include management equity rollover', 'Target 3x MOIC')"
                      : financeModelType === "ma"
                      ? "Add specific instructions for M&A model... (e.g., 'Model accretion/dilution analysis', 'Include synergy assumptions', 'Compare cash vs stock deal')"
                      : "Add specific instructions for 3-statement model... (e.g., 'Include quarterly breakdown', 'Add working capital schedule', 'Model debt covenants')"
                  }
                  className="min-h-[100px]"
                  data-testid="textarea-finance-custom-instructions"
                />
              </div>

              <div className="flex gap-3">
                <Button
                  onClick={() => handleFinanceModelGenerate(financeModelType)}
                  className="bg-blue-600 hover:bg-blue-700 text-white flex items-center"
                  disabled={financeLoading || !financeInputText.trim()}
                  data-testid="button-generate-finance-model"
                >
                  {financeLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      {financeLoadingPhase || "Analyzing..."}
                    </>
                  ) : (
                    <>
                      <Calculator className="w-4 h-4 mr-2" />
                      Analyze & Value
                    </>
                  )}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setShowFinanceCustomization(false)}
                  data-testid="button-cancel-finance-customization"
                >
                  Cancel
                </Button>
              </div>
            </div>
          )}

          {/* DCF Results Display */}
          {financeResult && financeResult.success && financeModelType === "dcf" && financeResult.valuation && (
            <div className="mb-6 space-y-6">
              {/* Header with company name and download button */}
              <div className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 rounded-lg border border-blue-200 dark:border-blue-700">
                <div>
                  <h3 className="text-xl font-bold text-blue-900 dark:text-blue-100">
                    {financeResult.assumptions.companyName} - DCF Valuation
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Analyzed by {financeResult.providerUsed}
                  </p>
                </div>
                <Button
                  onClick={handleFinanceDownloadExcel}
                  className="bg-green-600 hover:bg-green-700 text-white"
                  disabled={financeDownloadLoading}
                  data-testid="button-download-excel"
                >
                  {financeDownloadLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Downloading...
                    </>
                  ) : (
                    <>
                      <Download className="w-4 h-4 mr-2" />
                      Download Excel Model
                    </>
                  )}
                </Button>
              </div>

              {/* Valuation Summary - Three Scenarios */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Bear Case */}
                <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-700">
                  <h4 className="text-lg font-semibold text-red-800 dark:text-red-200 mb-3 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 rotate-180" />
                    Bear Case
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Enterprise Value:</span>
                      <span className="font-semibold text-red-700 dark:text-red-300">
                        ${(financeResult.valuation.bear.enterpriseValue).toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Equity Value:</span>
                      <span className="font-semibold text-red-700 dark:text-red-300">
                        ${(financeResult.valuation.bear.equityValue).toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between border-t border-red-200 dark:border-red-600 pt-2">
                      <span className="text-gray-700 dark:text-gray-300 font-medium">Share Price:</span>
                      <span className="font-bold text-red-800 dark:text-red-200 text-lg">
                        ${(financeResult.valuation.bear.sharePrice).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Base Case */}
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border-2 border-blue-400 dark:border-blue-500">
                  <h4 className="text-lg font-semibold text-blue-800 dark:text-blue-200 mb-3 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Base Case
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Enterprise Value:</span>
                      <span className="font-semibold text-blue-700 dark:text-blue-300">
                        ${(financeResult.valuation.base.enterpriseValue).toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Equity Value:</span>
                      <span className="font-semibold text-blue-700 dark:text-blue-300">
                        ${(financeResult.valuation.base.equityValue).toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between border-t border-blue-300 dark:border-blue-500 pt-2">
                      <span className="text-gray-700 dark:text-gray-300 font-medium">Share Price:</span>
                      <span className="font-bold text-blue-800 dark:text-blue-200 text-xl">
                        ${(financeResult.valuation.base.sharePrice).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Bull Case */}
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-700">
                  <h4 className="text-lg font-semibold text-green-800 dark:text-green-200 mb-3 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Bull Case
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Enterprise Value:</span>
                      <span className="font-semibold text-green-700 dark:text-green-300">
                        ${(financeResult.valuation.bull.enterpriseValue).toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Equity Value:</span>
                      <span className="font-semibold text-green-700 dark:text-green-300">
                        ${(financeResult.valuation.bull.equityValue).toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between border-t border-green-200 dark:border-green-600 pt-2">
                      <span className="text-gray-700 dark:text-gray-300 font-medium">Share Price:</span>
                      <span className="font-bold text-green-800 dark:text-green-200 text-lg">
                        ${(financeResult.valuation.bull.sharePrice).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Key Assumptions Table */}
              <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-blue-200 dark:border-blue-700">
                <h4 className="text-lg font-semibold text-blue-800 dark:text-blue-200 mb-4">
                  Extracted Assumptions
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Base Revenue</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      ${financeResult.assumptions.baseYearRevenue?.toLocaleString()}M
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">WACC</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {(financeResult.assumptions.wacc * 100)?.toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Terminal Growth</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {(financeResult.assumptions.terminalGrowthRate * 100)?.toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Tax Rate</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {(financeResult.assumptions.taxRate * 100)?.toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Base EBITDA Margin</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {(financeResult.assumptions.baseEBITDAMargin * 100)?.toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Target EBITDA Margin</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {(financeResult.assumptions.targetEBITDAMargin * 100)?.toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Net Debt</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      ${(financeResult.assumptions.totalDebt - financeResult.assumptions.cashAndEquivalents)?.toLocaleString()}M
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Shares Outstanding</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {financeResult.assumptions.sharesOutstanding?.toLocaleString()}M
                    </div>
                  </div>
                </div>

                {/* Revenue Growth Rates */}
                <div className="mt-4">
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">Revenue Growth Rates (Y1-Y5)</div>
                  <div className="flex gap-2">
                    {financeResult.assumptions.revenueGrowthRates?.map((rate: number, idx: number) => (
                      <div key={idx} className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 rounded text-sm font-medium text-blue-800 dark:text-blue-200">
                        Y{idx + 1}: {(rate * 100).toFixed(0)}%
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Projections Table */}
              <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-blue-200 dark:border-blue-700 overflow-x-auto">
                <h4 className="text-lg font-semibold text-blue-800 dark:text-blue-200 mb-4">
                  5-Year Projections
                </h4>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-600">
                      <th className="text-left py-2 px-3 text-gray-600 dark:text-gray-400">Metric</th>
                      {financeResult.projections?.years?.map((year: number) => (
                        <th key={year} className="text-right py-2 px-3 text-gray-600 dark:text-gray-400">Year {year}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">Revenue ($M)</td>
                      {financeResult.projections?.revenue?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">EBITDA ($M)</td>
                      {financeResult.projections?.ebitda?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">EBITDA Margin</td>
                      {financeResult.projections?.ebitdaMargin?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          {(val * 100).toFixed(1)}%
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300 font-semibold">Free Cash Flow ($M)</td>
                      {financeResult.projections?.fcf?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-bold text-blue-700 dark:text-blue-300">
                          ${val.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* LBO Results Display */}
          {financeResult && financeResult.success && financeModelType === "lbo" && financeResult.returns && (
            <div className="mb-6 space-y-6">
              {/* Header with company name and download button */}
              <div className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 rounded-lg border border-purple-200 dark:border-purple-700">
                <div>
                  <h3 className="text-xl font-bold text-purple-900 dark:text-purple-100">
                    {financeResult.assumptions?.companyName} - LBO Analysis
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Analyzed by {financeResult.providerUsed}
                  </p>
                </div>
                <Button
                  onClick={handleFinanceDownloadExcel}
                  className="bg-green-600 hover:bg-green-700 text-white"
                  disabled={financeDownloadLoading}
                  data-testid="button-download-lbo-excel"
                >
                  {financeDownloadLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Downloading...
                    </>
                  ) : (
                    <>
                      <Download className="w-4 h-4 mr-2" />
                      Download LBO Excel Model
                    </>
                  )}
                </Button>
              </div>

              {/* Returns Summary - Sponsor and Management */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Sponsor Returns */}
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border-2 border-purple-400 dark:border-purple-500">
                  <h4 className="text-lg font-semibold text-purple-800 dark:text-purple-200 mb-3 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Sponsor Returns
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Initial Equity:</span>
                      <span className="font-semibold text-purple-700 dark:text-purple-300">
                        ${financeResult.returns?.sponsor?.equity?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Exit Proceeds:</span>
                      <span className="font-semibold text-purple-700 dark:text-purple-300">
                        ${financeResult.returns?.sponsor?.exitProceeds?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between border-t border-purple-300 dark:border-purple-500 pt-2">
                      <span className="text-gray-700 dark:text-gray-300 font-medium">MOIC:</span>
                      <span className="font-bold text-purple-800 dark:text-purple-200 text-xl">
                        {financeResult.returns?.sponsor?.moic?.toFixed(1)}x
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-700 dark:text-gray-300 font-medium">IRR:</span>
                      <span className="font-bold text-purple-800 dark:text-purple-200 text-xl">
                        {((financeResult.returns?.sponsor?.irr || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Exit Valuation */}
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-700">
                  <h4 className="text-lg font-semibold text-green-800 dark:text-green-200 mb-3 flex items-center gap-2">
                    <DollarSign className="w-5 h-5" />
                    Exit Valuation
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Exit EBITDA:</span>
                      <span className="font-semibold text-green-700 dark:text-green-300">
                        ${financeResult.exitValuation?.exitEBITDA?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Exit Multiple:</span>
                      <span className="font-semibold text-green-700 dark:text-green-300">
                        {financeResult.exitValuation?.exitMultiple?.toFixed(1)}x
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Enterprise Value:</span>
                      <span className="font-semibold text-green-700 dark:text-green-300">
                        ${financeResult.exitValuation?.exitEV?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between border-t border-green-200 dark:border-green-600 pt-2">
                      <span className="text-gray-700 dark:text-gray-300 font-medium">Exit Equity Value:</span>
                      <span className="font-bold text-green-800 dark:text-green-200 text-lg">
                        ${financeResult.exitValuation?.exitEquityValue?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Sources & Uses */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Sources */}
                <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-purple-200 dark:border-purple-700">
                  <h4 className="text-lg font-semibold text-purple-800 dark:text-purple-200 mb-3">Sources of Funds</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Senior Debt:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        ${financeResult.sourcesAndUses?.sources?.seniorDebt?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Subordinated Debt:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        ${financeResult.sourcesAndUses?.sources?.subDebt?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Sponsor Equity:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        ${financeResult.sourcesAndUses?.sources?.sponsorEquity?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    {financeResult.sourcesAndUses?.sources?.managementRollover > 0 && (
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Management Rollover:</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          ${financeResult.sourcesAndUses?.sources?.managementRollover?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                        </span>
                      </div>
                    )}
                    <div className="flex justify-between border-t border-gray-200 dark:border-gray-600 pt-2">
                      <span className="font-semibold text-gray-700 dark:text-gray-300">Total Sources:</span>
                      <span className="font-bold text-purple-800 dark:text-purple-200">
                        ${financeResult.sourcesAndUses?.sources?.total?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                  </div>
                </div>

                {/* Uses */}
                <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-purple-200 dark:border-purple-700">
                  <h4 className="text-lg font-semibold text-purple-800 dark:text-purple-200 mb-3">Uses of Funds</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Purchase Price:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        ${financeResult.sourcesAndUses?.uses?.purchasePrice?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Transaction Costs:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        ${financeResult.sourcesAndUses?.uses?.transactionCosts?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between border-t border-gray-200 dark:border-gray-600 pt-2">
                      <span className="font-semibold text-gray-700 dark:text-gray-300">Total Uses:</span>
                      <span className="font-bold text-purple-800 dark:text-purple-200">
                        ${financeResult.sourcesAndUses?.uses?.total?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Key Metrics */}
              <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-purple-200 dark:border-purple-700">
                <h4 className="text-lg font-semibold text-purple-800 dark:text-purple-200 mb-4">Key Deal Metrics</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Entry Multiple</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {financeResult.keyMetrics?.entryMultiple?.toFixed(1)}x
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Exit Multiple</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {financeResult.keyMetrics?.exitMultiple?.toFixed(1)}x
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Entry Leverage</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {financeResult.keyMetrics?.entryLeverage?.toFixed(1)}x
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Exit Leverage</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {financeResult.keyMetrics?.exitLeverage?.toFixed(1)}x
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Debt Paydown</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      ${financeResult.keyMetrics?.debtPaydown?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">% Debt Repaid</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {((financeResult.keyMetrics?.debtPaydownPercent || 0) * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Hold Period</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {financeResult.exitValuation?.exitYear} Years
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Remaining Debt</div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      ${financeResult.exitValuation?.remainingDebt?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                    </div>
                  </div>
                </div>
              </div>

              {/* Projections Table */}
              <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-purple-200 dark:border-purple-700 overflow-x-auto">
                <h4 className="text-lg font-semibold text-purple-800 dark:text-purple-200 mb-4">
                  5-Year Projections
                </h4>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-600">
                      <th className="text-left py-2 px-3 text-gray-600 dark:text-gray-400">Metric</th>
                      {financeResult.projections?.years?.map((year: number) => (
                        <th key={year} className="text-right py-2 px-3 text-gray-600 dark:text-gray-400">Year {year}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">Revenue ($M)</td>
                      {financeResult.projections?.revenue?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">EBITDA ($M)</td>
                      {financeResult.projections?.ebitda?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">Senior Debt ($M)</td>
                      {financeResult.projections?.seniorDebt?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">Subordinated Debt ($M)</td>
                      {financeResult.projections?.subDebt?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300 font-semibold">Free Cash Flow ($M)</td>
                      {financeResult.projections?.freeCashFlow?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-bold text-purple-700 dark:text-purple-300">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* M&A Results Display */}
          {financeResult && financeResult.success && financeModelType === "ma" && financeResult.accretionDilution && (
            <div className="mb-6 space-y-6">
              {/* Header with company names and download button */}
              <div className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 rounded-lg border border-purple-200 dark:border-purple-700">
                <div>
                  <h3 className="text-xl font-bold text-purple-900 dark:text-purple-100">
                    {financeResult.assumptions?.acquirerName} Acquiring {financeResult.assumptions?.targetName}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    M&A Analysis by {financeResult.providerUsed}
                  </p>
                </div>
                <Button
                  onClick={handleFinanceDownloadExcel}
                  className="bg-green-600 hover:bg-green-700 text-white"
                  disabled={financeDownloadLoading}
                  data-testid="button-download-ma-excel"
                >
                  {financeDownloadLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Downloading...
                    </>
                  ) : (
                    <>
                      <Download className="w-4 h-4 mr-2" />
                      Download M&A Excel Model
                    </>
                  )}
                </Button>
              </div>

              {/* Transaction Summary and Accretion/Dilution */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Transaction Metrics */}
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border-2 border-purple-400 dark:border-purple-500">
                  <h4 className="text-lg font-semibold text-purple-800 dark:text-purple-200 mb-3 flex items-center gap-2">
                    <DollarSign className="w-5 h-5" />
                    Transaction Metrics
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Purchase Price:</span>
                      <span className="font-semibold text-purple-700 dark:text-purple-300">
                        ${financeResult.transactionMetrics?.purchasePrice?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Enterprise Value:</span>
                      <span className="font-semibold text-purple-700 dark:text-purple-300">
                        ${financeResult.transactionMetrics?.enterpriseValue?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">EV/EBITDA Multiple:</span>
                      <span className="font-semibold text-purple-700 dark:text-purple-300">
                        {financeResult.transactionMetrics?.evEbitdaMultiple?.toFixed(1)}x
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Premium Paid:</span>
                      <span className="font-semibold text-purple-700 dark:text-purple-300">
                        {((financeResult.assumptions?.premium || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between border-t border-purple-200 dark:border-purple-600 pt-2 mt-2">
                      <span className="text-gray-600 dark:text-gray-400">Goodwill Created:</span>
                      <span className="font-bold text-purple-800 dark:text-purple-200">
                        ${financeResult.transactionMetrics?.goodwill?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                  </div>
                </div>

                {/* Accretion/Dilution Analysis */}
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border-2 border-blue-400 dark:border-blue-500">
                  <h4 className="text-lg font-semibold text-blue-800 dark:text-blue-200 mb-3 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Accretion / Dilution
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Year 1:</span>
                      <span className={`font-semibold ${financeResult.accretionDilution?.isAccretiveY1 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {((financeResult.accretionDilution?.percentImpact?.[1] || 0) * 100).toFixed(1)}% 
                        ({financeResult.accretionDilution?.isAccretiveY1 ? 'Accretive' : 'Dilutive'})
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Year 2:</span>
                      <span className={`font-semibold ${financeResult.accretionDilution?.isAccretiveY2 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {((financeResult.accretionDilution?.percentImpact?.[2] || 0) * 100).toFixed(1)}% 
                        ({financeResult.accretionDilution?.isAccretiveY2 ? 'Accretive' : 'Dilutive'})
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Year 3:</span>
                      <span className={`font-semibold ${financeResult.accretionDilution?.isAccretiveY3 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {((financeResult.accretionDilution?.percentImpact?.[3] || 0) * 100).toFixed(1)}% 
                        ({financeResult.accretionDilution?.isAccretiveY3 ? 'Accretive' : 'Dilutive'})
                      </span>
                    </div>
                    <div className="flex justify-between border-t border-blue-200 dark:border-blue-600 pt-2 mt-2">
                      <span className="text-gray-600 dark:text-gray-400">Year 5:</span>
                      <span className={`font-bold ${(financeResult.accretionDilution?.percentImpact?.[5] || 0) > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {((financeResult.accretionDilution?.percentImpact?.[5] || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Consideration Mix and Synergies */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Consideration Mix */}
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-700">
                  <h4 className="font-semibold text-green-800 dark:text-green-200 mb-3 flex items-center gap-2">
                    <PiggyBank className="w-5 h-5" />
                    Consideration Mix
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Cash:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        ${financeResult.transactionMetrics?.cashConsideration?.toLocaleString(undefined, {maximumFractionDigits: 0})}M 
                        ({((financeResult.assumptions?.cashPercent || 0) * 100).toFixed(0)}%)
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Stock:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        ${financeResult.transactionMetrics?.stockConsideration?.toLocaleString(undefined, {maximumFractionDigits: 0})}M 
                        ({((financeResult.assumptions?.stockPercent || 0) * 100).toFixed(0)}%)
                      </span>
                    </div>
                    <div className="flex justify-between border-t border-green-200 dark:border-green-600 pt-2 mt-2">
                      <span className="text-gray-600 dark:text-gray-400">New Shares Issued:</span>
                      <span className="font-bold text-green-700 dark:text-green-300">
                        {financeResult.transactionMetrics?.newSharesIssued?.toFixed(1)}M
                      </span>
                    </div>
                  </div>
                </div>

                {/* Synergies */}
                <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-700">
                  <h4 className="font-semibold text-amber-800 dark:text-amber-200 mb-3 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Synergies (Run-Rate)
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Revenue Synergies:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        ${financeResult.synergies?.revenueSynergies?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Cost Synergies:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        ${financeResult.synergies?.costSynergies?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                    <div className="flex justify-between border-t border-amber-200 dark:border-amber-600 pt-2 mt-2">
                      <span className="text-gray-600 dark:text-gray-400">Total Synergies:</span>
                      <span className="font-bold text-amber-700 dark:text-amber-300">
                        ${financeResult.synergies?.totalSynergies?.toLocaleString(undefined, {maximumFractionDigits: 0})}M
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Pro Forma Projections Table */}
              <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-purple-200 dark:border-purple-700 overflow-x-auto">
                <h4 className="text-lg font-semibold text-purple-800 dark:text-purple-200 mb-4">
                  Pro Forma Combined 5-Year Projections
                </h4>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-600">
                      <th className="text-left py-2 px-3 text-gray-600 dark:text-gray-400">Metric</th>
                      {financeResult.proFormaProjections?.years?.slice(1).map((year: number) => (
                        <th key={year} className="text-right py-2 px-3 text-gray-600 dark:text-gray-400">Year {year}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">Combined Revenue ($M)</td>
                      {financeResult.proFormaProjections?.revenue?.slice(1).map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">Combined EBITDA ($M)</td>
                      {financeResult.proFormaProjections?.ebitda?.slice(1).map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">Net Income ($M)</td>
                      {financeResult.proFormaProjections?.netIncome?.slice(1).map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300 font-semibold">Pro Forma EPS</td>
                      {financeResult.proFormaProjections?.eps?.slice(1).map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-bold text-purple-700 dark:text-purple-300">
                          ${val?.toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 3-Statement Model Results Display */}
          {financeResult && financeResult.success && financeModelType === "threestatement" && financeResult.incomeStatement && (
            <div className="mb-6 space-y-6">
              {/* Header with company name and download button */}
              <div className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 rounded-lg border border-indigo-200 dark:border-indigo-700">
                <div>
                  <h3 className="text-xl font-bold text-indigo-900 dark:text-indigo-100">
                    {financeResult.assumptions?.companyName} - 3-Statement Integrated Model
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {financeResult.assumptions?.projectionYears || 5} Year Financial Projections | Analysis by {financeResult.providerUsed}
                  </p>
                  <p className={`text-xs font-medium ${financeResult.summary?.isBalanced ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                    {financeResult.summary?.isBalanced ? 'Balance Sheet Check: BALANCED' : 'Balance Sheet Check: ERROR'}
                  </p>
                </div>
                <Button
                  onClick={handleFinanceDownloadExcel}
                  className="bg-green-600 hover:bg-green-700 text-white"
                  disabled={financeDownloadLoading}
                  data-testid="button-download-3statement-excel"
                >
                  {financeDownloadLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Downloading...
                    </>
                  ) : (
                    <>
                      <Download className="w-4 h-4 mr-2" />
                      Download 3-Statement Excel
                    </>
                  )}
                </Button>
              </div>

              {/* Key Metrics Summary */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-700 text-center">
                  <div className="text-sm text-gray-600 dark:text-gray-400">Revenue CAGR</div>
                  <div className="text-2xl font-bold text-indigo-700 dark:text-indigo-300">
                    {((financeResult.summary?.revenueCAGR || 0) * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-700 text-center">
                  <div className="text-sm text-gray-600 dark:text-gray-400">EBITDA CAGR</div>
                  <div className="text-2xl font-bold text-purple-700 dark:text-purple-300">
                    {((financeResult.summary?.ebitdaCAGR || 0) * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-700 text-center">
                  <div className="text-sm text-gray-600 dark:text-gray-400">Avg ROIC</div>
                  <div className="text-2xl font-bold text-green-700 dark:text-green-300">
                    {((financeResult.summary?.averageROIC || 0) * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-700 text-center">
                  <div className="text-sm text-gray-600 dark:text-gray-400">End Net Debt/EBITDA</div>
                  <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">
                    {(financeResult.summary?.endingNetDebtToEBITDA || 0).toFixed(1)}x
                  </div>
                </div>
              </div>

              {/* Income Statement Table */}
              <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-indigo-200 dark:border-indigo-700 overflow-x-auto">
                <h4 className="text-lg font-semibold text-indigo-800 dark:text-indigo-200 mb-4">
                  Income Statement Projections
                </h4>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-600">
                      <th className="text-left py-2 px-3 text-gray-600 dark:text-gray-400">Metric</th>
                      {financeResult.incomeStatement?.years?.map((year: string, idx: number) => (
                        <th key={idx} className="text-right py-2 px-3 text-gray-600 dark:text-gray-400">{year}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300 font-semibold">Revenue ($M)</td>
                      {financeResult.incomeStatement?.revenue?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-600 dark:text-gray-400 text-xs">Growth %</td>
                      {financeResult.incomeStatement?.revenueGrowth?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 text-xs text-gray-500">
                          {(val * 100).toFixed(1)}%
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">Gross Profit ($M)</td>
                      {financeResult.incomeStatement?.grossProfit?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700 bg-indigo-50/50 dark:bg-indigo-900/10">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300 font-semibold">EBITDA ($M)</td>
                      {financeResult.incomeStatement?.ebitda?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-bold text-indigo-700 dark:text-indigo-300">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-600 dark:text-gray-400 text-xs">EBITDA Margin %</td>
                      {financeResult.incomeStatement?.ebitdaMargin?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 text-xs text-gray-500">
                          {(val * 100).toFixed(1)}%
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">EBIT ($M)</td>
                      {financeResult.incomeStatement?.ebit?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700 bg-green-50/50 dark:bg-green-900/10">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300 font-semibold">Net Income ($M)</td>
                      {financeResult.incomeStatement?.netIncome?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-bold text-green-700 dark:text-green-300">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300 font-semibold">EPS</td>
                      {financeResult.incomeStatement?.eps?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-bold text-purple-700 dark:text-purple-300">
                          ${val?.toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Balance Sheet Summary */}
              <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-indigo-200 dark:border-indigo-700 overflow-x-auto">
                <h4 className="text-lg font-semibold text-indigo-800 dark:text-indigo-200 mb-4">
                  Balance Sheet Summary
                </h4>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-600">
                      <th className="text-left py-2 px-3 text-gray-600 dark:text-gray-400">Metric</th>
                      {financeResult.balanceSheet?.years?.map((year: string, idx: number) => (
                        <th key={idx} className="text-right py-2 px-3 text-gray-600 dark:text-gray-400">{year}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">Cash ($M)</td>
                      {financeResult.balanceSheet?.cash?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-green-600 dark:text-green-400">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700 bg-indigo-50/50 dark:bg-indigo-900/10">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300 font-semibold">Total Assets ($M)</td>
                      {financeResult.balanceSheet?.totalAssets?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-bold text-indigo-700 dark:text-indigo-300">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">Total Liabilities ($M)</td>
                      {financeResult.balanceSheet?.totalLiabilities?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-red-600 dark:text-red-400">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="bg-purple-50/50 dark:bg-purple-900/10">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300 font-semibold">Shareholders' Equity ($M)</td>
                      {financeResult.balanceSheet?.totalEquity?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-bold text-purple-700 dark:text-purple-300">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Cash Flow Summary */}
              <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-indigo-200 dark:border-indigo-700 overflow-x-auto">
                <h4 className="text-lg font-semibold text-indigo-800 dark:text-indigo-200 mb-4">
                  Cash Flow Summary
                </h4>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-600">
                      <th className="text-left py-2 px-3 text-gray-600 dark:text-gray-400">Metric</th>
                      {financeResult.cashFlow?.years?.map((year: string, idx: number) => (
                        <th key={idx} className="text-right py-2 px-3 text-gray-600 dark:text-gray-400">{year}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">CFO ($M)</td>
                      {financeResult.cashFlow?.cfo?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">CFI ($M)</td>
                      {financeResult.cashFlow?.cfi?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-red-600 dark:text-red-400">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300">CFF ($M)</td>
                      {financeResult.cashFlow?.cff?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                    <tr className="bg-green-50/50 dark:bg-green-900/10">
                      <td className="py-2 px-3 text-gray-700 dark:text-gray-300 font-semibold">Free Cash Flow ($M)</td>
                      {financeResult.cashFlow?.freeCashFlow?.map((val: number, idx: number) => (
                        <td key={idx} className="text-right py-2 px-3 font-bold text-green-700 dark:text-green-300">
                          ${val?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Leverage and Ratios */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-700">
                  <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-3 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Leverage Ratios (Year {financeResult.assumptions?.projectionYears || 5})
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Debt/Equity:</span>
                      <span className="font-semibold text-blue-700 dark:text-blue-300">
                        {financeResult.ratioAnalysis?.debtToEquity?.[financeResult.assumptions?.projectionYears || 5]?.toFixed(1)}x
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Debt/EBITDA:</span>
                      <span className="font-semibold text-blue-700 dark:text-blue-300">
                        {financeResult.ratioAnalysis?.debtToEBITDA?.[financeResult.assumptions?.projectionYears || 5]?.toFixed(1)}x
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Interest Coverage:</span>
                      <span className="font-semibold text-blue-700 dark:text-blue-300">
                        {financeResult.ratioAnalysis?.interestCoverage?.[financeResult.assumptions?.projectionYears || 5]?.toFixed(1)}x
                      </span>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-700">
                  <h4 className="font-semibold text-amber-800 dark:text-amber-200 mb-3 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Profitability (Year {financeResult.assumptions?.projectionYears || 5})
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Gross Margin:</span>
                      <span className="font-semibold text-amber-700 dark:text-amber-300">
                        {((financeResult.ratioAnalysis?.grossMargin?.[financeResult.assumptions?.projectionYears || 5] || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">EBITDA Margin:</span>
                      <span className="font-semibold text-amber-700 dark:text-amber-300">
                        {((financeResult.ratioAnalysis?.ebitdaMargin?.[financeResult.assumptions?.projectionYears || 5] || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Net Margin:</span>
                      <span className="font-semibold text-amber-700 dark:text-amber-300">
                        {((financeResult.ratioAnalysis?.netMargin?.[financeResult.assumptions?.projectionYears || 5] || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Clear Finance Panel Button */}
          <div className="mt-4 text-center">
            <Button
              onClick={handleFinanceClear}
              variant="outline"
              className="px-6 py-2 border-red-300 text-red-600 hover:bg-red-50 hover:border-red-400 dark:hover:bg-red-900/20 flex items-center mx-auto"
              disabled={financeLoading}
              data-testid="button-finance-clear-all"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Clear Finance
            </Button>
          </div>
        </div>
      </div>

      {/* DATA SCIENCE MODELS PANEL */}
      <div className="mt-16 bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/10 dark:to-purple-900/10 p-8 rounded-lg border-2 border-violet-200 dark:border-violet-700">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-violet-900 dark:text-violet-100 mb-3 flex items-center justify-center gap-3">
              <TrendingUp className="w-8 h-8 text-violet-600" />
              Data Science Models Panel
            </h2>
            <p className="text-lg text-gray-700 dark:text-gray-300 mb-2">
              Generate production-ready Python code for machine learning and statistical analysis
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Describe your problem in natural language - get complete, executable code with data preprocessing, model training, evaluation, and visualization
            </p>
            {/* Clear All Button */}
            <div className="mt-4">
              <Button
                onClick={handleDataScienceClear}
                variant="outline"
                size="sm"
                className="border-red-300 text-red-600 hover:bg-red-50 dark:border-red-700 dark:text-red-400 dark:hover:bg-red-900/20"
                disabled={dataScienceLoading}
                data-testid="button-datascience-clear"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear Data Science Panel
              </Button>
            </div>
          </div>

          {/* LLM Provider Selection */}
          <div className="flex justify-center mb-6">
            <div className="flex items-center gap-4 bg-white dark:bg-gray-800 p-4 rounded-lg border border-violet-200 dark:border-violet-700">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">AI Model:</span>
              <select
                value={dataScienceLLMProvider}
                onChange={(e) => setDataScienceLLMProvider(e.target.value as any)}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm"
                data-testid="select-datascience-llm"
              >
                <option value="zhi5">ZHI 5 - Default</option>
                <option value="zhi1">ZHI 1</option>
                <option value="zhi2">ZHI 2</option>
                <option value="zhi3">ZHI 3</option>
                <option value="zhi4">ZHI 4</option>
              </select>
            </div>
          </div>

          {/* Input Area */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            {/* Problem Description Input */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border-2 border-violet-200 dark:border-violet-700">
              <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                Problem Description
              </label>
              <Textarea
                value={dataScienceInputText}
                onChange={(e) => setDataScienceInputText(e.target.value)}
                placeholder="Describe your data science problem in natural language...

Example: Build me a multiple regression model to predict house prices. The target variable is SalePrice. The features I want to use are: SquareFootage, Bedrooms, Bathrooms, YearBuilt, and GarageSize. Generate synthetic data for 500 houses. Use an 80/20 train-test split with random state 42. Enable feature scaling. Run 5-fold cross-validation."
                className="min-h-[200px] text-base"
                disabled={dataScienceLoading}
                data-testid="textarea-datascience-input"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                Include: target variable, features, data source/requirements, model preferences
              </p>
            </div>

            {/* Custom Instructions */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border-2 border-violet-200 dark:border-violet-700">
              <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                Custom Instructions (Optional)
              </label>
              <Textarea
                value={dataScienceCustomInstructions}
                onChange={(e) => setDataScienceCustomInstructions(e.target.value)}
                placeholder="Add any special requirements...

Examples:
- Use polynomial features with degree 3
- Include additional visualizations
- Add hyperparameter tuning with GridSearchCV
- Generate code compatible with Google Colab
- Add detailed comments for teaching purposes"
                className="min-h-[200px] text-base"
                disabled={dataScienceLoading}
                data-testid="textarea-datascience-instructions"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                Specify output format, additional analysis, or code style preferences
              </p>
            </div>
          </div>

          {/* Function Buttons */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <Button
              onClick={() => handleDataScienceProcess("regression")}
              disabled={dataScienceLoading || !dataScienceInputText.trim()}
              className="h-24 bg-gradient-to-br from-violet-600 to-purple-700 hover:from-violet-700 hover:to-purple-800 text-white flex flex-col items-center justify-center gap-2 rounded-lg shadow-lg"
              data-testid="button-datascience-regression"
            >
              {dataScienceLoading && dataScienceModelType === "regression" ? (
                <Loader2 className="h-6 w-6 animate-spin" />
              ) : (
                <TrendingUp className="h-6 w-6" />
              )}
              <span className="font-bold text-lg">Regression Models</span>
              <span className="text-xs opacity-80">Linear, Ridge, Lasso, Logistic</span>
            </Button>

            <Button
              onClick={() => handleDataScienceProcess("ml")}
              disabled={dataScienceLoading || !dataScienceInputText.trim()}
              className="h-24 bg-gradient-to-br from-indigo-600 to-blue-700 hover:from-indigo-700 hover:to-blue-800 text-white flex flex-col items-center justify-center gap-2 rounded-lg shadow-lg"
              data-testid="button-datascience-ml"
            >
              {dataScienceLoading && dataScienceModelType === "ml" ? (
                <Loader2 className="h-6 w-6 animate-spin" />
              ) : (
                <Brain className="h-6 w-6" />
              )}
              <span className="font-bold text-lg">Machine Learning</span>
              <span className="text-xs opacity-80">Classification, Clustering, PCA</span>
            </Button>

            <Button
              onClick={() => handleDataScienceProcess("forecasting")}
              disabled={dataScienceLoading || !dataScienceInputText.trim()}
              className="h-24 bg-gradient-to-br from-teal-600 to-cyan-700 hover:from-teal-700 hover:to-cyan-800 text-white flex flex-col items-center justify-center gap-2 rounded-lg shadow-lg"
              data-testid="button-datascience-forecasting"
            >
              {dataScienceLoading && dataScienceModelType === "forecasting" ? (
                <Loader2 className="h-6 w-6 animate-spin" />
              ) : (
                <LineChart className="h-6 w-6" />
              )}
              <span className="font-bold text-lg">Statistical Forecasting</span>
              <span className="text-xs opacity-80">ARIMA, Holt-Winters, Prophet</span>
            </Button>

            <Button
              onClick={() => handleDataScienceProcess("predictive")}
              disabled={dataScienceLoading}
              className="h-24 bg-gradient-to-br from-purple-600 to-pink-700 hover:from-purple-700 hover:to-pink-800 text-white flex flex-col items-center justify-center gap-2 rounded-lg shadow-lg"
              data-testid="button-datascience-predictive"
            >
              {dataScienceLoading && dataScienceModelType === "predictive" ? (
                <Loader2 className="h-6 w-6 animate-spin" />
              ) : (
                <BarChart3 className="h-6 w-6" />
              )}
              <span className="font-bold text-lg">Predictive Analytics</span>
              <span className="text-xs opacity-80">End-to-End ML Pipeline</span>
            </Button>
          </div>

          {/* Results Display */}
          {dataScienceResult && dataScienceResult.success && (() => {
            // Reliable forecasting detection: check modelType state OR forecasting-specific parameters
            const isForecastingResult = dataScienceModelType === "forecasting" || 
              dataScienceResult.parameters?.forecastHorizon !== undefined ||
              ['arima', 'sarima', 'sarimax', 'holtwinters', 'prophet', 'auto'].includes(
                dataScienceResult.parameters?.modelType?.toLowerCase() || ''
              );
            
            // Predictive analytics detection: check modelType state OR predictive-specific parameters
            const isPredictiveResult = dataScienceModelType === "predictive" ||
              dataScienceResult.parameters?.featureEngineeringLevel !== undefined ||
              dataScienceResult.parameters?.ensembleMethods !== undefined ||
              dataScienceResult.parameters?.interpretabilityLevel !== undefined;
            
            return (
            <div className="mb-6 space-y-6">
              {/* Header with model info and download button */}
              <div className="flex items-center justify-between flex-wrap gap-4 p-4 bg-white dark:bg-gray-800 rounded-lg border border-violet-200 dark:border-violet-700">
                <div>
                  <h3 className="text-xl font-bold text-violet-900 dark:text-violet-100">
                    {isPredictiveResult
                      ? `Predictive Analytics Pipeline - ${dataScienceResult.parameters?.problemType?.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()) || 'Auto'}`
                      : isForecastingResult 
                        ? `Time Series Forecasting - ${dataScienceResult.parameters?.modelType?.toUpperCase() || 'Auto'} Model`
                        : dataScienceResult.parameters?.problemType 
                          ? `${dataScienceResult.parameters.problemType.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())} - ${dataScienceResult.parameters.modelType?.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()) || 'Auto'} Model`
                          : `${dataScienceResult.parameters?.regressionType?.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()) || 'Regression'} Model Generated`
                    }
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {isPredictiveResult 
                      ? `Target: ${dataScienceResult.parameters?.targetVariable || 'target'} | Feature Eng: ${dataScienceResult.parameters?.featureEngineeringLevel?.replace(/\b\w/g, (l: string) => l.toUpperCase()) || 'Standard'} | `
                      : isForecastingResult 
                        ? `Forecast: ${dataScienceResult.parameters?.forecastHorizon || 12} periods | Frequency: ${dataScienceResult.parameters?.frequency || 'Monthly'} | `
                        : dataScienceResult.parameters?.targetVariable 
                          ? `Target: ${dataScienceResult.parameters.targetVariable} | `
                          : ''
                    }
                    {!isPredictiveResult && dataScienceResult.parameters?.featureVariables?.length > 0 && `Features: ${dataScienceResult.parameters.featureVariables.slice(0, 3).join(', ')}${dataScienceResult.parameters.featureVariables.length > 3 ? '...' : ''} | `}
                    Analysis by {dataScienceResult.providerUsed}
                  </p>
                </div>
                <Button
                  onClick={handleDataScienceDownload}
                  className="bg-green-600 hover:bg-green-700 text-white"
                  disabled={dataScienceDownloadLoading}
                  data-testid="button-datascience-download"
                >
                  {dataScienceDownloadLoading ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4 mr-2" />
                  )}
                  Download Python File
                </Button>
              </div>

              {/* Model Parameters Summary - Adaptive for Regression, ML, Forecasting, and Predictive Analytics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 bg-violet-100 dark:bg-violet-900/30 rounded-lg text-center">
                  <p className="text-sm text-violet-600 dark:text-violet-400">
                    {isPredictiveResult
                      ? 'Problem Type'
                      : isForecastingResult 
                        ? 'Model Type'
                        : dataScienceResult.parameters?.problemType 
                          ? 'Problem Type' 
                          : 'Regression Type'}
                  </p>
                  <p className="text-lg font-bold text-violet-900 dark:text-violet-100">
                    {isPredictiveResult
                      ? dataScienceResult.parameters?.problemType?.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()) || 'Auto'
                      : isForecastingResult 
                        ? dataScienceResult.parameters?.modelType?.toUpperCase() || 'Auto'
                        : dataScienceResult.parameters?.problemType 
                          ? dataScienceResult.parameters.problemType.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())
                          : dataScienceResult.parameters?.regressionType?.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()) || 'Regression'
                    }
                  </p>
                </div>
                <div className="p-4 bg-purple-100 dark:bg-purple-900/30 rounded-lg text-center">
                  <p className="text-sm text-purple-600 dark:text-purple-400">
                    {isPredictiveResult
                      ? 'Feature Engineering'
                      : isForecastingResult 
                        ? 'Forecast Horizon'
                        : dataScienceResult.parameters?.problemType 
                          ? 'Model Type' 
                          : 'Test Size'}
                  </p>
                  <p className="text-lg font-bold text-purple-900 dark:text-purple-100">
                    {isPredictiveResult
                      ? dataScienceResult.parameters?.featureEngineeringLevel?.replace(/\b\w/g, (l: string) => l.toUpperCase()) || 'Standard'
                      : isForecastingResult 
                        ? `${dataScienceResult.parameters?.forecastHorizon || 12} periods`
                        : dataScienceResult.parameters?.problemType 
                          ? (dataScienceResult.parameters.modelType?.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()) || 'Auto')
                          : `${((dataScienceResult.parameters?.testSize || 0.2) * 100).toFixed(0)}%`
                    }
                  </p>
                </div>
                <div className="p-4 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg text-center">
                  <p className="text-sm text-indigo-600 dark:text-indigo-400">
                    {isPredictiveResult
                      ? 'Ensemble Methods'
                      : isForecastingResult 
                        ? 'Seasonal Period'
                        : dataScienceResult.parameters?.problemType === 'clustering' 
                          ? 'Clusters' 
                          : 'CV Folds'}
                  </p>
                  <p className="text-lg font-bold text-indigo-900 dark:text-indigo-100">
                    {isPredictiveResult
                      ? dataScienceResult.parameters?.ensembleMethods ? 'Enabled' : 'Disabled'
                      : isForecastingResult 
                        ? dataScienceResult.parameters?.seasonalPeriod || 12
                        : dataScienceResult.parameters?.problemType === 'clustering'
                          ? (dataScienceResult.parameters.nClusters || 'Auto')
                          : (dataScienceResult.parameters?.cvFolds || dataScienceResult.parameters?.crossValidationFolds || 5)
                    }
                  </p>
                </div>
                <div className="p-4 bg-fuchsia-100 dark:bg-fuchsia-900/30 rounded-lg text-center">
                  <p className="text-sm text-fuchsia-600 dark:text-fuchsia-400">
                    {isPredictiveResult
                      ? 'Interpretability'
                      : isForecastingResult 
                        ? 'Confidence Level'
                        : dataScienceResult.parameters?.problemType 
                          ? 'Tuning' 
                          : 'Feature Scaling'}
                  </p>
                  <p className="text-lg font-bold text-fuchsia-900 dark:text-fuchsia-100">
                    {isPredictiveResult
                      ? dataScienceResult.parameters?.interpretabilityLevel?.replace(/\b\w/g, (l: string) => l.toUpperCase()) || 'Full'
                      : isForecastingResult 
                        ? `${((dataScienceResult.parameters?.confidenceLevel || 0.95) * 100).toFixed(0)}%`
                        : dataScienceResult.parameters?.problemType 
                          ? (dataScienceResult.parameters.hyperparameterTuning?.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()) || 'Random')
                          : (dataScienceResult.parameters?.scaleFeatures ? 'Enabled' : 'Disabled')
                    }
                  </p>
                </div>
              </div>

              {/* Python Code Preview */}
              <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                    <FileCode className="w-4 h-4" />
                    Generated Python Code Preview
                  </h4>
                  <span className="text-xs text-gray-500">
                    {dataScienceResult.pythonCode?.split('\n').length || 0} lines
                  </span>
                </div>
                <pre className="text-sm text-green-400 overflow-x-auto max-h-96 overflow-y-auto font-mono">
                  {dataScienceResult.pythonCode?.slice(0, 3000)}
                  {dataScienceResult.pythonCode?.length > 3000 && '\n\n... [Download for full code] ...'}
                </pre>
              </div>

              {/* Code Sections Description */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-violet-200 dark:border-violet-700 text-center">
                  <FileCode className="w-5 h-5 mx-auto mb-1 text-violet-600" />
                  <p className="text-xs font-medium text-gray-700 dark:text-gray-300">Data Loading</p>
                </div>
                <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-violet-200 dark:border-violet-700 text-center">
                  <Search className="w-5 h-5 mx-auto mb-1 text-violet-600" />
                  <p className="text-xs font-medium text-gray-700 dark:text-gray-300">EDA</p>
                </div>
                <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-violet-200 dark:border-violet-700 text-center">
                  <Brain className="w-5 h-5 mx-auto mb-1 text-violet-600" />
                  <p className="text-xs font-medium text-gray-700 dark:text-gray-300">Model Training</p>
                </div>
                <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-violet-200 dark:border-violet-700 text-center">
                  <BarChart3 className="w-5 h-5 mx-auto mb-1 text-violet-600" />
                  <p className="text-xs font-medium text-gray-700 dark:text-gray-300">Evaluation</p>
                </div>
                <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-violet-200 dark:border-violet-700 text-center">
                  <LineChart className="w-5 h-5 mx-auto mb-1 text-violet-600" />
                  <p className="text-xs font-medium text-gray-700 dark:text-gray-300">Visualization</p>
                </div>
              </div>
            </div>
            );
          })()}

          {/* Clear Data Science Panel Button */}
          <div className="mt-4 text-center">
            <Button
              onClick={handleDataScienceClear}
              variant="outline"
              className="px-6 py-2 border-red-300 text-red-600 hover:bg-red-50 hover:border-red-400 dark:hover:bg-red-900/20 flex items-center mx-auto"
              disabled={dataScienceLoading}
              data-testid="button-datascience-clear-all"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Clear Data Science
            </Button>
          </div>
        </div>
      </div>

      {/* GLOBAL CLEAR ALL - Clear Finance & Data Science Panels */}
      <div className="mt-8 mb-8 text-center">
        <div className="inline-flex flex-col items-center gap-3 p-6 bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/10 dark:to-orange-900/10 rounded-lg border-2 border-red-200 dark:border-red-700">
          <h3 className="text-lg font-semibold text-red-800 dark:text-red-200">
            Reset All Panels
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 max-w-md">
            Clear all inputs and results from both Finance and Data Science panels
          </p>
          <Button
            onClick={handleGlobalClearAll}
            variant="destructive"
            size="lg"
            className="px-8 py-3 bg-red-600 hover:bg-red-700 text-white"
            disabled={financeLoading || dataScienceLoading}
            data-testid="button-global-clear-all"
          >
            <Trash2 className="h-5 w-5 mr-2" />
            Clear All Panels
          </Button>
        </div>
      </div>

      {/* TEXT MODEL VALIDATOR - Interpretive Generosity Framework */}
      <div className="mt-16 bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/10 dark:to-teal-900/10 p-8 rounded-lg border-2 border-emerald-200 dark:border-emerald-700">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-emerald-900 dark:text-emerald-100 mb-3 flex items-center justify-center gap-3">
              <BookOpen className="w-8 h-8 text-emerald-600" />
              Text Model Validator
            </h2>
            <p className="text-lg text-gray-700 dark:text-gray-300 mb-2">
              Validate texts through interpretive generosity - find models that make difficult texts coherent
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Extract logic, swap domains, formalize mathematically, or let AI decide the best approach
            </p>
          </div>

          {/* Input Area */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-semibold text-emerald-800 dark:text-emerald-200">
                Input Text to Validate
              </label>
              <label className="cursor-pointer">
                <input
                  type="file"
                  accept=".pdf,.doc,.docx,.txt"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFileUpload(file, setValidatorInputText);
                  }}
                  data-testid="input-validator-upload"
                />
                <Button
                  variant="outline"
                  size="sm"
                  className="border-emerald-300 hover:bg-emerald-50 dark:hover:bg-emerald-900/20"
                  onClick={(e) => {
                    e.preventDefault();
                    (e.currentTarget.previousElementSibling as HTMLInputElement)?.click();
                  }}
                  data-testid="button-validator-upload"
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Upload Document
                </Button>
              </label>
            </div>
            <Textarea
              value={validatorInputText}
              onChange={(e) => setValidatorInputText(e.target.value)}
              placeholder="Paste complex, obscure, or muddled text here... (philosophy papers, technical documents, draft arguments, etc.)"
              className="min-h-[200px] font-mono text-sm"
              data-testid="textarea-validator-input"
            />
          </div>

          {/* Six Main Buttons */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            <Button
              onClick={() => {
                setShowValidatorCustomization(prev => validatorMode === "reconstruction" ? !prev : true);
                setValidatorMode("reconstruction");
              }}
              className={`flex flex-col items-center justify-center p-6 h-auto ${
                validatorMode === "reconstruction" 
                  ? "bg-emerald-600 hover:bg-emerald-700 text-white" 
                  : "bg-white dark:bg-gray-800 hover:bg-emerald-50 dark:hover:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300 border-2 border-emerald-300"
              }`}
              disabled={validatorLoading}
              data-testid="button-reconstruction"
            >
              <RefreshCw className="w-6 h-6 mb-2" />
              <span className="font-bold text-lg">RECONSTRUCTION</span>
              <span className="text-xs mt-1 text-center opacity-80">Clean up logic</span>
            </Button>

            <Button
              onClick={() => {
                setShowValidatorCustomization(prev => validatorMode === "isomorphism" ? !prev : true);
                setValidatorMode("isomorphism");
              }}
              className={`flex flex-col items-center justify-center p-6 h-auto ${
                validatorMode === "isomorphism" 
                  ? "bg-teal-600 hover:bg-teal-700 text-white" 
                  : "bg-white dark:bg-gray-800 hover:bg-teal-50 dark:hover:bg-teal-900/20 text-teal-700 dark:text-teal-300 border-2 border-teal-300"
              }`}
              disabled={validatorLoading}
              data-testid="button-isomorphism"
            >
              <FileEdit className="w-6 h-6 mb-2" />
              <span className="font-bold text-lg">ISOMORPHISM</span>
              <span className="text-xs mt-1 text-center opacity-80">Swap domains</span>
            </Button>

            <Button
              onClick={() => {
                setShowValidatorCustomization(prev => validatorMode === "mathmodel" ? !prev : true);
                setValidatorMode("mathmodel");
              }}
              className={`flex flex-col items-center justify-center p-6 h-auto ${
                validatorMode === "mathmodel" 
                  ? "bg-blue-600 hover:bg-blue-700 text-white" 
                  : "bg-white dark:bg-gray-800 hover:bg-blue-50 dark:hover:bg-blue-900/20 text-blue-700 dark:text-blue-300 border-2 border-blue-300"
              }`}
              disabled={validatorLoading}
              data-testid="button-mathmodel"
            >
              <Zap className="w-6 h-6 mb-2" />
              <span className="font-bold text-lg">MATH MODEL</span>
              <span className="text-xs mt-1 text-center opacity-80">Formalize it</span>
            </Button>

            <Button
              onClick={() => {
                setShowValidatorCustomization(prev => validatorMode === "truth-isomorphism" ? !prev : true);
                setValidatorMode("truth-isomorphism");
              }}
              className={`flex flex-col items-center justify-center p-6 h-auto ${
                validatorMode === "truth-isomorphism" 
                  ? "bg-orange-600 hover:bg-orange-700 text-white" 
                  : "bg-white dark:bg-gray-800 hover:bg-orange-50 dark:hover:bg-orange-900/20 text-orange-700 dark:text-orange-300 border-2 border-orange-300"
              }`}
              disabled={validatorLoading}
              data-testid="button-truth-isomorphism"
            >
              <Shield className="w-6 h-6 mb-2" />
              <span className="font-bold text-lg">TRUTH SELECT</span>
              <span className="text-xs mt-1 text-center opacity-80">Choose truth mapping</span>
            </Button>

            <Button
              onClick={() => {
                setShowValidatorCustomization(prev => validatorMode === "math-truth-select" ? !prev : true);
                setValidatorMode("math-truth-select");
              }}
              className={`flex flex-col items-center justify-center p-6 h-auto ${
                validatorMode === "math-truth-select" 
                  ? "bg-indigo-600 hover:bg-indigo-700 text-white" 
                  : "bg-white dark:bg-gray-800 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 text-indigo-700 dark:text-indigo-300 border-2 border-indigo-300"
              }`}
              disabled={validatorLoading}
              data-testid="button-math-truth-select"
            >
              <BarChart3 className="w-6 h-6 mb-2" />
              <span className="font-bold text-lg">MATH + TRUTH</span>
              <span className="text-xs mt-1 text-center opacity-80">Formalize with truth control</span>
            </Button>

            <Button
              onClick={() => {
                setShowValidatorCustomization(false);
                handleValidatorProcess("autodecide");
              }}
              className={`flex flex-col items-center justify-center p-6 h-auto ${
                validatorMode === "autodecide" 
                  ? "bg-purple-600 hover:bg-purple-700 text-white" 
                  : "bg-white dark:bg-gray-800 hover:bg-purple-50 dark:hover:bg-purple-900/20 text-purple-700 dark:text-purple-300 border-2 border-purple-300"
              }`}
              disabled={validatorLoading}
              data-testid="button-autodecide"
            >
              <Brain className="w-6 h-6 mb-2" />
              <span className="font-bold text-lg">AUTO-DECIDE</span>
              <span className="text-xs mt-1 text-center opacity-80">Let AI choose</span>
            </Button>
          </div>

          {/* Clear All Button */}
          <div className="mt-4 text-center">
            <Button
              onClick={handleValidatorClear}
              variant="outline"
              className="px-6 py-2 border-red-300 text-red-600 hover:bg-red-50 hover:border-red-400 dark:hover:bg-red-900/20 flex items-center mx-auto"
              disabled={validatorLoading}
              data-testid="button-validator-clear-all"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              <span>New Analysis / Clear All</span>
            </Button>
          </div>

          {/* Optional Custom Instructions Box */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-gray-200 dark:border-gray-700 mt-6">
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Custom Instructions (Optional)
            </label>
            <Textarea
              value={validatorCustomInstructions}
              onChange={(e) => setValidatorCustomInstructions(e.target.value)}
              placeholder="e.g., 'Identify an isomorphism from biochemistry' or 'Reconstruct as a control theory model' or 'Use game theory framework'"
              className="min-h-[100px] text-sm"
              data-testid="textarea-validator-custom-instructions"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Provide specific guidance about the nature of the reconstruction, isomorphism, formalization, etc.
            </p>
          </div>

          {/* LLM Provider Selector */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-gray-200 dark:border-gray-700 mt-4">
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              AI Model Selection
            </label>
            <Select value={validatorLLMProvider} onValueChange={setValidatorLLMProvider}>
              <SelectTrigger data-testid="select-validator-llm" className="w-full">
                <SelectValue placeholder="Select AI Model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="zhi5">ZHI 5 - Default</SelectItem>
                <SelectItem value="zhi1">ZHI 1</SelectItem>
                <SelectItem value="zhi2">ZHI 2</SelectItem>
                <SelectItem value="zhi3">ZHI 3</SelectItem>
                <SelectItem value="zhi4">ZHI 4</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Choose which AI model powers the validation. Grok is recommended for most tasks.
            </p>
          </div>

          {/* Customization Panel */}
          {showValidatorCustomization && validatorMode && validatorMode !== "autodecide" && (
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border-2 border-gray-200 dark:border-gray-700 mb-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <FileEdit className="w-5 h-5" />
                Customization Options - {validatorMode.toUpperCase()}
              </h3>

              {validatorMode === "reconstruction" && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Target Domain</label>
                    <input
                      type="text"
                      value={validatorTargetDomain}
                      onChange={(e) => setValidatorTargetDomain(e.target.value)}
                      placeholder="e.g., cognitive science, information theory, computational model"
                      className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                      data-testid="input-target-domain"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Fidelity Level</label>
                    <Select value={validatorFidelityLevel} onValueChange={(value: "conservative" | "aggressive") => setValidatorFidelityLevel(value)}>
                      <SelectTrigger data-testid="select-fidelity-level">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="conservative">Conservative (Stay close to original)</SelectItem>
                        <SelectItem value="aggressive">Aggressive (Maximize clarity)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}

              {validatorMode === "isomorphism" && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Target Domain</label>
                    <input
                      type="text"
                      value={validatorTargetDomain}
                      onChange={(e) => setValidatorTargetDomain(e.target.value)}
                      placeholder="e.g., economics, evolutionary psychology, thermodynamics"
                      className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                      data-testid="input-target-domain"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Constraint Type</label>
                    <Select value={validatorConstraintType} onValueChange={(value: "pure-swap" | "true-statements" | "historical") => setValidatorConstraintType(value)}>
                      <SelectTrigger data-testid="select-constraint-type">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="pure-swap">Pure Swap (Just replace terms)</SelectItem>
                        <SelectItem value="true-statements">True Statements (Find verified true claims)</SelectItem>
                        <SelectItem value="historical">Historical (Find historical theory)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}

              {validatorMode === "mathmodel" && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Mathematical Framework</label>
                    <Select value={validatorMathFramework} onValueChange={setValidatorMathFramework}>
                      <SelectTrigger data-testid="select-math-framework">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="variational-inference">Variational Inference</SelectItem>
                        <SelectItem value="game-theory">Game Theory</SelectItem>
                        <SelectItem value="category-theory">Category Theory</SelectItem>
                        <SelectItem value="dynamical-systems">Dynamical Systems</SelectItem>
                        <SelectItem value="graph-theory">Graph Theory</SelectItem>
                        <SelectItem value="optimization">Optimization Problems</SelectItem>
                        <SelectItem value="probability">Probability Theory</SelectItem>
                        <SelectItem value="differential-equations">Differential Equations</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Rigor Level</label>
                    <Select value={validatorRigorLevel} onValueChange={(value: "sketch" | "semi-formal" | "proof-ready") => setValidatorRigorLevel(value)}>
                      <SelectTrigger data-testid="select-rigor-level">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sketch">Sketch (Intuitive formalization)</SelectItem>
                        <SelectItem value="semi-formal">Semi-formal (Notation with explanations)</SelectItem>
                        <SelectItem value="proof-ready">Proof-ready (Complete formal spec)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}

              {validatorMode === "truth-isomorphism" && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Target Domain</label>
                    <input
                      type="text"
                      value={validatorTargetDomain}
                      onChange={(e) => setValidatorTargetDomain(e.target.value)}
                      placeholder="e.g., physics, sociology, computer science"
                      className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                      data-testid="input-target-domain-truth"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Truth-Value Mapping</label>
                    <Select value={validatorTruthMapping} onValueChange={(value: "false-to-true" | "true-to-true" | "true-to-false") => setValidatorTruthMapping(value)}>
                      <SelectTrigger data-testid="select-truth-mapping">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="false-to-true">FALSE → TRUE (Map false statements to true ones)</SelectItem>
                        <SelectItem value="true-to-true">TRUE → TRUE (Preserve truth while swapping domains)</SelectItem>
                        <SelectItem value="true-to-false">TRUE → FALSE (Map true statements to false ones)</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                      Choose how to handle truth values when mapping to the target domain
                    </p>
                  </div>
                  <div className="flex items-center justify-between p-3 border rounded dark:border-gray-600 bg-blue-50 dark:bg-blue-900/20">
                    <div>
                      <label htmlFor="literal-truth-toggle" className="text-sm font-medium cursor-pointer">
                        Literal Truth Mode
                      </label>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        Ensure statements are LITERALLY true (not approximately or qualifiedly true)
                      </p>
                    </div>
                    <Switch
                      id="literal-truth-toggle"
                      checked={validatorLiteralTruth}
                      onCheckedChange={setValidatorLiteralTruth}
                      data-testid="switch-literal-truth"
                    />
                  </div>
                </div>
              )}

              {validatorMode === "math-truth-select" && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Mathematical Framework</label>
                    <Select value={validatorMathFramework} onValueChange={setValidatorMathFramework}>
                      <SelectTrigger data-testid="select-math-framework-truth">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="variational-inference">Variational Inference</SelectItem>
                        <SelectItem value="game-theory">Game Theory</SelectItem>
                        <SelectItem value="category-theory">Category Theory</SelectItem>
                        <SelectItem value="dynamical-systems">Dynamical Systems</SelectItem>
                        <SelectItem value="graph-theory">Graph Theory</SelectItem>
                        <SelectItem value="optimization">Optimization Problems</SelectItem>
                        <SelectItem value="probability">Probability Theory</SelectItem>
                        <SelectItem value="differential-equations">Differential Equations</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Truth-Value Assignment</label>
                    <Select value={validatorMathTruthMapping} onValueChange={(value: "make-true" | "keep-true" | "make-false") => setValidatorMathTruthMapping(value)}>
                      <SelectTrigger data-testid="select-math-truth-mapping">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="make-true">MAKE TRUE (Assign values to make formalization true)</SelectItem>
                        <SelectItem value="keep-true">KEEP TRUE (Assign values preserving truth)</SelectItem>
                        <SelectItem value="make-false">MAKE FALSE (Assign values to make formalization false)</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                      Control truth value through semantic value assignment to mathematical constants
                    </p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Rigor Level</label>
                    <Select value={validatorRigorLevel} onValueChange={(value: "sketch" | "semi-formal" | "proof-ready") => setValidatorRigorLevel(value)}>
                      <SelectTrigger data-testid="select-rigor-level-truth">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sketch">Sketch (Intuitive formalization)</SelectItem>
                        <SelectItem value="semi-formal">Semi-formal (Notation with explanations)</SelectItem>
                        <SelectItem value="proof-ready">Proof-ready (Complete formal spec)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-center justify-between p-3 border rounded dark:border-gray-600 bg-blue-50 dark:bg-blue-900/20">
                    <div>
                      <label htmlFor="literal-truth-toggle-math" className="text-sm font-medium cursor-pointer">
                        Literal Truth Mode
                      </label>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        Ensure statements are LITERALLY true (not approximately or qualifiedly true)
                      </p>
                    </div>
                    <Switch
                      id="literal-truth-toggle-math"
                      checked={validatorLiteralTruth}
                      onCheckedChange={setValidatorLiteralTruth}
                      data-testid="switch-literal-truth-math"
                    />
                  </div>
                </div>
              )}

              <div className="flex gap-3 mt-6">
                <Button
                  onClick={() => handleValidatorProcess(validatorMode)}
                  className="bg-emerald-600 hover:bg-emerald-700 text-white"
                  disabled={validatorLoading}
                  data-testid="button-generate-validator"
                >
                  {validatorLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4 mr-2" />
                      Generate
                    </>
                  )}
                </Button>
                <Button
                  onClick={() => setShowValidatorCustomization(false)}
                  variant="outline"
                  data-testid="button-hide-customization"
                >
                  Hide Options
                </Button>
              </div>
            </div>
          )}

          {/* Output Display */}
          {validatorOutput && (
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border-2 border-emerald-300 dark:border-emerald-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-emerald-600" />
                  Validation Result ({validatorMode?.toUpperCase()})
                </h3>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleDownloadText(validatorOutput, `validator-output-${validatorMode}.txt`)}
                    data-testid="button-download-validator-output"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                  <SendToButton
                    text={validatorOutput}
                    onSendToIntelligence={(text) => setDocumentA({ content: text })}
                    onSendToHumanizer={(text) => setBoxA(text)}
                    onSendToChat={(text) => {
                      const chatInput = document.querySelector('[data-testid="chat-input"]') as HTMLTextAreaElement;
                      if (chatInput) {
                        chatInput.value = text;
                        chatInput.dispatchEvent(new Event('input', { bubbles: true }));
                      }
                    }}
                  />
                  <CopyButton text={validatorOutput} />
                  <Button
                    onClick={() => {
                      setRedoCustomInstructions("");
                      setShowRedoModal(true);
                    }}
                    variant="outline"
                    size="sm"
                    className="bg-amber-50 hover:bg-amber-100 border-amber-300 text-amber-800 dark:bg-amber-900/20 dark:hover:bg-amber-900/30 dark:border-amber-700 dark:text-amber-300"
                    data-testid="button-redo-validator"
                  >
                    <RefreshCw className="w-4 h-4 mr-1" />
                    Redo
                  </Button>
                  <Button
                    onClick={handleValidatorClear}
                    variant="outline"
                    size="sm"
                    data-testid="button-clear-validator"
                  >
                    <Trash2 className="w-4 h-4 mr-1" />
                    Clear
                  </Button>
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded border border-gray-200 dark:border-gray-700 max-h-[600px] overflow-y-auto">
                {validatorMode === "mathmodel" ? (
                  <MathRenderer 
                    content={validatorOutput} 
                    className="text-gray-800 dark:text-gray-200"
                  />
                ) : (
                  <pre className="whitespace-pre-wrap font-mono text-sm text-gray-800 dark:text-gray-200">
                    {validatorOutput}
                  </pre>
                )}
              </div>
            </div>
          )}

          {/* Loading State */}
          {validatorLoading && (
            <div className="flex flex-col items-center justify-center py-12">
              <Loader2 className="w-12 h-12 animate-spin text-emerald-600 mb-4" />
              <p className="text-gray-600 dark:text-gray-400">Processing text validation...</p>
            </div>
          )}

          {/* Redo Modal with Custom Instructions */}
          <Dialog open={showRedoModal} onOpenChange={setShowRedoModal}>
            <DialogContent className="sm:max-w-[600px]">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <RefreshCw className="w-5 h-5 text-amber-600" />
                  Redo with Custom Instructions
                </DialogTitle>
                <DialogDescription>
                  Enter specific instructions to guide the reconstruction. Leave blank for default behavior.
                </DialogDescription>
              </DialogHeader>
              <div className="py-4">
                <Textarea
                  value={redoCustomInstructions}
                  onChange={(e) => setRedoCustomInstructions(e.target.value)}
                  placeholder="e.g., 'Focus on the economic arguments' or 'Make the thesis about evolutionary biology' or 'Add specific scientific studies as evidence' or 'Make it more concise - half the length'"
                  className="min-h-[150px] text-sm"
                  data-testid="textarea-redo-custom-instructions"
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                  Examples: "Add real statistics" / "Focus only on the strongest argument" / "Make it half as long" / "Frame it as a philosophical argument"
                </p>
              </div>
              <DialogFooter className="gap-2">
                <Button
                  variant="outline"
                  onClick={() => setShowRedoModal(false)}
                  data-testid="button-cancel-redo"
                >
                  Cancel
                </Button>
                <Button
                  onClick={async () => {
                    setShowRedoModal(false);
                    setValidatorCustomInstructions(redoCustomInstructions);
                    setValidatorLoading(true);
                    try {
                      const response = await fetch("/api/text-model-validator", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          text: validatorInputText,
                          mode: validatorMode,
                          targetDomain: validatorTargetDomain,
                          fidelityLevel: validatorFidelityLevel,
                          mathFramework: validatorMathFramework,
                          constraintType: validatorConstraintType,
                          rigorLevel: validatorRigorLevel,
                          customInstructions: redoCustomInstructions,
                          truthMapping: validatorTruthMapping,
                          mathTruthMapping: validatorMathTruthMapping,
                          literalTruth: validatorLiteralTruth,
                          llmProvider: validatorLLMProvider,
                        }),
                      });
                      const data = await response.json();
                      if (data.success) {
                        setValidatorOutput(data.output);
                        toast({
                          title: "Reconstruction Complete",
                          description: redoCustomInstructions ? "Regenerated with your custom instructions" : "Regenerated with default settings",
                        });
                      } else {
                        toast({
                          title: "Error",
                          description: data.message || "Failed to process",
                          variant: "destructive",
                        });
                      }
                    } catch (error: any) {
                      toast({
                        title: "Error",
                        description: error.message || "Failed to process",
                        variant: "destructive",
                      });
                    } finally {
                      setValidatorLoading(false);
                    }
                  }}
                  className="bg-amber-600 hover:bg-amber-700 text-white"
                  data-testid="button-confirm-redo"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Regenerate
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* COHERENCE METER - Analyze and Improve Text Coherence */}
      <div className="mt-16 bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/10 dark:to-blue-900/10 p-8 rounded-lg border-2 border-indigo-200 dark:border-indigo-700">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-indigo-900 dark:text-indigo-100 mb-3 flex items-center justify-center gap-3">
              <BarChart3 className="w-8 h-8 text-indigo-600" />
              Coherence Meter
            </h2>
            <p className="text-lg text-gray-700 dark:text-gray-300 mb-2">
              Analyze and improve text coherence across multiple dimensions
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Evaluate logical, scientific, thematic, instructional, or motivational coherence - then get rewrites that maximize it
            </p>
          </div>

          {/* Input Area */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-semibold text-indigo-800 dark:text-indigo-200">
                Input Text (500 word limit)
              </label>
              <div className="flex items-center gap-3">
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Word Count: {coherenceInputText.trim() ? coherenceInputText.trim().split(/\s+/).length : 0} / 500
                </span>
                <label className="cursor-pointer">
                  <input
                    type="file"
                    accept=".pdf,.doc,.docx,.txt"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleFileUpload(file, setCoherenceInputText);
                    }}
                    data-testid="input-coherence-upload"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-indigo-300 hover:bg-indigo-50 dark:hover:bg-indigo-900/20"
                    onClick={(e) => {
                      e.preventDefault();
                      (e.currentTarget.previousElementSibling as HTMLInputElement)?.click();
                    }}
                    data-testid="button-coherence-upload"
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    Upload Document
                  </Button>
                </label>
              </div>
            </div>
            <Textarea
              value={coherenceInputText}
              onChange={(e) => setCoherenceInputText(e.target.value)}
              placeholder="Paste your text here to analyze coherence... (under 500 words)"
              className="min-h-[200px] font-mono text-sm"
              data-testid="textarea-coherence-input"
            />
          </div>

          {/* Chunk Selector - appears when text > 500 words */}
          {showCoherenceChunkSelector && coherenceChunks.length > 0 && (
            <div className="mb-6 bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-lg border-2 border-yellow-400 dark:border-yellow-600">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-yellow-900 dark:text-yellow-100 flex items-center gap-2">
                  <AlertCircle className="w-5 h-5" />
                  Text Too Long - Select Sections to Process
                </h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    if (selectedCoherenceChunks.length === coherenceChunks.length) {
                      setSelectedCoherenceChunks([]);
                    } else {
                      setSelectedCoherenceChunks(coherenceChunks.map(c => c.id));
                    }
                  }}
                  data-testid="button-toggle-all-chunks"
                >
                  {selectedCoherenceChunks.length === coherenceChunks.length ? "Deselect All" : "Select All"}
                </Button>
              </div>
              
              <p className="text-sm text-yellow-800 dark:text-yellow-200 mb-4">
                Your text has been divided into {coherenceChunks.length} sections (~400 words each). Select which sections you want to analyze or rewrite:
              </p>

              <div className="space-y-2 max-h-[300px] overflow-y-auto">
                {coherenceChunks.map((chunk, index) => (
                  <label
                    key={chunk.id}
                    className={`flex items-start gap-3 p-3 rounded border-2 cursor-pointer transition ${
                      selectedCoherenceChunks.includes(chunk.id)
                        ? "bg-indigo-100 dark:bg-indigo-900/40 border-indigo-500"
                        : "bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedCoherenceChunks.includes(chunk.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedCoherenceChunks([...selectedCoherenceChunks, chunk.id]);
                        } else {
                          setSelectedCoherenceChunks(selectedCoherenceChunks.filter(id => id !== chunk.id));
                        }
                      }}
                      className="w-5 h-5 mt-1"
                      data-testid={`checkbox-${chunk.id}`}
                    />
                    <div className="flex-1">
                      <div className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
                        Section {index + 1}
                        <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">
                          (~{chunk.text.split(/\s+/).length} words)
                        </span>
                      </div>
                      <div className="text-sm text-gray-700 dark:text-gray-300 italic">
                        "{chunk.preview}"
                      </div>
                    </div>
                  </label>
                ))}
              </div>

              <div className="mt-4 flex gap-3">
                <Button
                  onClick={() => handleProcessSelectedChunks("analyze")}
                  disabled={coherenceLoading || selectedCoherenceChunks.length === 0}
                  className="flex-1 bg-indigo-600 hover:bg-indigo-700"
                  data-testid="button-analyze-selected-chunks"
                >
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Analyze Selected Sections ({selectedCoherenceChunks.length})
                </Button>

                <Button
                  onClick={() => handleProcessSelectedChunks("rewrite")}
                  disabled={coherenceLoading || selectedCoherenceChunks.length === 0}
                  className="flex-1 bg-purple-600 hover:bg-purple-700"
                  data-testid="button-rewrite-selected-chunks"
                >
                  <Sparkles className="w-4 h-4 mr-2" />
                  Rewrite Selected Sections ({selectedCoherenceChunks.length})
                </Button>

                <Button
                  variant="outline"
                  onClick={() => {
                    setShowCoherenceChunkSelector(false);
                    setCoherenceChunks([]);
                    setSelectedCoherenceChunks([]);
                  }}
                  data-testid="button-cancel-chunks"
                >
                  Cancel
                </Button>
              </div>
            </div>
          )}

          {/* Coherence Type Selection */}
          <div className="mb-6 bg-white dark:bg-gray-800 p-6 rounded-lg border border-indigo-200 dark:border-indigo-700">
            <label className="block text-sm font-semibold text-indigo-800 dark:text-indigo-200 mb-4">
              Select Coherence Type:
            </label>
            <div className="space-y-3">
              <label className="flex items-center gap-3 cursor-pointer hover:bg-indigo-50 dark:hover:bg-indigo-900/20 p-3 rounded transition">
                <input
                  type="radio"
                  name="coherence-type"
                  value="logical-consistency"
                  checked={coherenceType === "logical-consistency"}
                  onChange={(e) => setCoherenceType(e.target.value as any)}
                  className="w-4 h-4 text-indigo-600"
                  data-testid="radio-logical-consistency"
                />
                <div className="flex-1">
                  <span className="font-medium text-gray-900 dark:text-gray-100">Logical Consistency</span>
                  <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">- Non-contradiction only</span>
                </div>
              </label>

              <label className="flex items-center gap-3 cursor-pointer hover:bg-indigo-50 dark:hover:bg-indigo-900/20 p-3 rounded transition">
                <input
                  type="radio"
                  name="coherence-type"
                  value="logical-cohesiveness"
                  checked={coherenceType === "logical-cohesiveness"}
                  onChange={(e) => setCoherenceType(e.target.value as any)}
                  className="w-4 h-4 text-indigo-600"
                  data-testid="radio-logical-cohesiveness"
                />
                <div className="flex-1">
                  <span className="font-medium text-gray-900 dark:text-gray-100">Logical Cohesiveness</span>
                  <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">- Argumentative structure</span>
                </div>
              </label>

              <label className="flex items-center gap-3 cursor-pointer hover:bg-indigo-50 dark:hover:bg-indigo-900/20 p-3 rounded transition">
                <input
                  type="radio"
                  name="coherence-type"
                  value="scientific-explanatory"
                  checked={coherenceType === "scientific-explanatory"}
                  onChange={(e) => setCoherenceType(e.target.value as any)}
                  className="w-4 h-4 text-indigo-600"
                  data-testid="radio-scientific-explanatory"
                />
                <div className="flex-1">
                  <span className="font-medium text-gray-900 dark:text-gray-100">Scientific/Explanatory</span>
                  <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">- Aligns with natural law</span>
                </div>
              </label>

              <label className="flex items-center gap-3 cursor-pointer hover:bg-indigo-50 dark:hover:bg-indigo-900/20 p-3 rounded transition">
                <input
                  type="radio"
                  name="coherence-type"
                  value="thematic-psychological"
                  checked={coherenceType === "thematic-psychological"}
                  onChange={(e) => setCoherenceType(e.target.value as any)}
                  className="w-4 h-4 text-indigo-600"
                  data-testid="radio-thematic-psychological"
                />
                <div className="flex-1">
                  <span className="font-medium text-gray-900 dark:text-gray-100">Thematic/Psychological</span>
                  <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">- Emotional & mood flow</span>
                </div>
              </label>

              <label className="flex items-center gap-3 cursor-pointer hover:bg-indigo-50 dark:hover:bg-indigo-900/20 p-3 rounded transition">
                <input
                  type="radio"
                  name="coherence-type"
                  value="instructional"
                  checked={coherenceType === "instructional"}
                  onChange={(e) => setCoherenceType(e.target.value as any)}
                  className="w-4 h-4 text-indigo-600"
                  data-testid="radio-instructional"
                />
                <div className="flex-1">
                  <span className="font-medium text-gray-900 dark:text-gray-100">Instructional</span>
                  <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">- Clear, actionable message</span>
                </div>
              </label>

              <label className="flex items-center gap-3 cursor-pointer hover:bg-indigo-50 dark:hover:bg-indigo-900/20 p-3 rounded transition">
                <input
                  type="radio"
                  name="coherence-type"
                  value="motivational"
                  checked={coherenceType === "motivational"}
                  onChange={(e) => setCoherenceType(e.target.value as any)}
                  className="w-4 h-4 text-indigo-600"
                  data-testid="radio-motivational"
                />
                <div className="flex-1">
                  <span className="font-medium text-gray-900 dark:text-gray-100">Motivational</span>
                  <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">- Consistent emotional direction</span>
                </div>
              </label>

              <label className="flex items-center gap-3 cursor-pointer hover:bg-indigo-50 dark:hover:bg-indigo-900/20 p-3 rounded transition">
                <input
                  type="radio"
                  name="coherence-type"
                  value="mathematical"
                  checked={coherenceType === "mathematical"}
                  onChange={(e) => setCoherenceType(e.target.value as any)}
                  className="w-4 h-4 text-indigo-600"
                  data-testid="radio-mathematical"
                />
                <div className="flex-1">
                  <span className="font-medium text-gray-900 dark:text-gray-100">Mathematical (Proof Validity)</span>
                  <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">- Rigorous proof checking</span>
                </div>
              </label>

              <label className="flex items-center gap-3 cursor-pointer hover:bg-indigo-50 dark:hover:bg-indigo-900/20 p-3 rounded transition">
                <input
                  type="radio"
                  name="coherence-type"
                  value="philosophical"
                  checked={coherenceType === "philosophical"}
                  onChange={(e) => setCoherenceType(e.target.value as any)}
                  className="w-4 h-4 text-indigo-600"
                  data-testid="radio-philosophical"
                />
                <div className="flex-1">
                  <span className="font-medium text-gray-900 dark:text-gray-100">Philosophical (Conceptual Rigor)</span>
                  <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">- Conceptual consistency and dialectical engagement</span>
                </div>
              </label>

              <label className="flex items-center gap-3 cursor-pointer hover:bg-indigo-50 dark:hover:bg-indigo-900/20 p-3 rounded transition">
                <input
                  type="radio"
                  name="coherence-type"
                  value="auto-detect"
                  checked={coherenceType === "auto-detect"}
                  onChange={(e) => setCoherenceType(e.target.value as any)}
                  className="w-4 h-4 text-indigo-600"
                  data-testid="radio-auto-detect"
                />
                <div className="flex-1">
                  <span className="font-medium text-gray-900 dark:text-gray-100">Auto-Detect</span>
                  <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">- Let AI determine type</span>
                </div>
              </label>
            </div>
          </div>

          {/* Processing Mode Selection */}
          <div className="mb-6 bg-white dark:bg-gray-800 p-6 rounded-lg border border-indigo-200 dark:border-indigo-700">
            <label className="block text-sm font-semibold text-indigo-800 dark:text-indigo-200 mb-4">
              Processing Mode for Long Texts (&gt;500 words):
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <label 
                className={`flex items-start gap-3 cursor-pointer p-5 rounded border-2 transition ${
                  coherenceProcessingMode === "simple" 
                    ? "bg-indigo-100 dark:bg-indigo-900/40 border-indigo-500" 
                    : "bg-white dark:bg-gray-800 border-indigo-200 dark:border-indigo-700 hover:bg-indigo-50 dark:hover:bg-indigo-900/20"
                }`}
              >
                <input
                  type="radio"
                  name="coherence-processing-mode"
                  value="simple"
                  checked={coherenceProcessingMode === "simple"}
                  onChange={(e) => setCoherenceProcessingMode(e.target.value as any)}
                  className="w-5 h-5 text-indigo-600 mt-1"
                  data-testid="radio-mode-simple"
                />
                <div className="flex-1">
                  <span className="font-semibold text-gray-900 dark:text-gray-100 block mb-2 text-lg">⚡ Simple Chunking</span>
                  <span className="text-sm text-gray-700 dark:text-gray-300 block mb-2">Process sections independently for speed</span>
                  <span className="text-xs text-gray-600 dark:text-gray-400">• Faster processing<br/>• Good for quick analysis<br/>• Each section processed separately</span>
                </div>
              </label>

              <label 
                className={`flex items-start gap-3 cursor-pointer p-5 rounded border-2 transition ${
                  coherenceProcessingMode === "outline-guided" 
                    ? "bg-indigo-100 dark:bg-indigo-900/40 border-indigo-500" 
                    : "bg-white dark:bg-gray-800 border-indigo-200 dark:border-indigo-700 hover:bg-indigo-50 dark:hover:bg-indigo-900/20"
                }`}
              >
                <input
                  type="radio"
                  name="coherence-processing-mode"
                  value="outline-guided"
                  checked={coherenceProcessingMode === "outline-guided"}
                  onChange={(e) => setCoherenceProcessingMode(e.target.value as any)}
                  className="w-5 h-5 text-indigo-600 mt-1"
                  data-testid="radio-mode-outline"
                />
                <div className="flex-1">
                  <span className="font-semibold text-gray-900 dark:text-gray-100 block mb-2 text-lg">🎯 Outline-Guided (Recommended)</span>
                  <span className="text-sm text-gray-700 dark:text-gray-300 block mb-2">Two-stage process for maximum global coherence</span>
                  <span className="text-xs text-gray-600 dark:text-gray-400">• Creates coherent outline first<br/>• Rewrites sections to align with outline<br/>• Better consistency across entire document</span>
                </div>
              </label>
            </div>
          </div>

          {/* Rewrite Aggressiveness Selection */}
          <div className="mb-6 bg-white dark:bg-gray-800 p-6 rounded-lg border border-purple-200 dark:border-purple-700">
            <label className="block text-sm font-semibold text-purple-800 dark:text-purple-200 mb-4">
              Rewrite Aggressiveness (for "Rewrite to Max Coherence"):
            </label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <label 
                className={`flex items-start gap-3 cursor-pointer p-4 rounded border-2 transition ${
                  coherenceAggressiveness === "conservative" 
                    ? "bg-purple-100 dark:bg-purple-900/40 border-purple-500" 
                    : "bg-white dark:bg-gray-800 border-purple-200 dark:border-purple-700 hover:bg-purple-50 dark:hover:bg-purple-900/20"
                }`}
              >
                <input
                  type="radio"
                  name="coherence-aggressiveness"
                  value="conservative"
                  checked={coherenceAggressiveness === "conservative"}
                  onChange={(e) => setCoherenceAggressiveness(e.target.value as any)}
                  className="w-4 h-4 text-purple-600 mt-1"
                  data-testid="radio-aggressiveness-conservative"
                />
                <div className="flex-1">
                  <span className="font-semibold text-gray-900 dark:text-gray-100 block mb-1">Conservative</span>
                  <span className="text-xs text-gray-600 dark:text-gray-400">Minimal changes - preserve original structure and wording as much as possible</span>
                </div>
              </label>

              <label 
                className={`flex items-start gap-3 cursor-pointer p-4 rounded border-2 transition ${
                  coherenceAggressiveness === "moderate" 
                    ? "bg-purple-100 dark:bg-purple-900/40 border-purple-500" 
                    : "bg-white dark:bg-gray-800 border-purple-200 dark:border-purple-700 hover:bg-purple-50 dark:hover:bg-purple-900/20"
                }`}
              >
                <input
                  type="radio"
                  name="coherence-aggressiveness"
                  value="moderate"
                  checked={coherenceAggressiveness === "moderate"}
                  onChange={(e) => setCoherenceAggressiveness(e.target.value as any)}
                  className="w-4 h-4 text-purple-600 mt-1"
                  data-testid="radio-aggressiveness-moderate"
                />
                <div className="flex-1">
                  <span className="font-semibold text-gray-900 dark:text-gray-100 block mb-1">Moderate</span>
                  <span className="text-xs text-gray-600 dark:text-gray-400">Fix major issues - add missing arguments and clarify key points</span>
                </div>
              </label>

              <label 
                className={`flex items-start gap-3 cursor-pointer p-4 rounded border-2 transition ${
                  coherenceAggressiveness === "aggressive" 
                    ? "bg-purple-100 dark:bg-purple-900/40 border-purple-500" 
                    : "bg-white dark:bg-gray-800 border-purple-200 dark:border-purple-700 hover:bg-purple-50 dark:hover:bg-purple-900/20"
                }`}
              >
                <input
                  type="radio"
                  name="coherence-aggressiveness"
                  value="aggressive"
                  checked={coherenceAggressiveness === "aggressive"}
                  onChange={(e) => setCoherenceAggressiveness(e.target.value as any)}
                  className="w-4 h-4 text-purple-600 mt-1"
                  data-testid="radio-aggressiveness-aggressive"
                />
                <div className="flex-1">
                  <span className="font-semibold text-gray-900 dark:text-gray-100 block mb-1">Aggressive ⚡</span>
                  <span className="text-xs text-gray-600 dark:text-gray-400">Maximize coherence - expand significantly, restructure completely, add extensive context for 9-10/10 score</span>
                </div>
              </label>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-4 mb-6">
            <Button
              onClick={handleCoherenceAnalyze}
              disabled={coherenceLoading}
              className="bg-indigo-600 hover:bg-indigo-700 text-white px-8 py-6 text-lg flex-1 min-w-[200px]"
              data-testid="button-analyze-coherence"
            >
              {coherenceLoading && coherenceMode === "analyze" ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <BarChart3 className="w-5 h-5 mr-2" />
                  ANALYZE COHERENCE
                </>
              )}
            </Button>

            <Button
              onClick={handleCoherenceRewrite}
              disabled={coherenceLoading}
              className="bg-purple-600 hover:bg-purple-700 text-white px-8 py-6 text-lg flex-1 min-w-[200px]"
              data-testid="button-rewrite-coherence"
            >
              {coherenceLoading && coherenceMode === "rewrite" ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Rewriting...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5 mr-2" />
                  REWRITE TO MAX COHERENCE
                </>
              )}
            </Button>

            <Button
              onClick={handleCoherenceClear}
              variant="outline"
              className="border-indigo-300 hover:bg-indigo-50 dark:hover:bg-indigo-900/20"
              data-testid="button-clear-coherence"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Clear All
            </Button>
          </div>

          {/* Stage Progress Indicator */}
          {coherenceLoading && coherenceStageProgress && (
            <div className="mt-8 bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border-2 border-blue-400 dark:border-blue-600">
              <div className="flex items-center gap-3 mb-4">
                <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
                <h3 className="text-lg font-bold text-blue-900 dark:text-blue-100">Processing Long Document...</h3>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded border border-blue-300 dark:border-blue-700">
                <pre className="whitespace-pre-wrap font-mono text-sm text-gray-800 dark:text-gray-200">
                  {coherenceStageProgress}
                </pre>
              </div>
            </div>
          )}

          {/* Analysis Output */}
          {coherenceMode === "analyze" && coherenceAnalysis && (
            <div className="mt-8 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-bold text-indigo-900 dark:text-indigo-100">Coherence Analysis</h3>
                <div className="flex items-center gap-4">
                  {coherenceScore !== null && (
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">Score:</span>
                      <Badge className={`text-lg px-3 py-1 ${
                        coherenceAssessment === "PASS" ? "bg-green-600" :
                        coherenceAssessment === "WEAK" ? "bg-yellow-600" :
                        "bg-red-600"
                      }`}>
                        {coherenceScore}/10 - {coherenceAssessment}
                      </Badge>
                    </div>
                  )}
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleDownloadText(coherenceAnalysis, `coherence-analysis-${coherenceType}.txt`)}
                      data-testid="button-download-coherence-analysis"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </Button>
                    <CopyButton text={coherenceAnalysis} />
                    <SendToButton
                      text={coherenceAnalysis}
                      onSendToIntelligence={(text) => {
                        setDocumentA({ ...documentA, content: text });
                        toast({
                          title: "Analysis sent to Intelligence Analysis",
                          description: "Coherence analysis has been sent to the intelligence analysis input"
                        });
                      }}
                      onSendToHumanizer={(text) => {
                        setBoxA(text);
                        toast({
                          title: "Analysis sent to Humanizer",
                          description: "Coherence analysis has been sent to the Humanizer input box"
                        });
                      }}
                      onSendToChat={(text) => {
                        toast({
                          title: "Analysis available to Chat",
                          description: "The coherence analysis is now available as context for AI chat"
                        });
                      }}
                    />
                  </div>
                </div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-indigo-200 dark:border-indigo-700">
                <pre className="whitespace-pre-wrap font-sans text-sm text-gray-800 dark:text-gray-200">
                  {coherenceAnalysis}
                </pre>
              </div>
            </div>
          )}

          {/* Rewrite Output */}
          {coherenceMode === "rewrite" && coherenceRewrite && (
            <div className="mt-8 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-bold text-indigo-900 dark:text-indigo-100">Rewritten Version</h3>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleDownloadText(coherenceRewrite, `coherence-rewrite-${coherenceType}.txt`)}
                    data-testid="button-download-coherence-rewrite"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                  <CopyButton text={coherenceRewrite} />
                  <SendToButton
                    text={coherenceRewrite}
                    onSendToIntelligence={(text) => {
                      setDocumentA({ ...documentA, content: text });
                      toast({
                        title: "Text sent to Intelligence Analysis",
                        description: "Rewritten text has been sent to the intelligence analysis input"
                      });
                    }}
                    onSendToHumanizer={(text) => {
                      setBoxA(text);
                      toast({
                        title: "Text sent to Humanizer",
                        description: "Rewritten text has been sent to the Humanizer input box"
                      });
                    }}
                    onSendToChat={(text) => {
                      toast({
                        title: "Text available to Chat",
                        description: "The rewritten text is now available as context for AI chat"
                      });
                    }}
                  />
                </div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-indigo-200 dark:border-indigo-700 max-h-96 overflow-y-auto">
                <pre className="whitespace-pre-wrap font-sans text-sm text-gray-800 dark:text-gray-200">
                  {coherenceRewrite}
                </pre>
              </div>

              {/* Changes Made */}
              {coherenceChanges && (
                <div className="mt-6">
                  <h4 className="text-lg font-semibold text-indigo-900 dark:text-indigo-100 mb-3">Changes Made</h4>
                  <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-lg border border-indigo-200 dark:border-indigo-700">
                    <pre className="whitespace-pre-wrap font-sans text-sm text-gray-800 dark:text-gray-200">
                      {coherenceChanges}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Loading State */}
          {coherenceLoading && (
            <div className="flex flex-col items-center justify-center py-12">
              <Loader2 className="w-12 h-12 animate-spin text-indigo-600 mb-4" />
              <p className="text-gray-600 dark:text-gray-400">
                {coherenceMode === "analyze" ? "Analyzing text coherence..." : "Rewriting text for maximum coherence..."}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* GPT BYPASS HUMANIZER - Following Exact Protocol - HIDDEN BY USER REQUEST */}
      <div className="hidden mt-16 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/10 dark:to-purple-900/10 p-8 rounded-lg border-2 border-blue-200 dark:border-blue-700">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-blue-900 dark:text-blue-100 mb-3 flex items-center justify-center gap-3">
              <Shield className="w-8 h-8 text-blue-600" />
              GPT Bypass Humanizer
            </h2>
            <p className="text-lg text-gray-700 dark:text-gray-300 mb-2">
              Transform AI text into undetectable human writing with surgical precision
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Box A: AI Text → Box B: Human Style Sample → Box C: Humanized Output
            </p>
          </div>

          <div className="grid gap-8 lg:grid-cols-4">
            {/* Left Column - Writing Samples & Style Presets */}
            <div className="lg:col-span-1 space-y-6">
              {/* Writing Samples Dropdown */}
              <div className="space-y-3">
                <label className="block text-sm font-semibold text-blue-800 dark:text-blue-200">
                  Writing Samples
                </label>
                <Select value={selectedWritingSample} onValueChange={(value) => {
                  setSelectedWritingSample(value);
                  const [category, sample] = value.split('|');
                  if (writingSamples[category] && writingSamples[category][sample]) {
                    setBoxB(writingSamples[category][sample]);
                  }
                }}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Choose writing sample...">
                      {selectedWritingSample ? selectedWritingSample.split('|')[1] : "Choose writing sample..."}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent className="max-h-80">
                    {Object.entries(writingSamples).map(([category, samples]) => (
                      <div key={category}>
                        <div className="px-2 py-1 text-xs font-bold text-gray-700 dark:text-gray-300 uppercase bg-gray-100 dark:bg-gray-800">
                          {category.replace(/([A-Z])/g, ' $1').trim()}
                        </div>
                        {Object.keys(samples as object).map((sampleName) => (
                          <SelectItem key={`${category}|${sampleName}`} value={`${category}|${sampleName}`}>
                            {sampleName}
                          </SelectItem>
                        ))}
                      </div>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Style Presets */}
              <div className="space-y-3">
                <label className="block text-sm font-semibold text-blue-800 dark:text-blue-200">
                  Style Presets
                </label>
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-blue-200 dark:border-blue-700 max-h-96 overflow-y-auto">
                  {/* Most Important (1-8) */}
                  <div className="mb-4">
                    <h4 className="text-xs font-bold text-green-700 dark:text-green-300 mb-2 uppercase bg-green-50 dark:bg-green-900/20 p-2 rounded">
                      ⭐ Most Important for Humanizing (1-8)
                    </h4>
                    <div className="space-y-2">
                      {[
                        "Mixed cadence + clause sprawl",
                        "Asymmetric emphasis", 
                        "One aside",
                        "Hedge twice",
                        "Local disfluency",
                        "Analogy injection",
                        "Topic snap",
                        "Friction detail"
                      ].map((preset) => (
                        <label key={preset} className="flex items-start gap-2 text-xs cursor-pointer hover:bg-green-50 dark:hover:bg-green-900/10 p-1 rounded">
                          <input
                            type="checkbox"
                            checked={selectedStylePresets.includes(preset)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedStylePresets([...selectedStylePresets, preset]);
                              } else {
                                setSelectedStylePresets(selectedStylePresets.filter(p => p !== preset));
                              }
                            }}
                            className="mt-0.5"
                          />
                          <div>
                            <div className="font-medium text-gray-900 dark:text-gray-100">{preset}</div>
                            <div className="text-gray-600 dark:text-gray-400">{stylePresets[preset]}</div>
                          </div>
                        </label>
                      ))}
                    </div>
                  </div>

                  {/* Other Style Techniques */}
                  <div>
                    <h4 className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2 uppercase">Additional Techniques</h4>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {Object.entries(stylePresets).filter(([preset]) => ![
                        "Mixed cadence + clause sprawl",
                        "Asymmetric emphasis", 
                        "One aside",
                        "Hedge twice",
                        "Local disfluency",
                        "Analogy injection",
                        "Topic snap",
                        "Friction detail"
                      ].includes(preset)).map(([preset, description]) => (
                        <label key={preset} className="flex items-start gap-2 text-xs cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 p-1 rounded">
                          <input
                            type="checkbox"
                            checked={selectedStylePresets.includes(preset)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedStylePresets([...selectedStylePresets, preset]);
                              } else {
                                setSelectedStylePresets(selectedStylePresets.filter(p => p !== preset));
                              }
                            }}
                            className="mt-0.5"
                          />
                          <div>
                            <div className="font-medium text-gray-900 dark:text-gray-100">{preset}</div>
                            <div className="text-gray-600 dark:text-gray-400">{String(description)}</div>
                          </div>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* LLM Provider Selection */}
              <div className="space-y-3">
                <label className="block text-sm font-semibold text-blue-800 dark:text-blue-200">
                  AI Provider
                </label>
                <Select value={humanizerProvider} onValueChange={(value) => setHumanizerProvider(value as LLMProvider)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="zhi2">ZHI 2 - Default</SelectItem>
                    <SelectItem value="zhi1">ZHI 1</SelectItem>
                    <SelectItem value="zhi3">ZHI 3</SelectItem>
                    <SelectItem value="zhi4">ZHI 4</SelectItem>
                    <SelectItem value="zhi5">ZHI 5</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Center & Right Columns - Main Boxes */}
            <div className="lg:col-span-3 space-y-6">
              {/* Top Row - Box A and Box B */}
              <div className="grid gap-6 md:grid-cols-2">
                {/* Box A - AI Text Input */}
                <div className="space-y-3">
                  <label className="block text-sm font-semibold text-blue-800 dark:text-blue-200">
                    Box A - AI-Generated Text to Humanize
                    {boxAScore !== null && (
                      <span className={`ml-2 px-3 py-1 text-sm rounded font-bold ${
                        boxAScore >= 70 ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 
                        boxAScore >= 50 ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' : 
                        'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                      }`}>
                        {boxAScore}% HUMAN
                      </span>
                    )}
                  </label>
                  <div className="relative">
                    <Textarea
                      value={boxA}
                      onChange={(e) => {
                        setBoxA(e.target.value);
                        if (e.target.value.length > 100) {
                          debounce(() => evaluateTextAI(e.target.value, setBoxAScore), 2000)();
                        }
                      }}
                      placeholder="Paste or upload AI-generated text here that needs to be humanized..."
                      className="min-h-[300px] border-blue-200 dark:border-blue-700 focus:border-blue-500 dark:focus:border-blue-400 pr-12"
                      data-testid="textarea-box-a"
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute top-2 right-2 hover:bg-blue-100 dark:hover:bg-blue-800"
                      onClick={() => {
                        document.getElementById('file-upload-a')?.click();
                      }}
                      data-testid="button-upload-box-a"
                    >
                      <Upload className="w-4 h-4" />
                    </Button>
                    <input
                      id="file-upload-a"
                      type="file"
                      accept=".pdf,.doc,.docx,.txt"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleFileUpload(file, setBoxA);
                      }}
                    />
                  </div>
                  
                  {/* Chunk Text Button for Large Documents */}
                  {boxA.length > 3000 && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleChunkText(boxA)}
                      className="text-xs border-blue-300 text-blue-700 hover:bg-blue-50"
                      data-testid="button-chunk-box-a"
                    >
                      <FileText className="w-3 h-3 mr-1" />
                      Chunk Large Text (500 words)
                    </Button>
                  )}
                </div>

                {/* Box B - Human Style Sample */}
                <div className="space-y-3">
                  <label className="block text-sm font-semibold text-blue-800 dark:text-blue-200">
                    Box B - Human Writing Style Sample
                    {boxBScore !== null && (
                      <span className={`ml-2 px-3 py-1 text-sm rounded font-bold ${
                        boxBScore >= 70 ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 
                        boxBScore >= 50 ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' : 
                        'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                      }`}>
                        {boxBScore}% HUMAN
                      </span>
                    )}
                  </label>
                  <div className="relative">
                    <Textarea
                      value={boxB}
                      onChange={(e) => {
                        setBoxB(e.target.value);
                        if (e.target.value.length > 100) {
                          debounce(() => evaluateTextAI(e.target.value, setBoxBScore), 2000)();
                        }
                      }}
                      placeholder="Paste or upload human-written text whose style you want to mimic..."
                      className="min-h-[300px] border-blue-200 dark:border-blue-700 focus:border-blue-500 dark:focus:border-blue-400 pr-12"
                      data-testid="textarea-box-b"
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute top-2 right-2 hover:bg-blue-100 dark:hover:bg-blue-800"
                      onClick={() => {
                        document.getElementById('file-upload-b')?.click();
                      }}
                      data-testid="button-upload-box-b"
                    >
                      <Upload className="w-4 h-4" />
                    </Button>
                    <input
                      id="file-upload-b"
                      type="file"
                      accept=".pdf,.doc,.docx,.txt"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleFileUpload(file, setBoxB);
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Custom Instructions Box - Under Box A */}
              <div className="space-y-3">
                <label className="block text-sm font-semibold text-blue-800 dark:text-blue-200">
                  Custom Instructions (Optional)
                </label>
                <Textarea
                  value={humanizerCustomInstructions}
                  onChange={(e) => setHumanizerCustomInstructions(e.target.value)}
                  placeholder="Enter specific instructions for the rewrite (e.g., 'maintain technical terminology', 'use more casual tone', 'preserve all statistics')..."
                  className="min-h-[120px] border-blue-200 dark:border-blue-700 focus:border-blue-500 dark:focus:border-blue-400"
                  rows={4}
                  data-testid="textarea-custom-instructions"
                />
              </div>

              {/* Action Button */}
              <div className="flex justify-center">
                <Button
                  onClick={handleHumanize}
                  disabled={isHumanizerLoading || !boxA.trim() || !boxB.trim()}
                  className="px-12 py-4 bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50 text-lg font-semibold"
                  data-testid="button-humanize"
                >
                  {isHumanizerLoading ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-3 animate-spin" />
                      Humanizing with Surgical Precision...
                    </>
                  ) : (
                    <>
                      <Shield className="w-5 h-5 mr-3" />
                      Humanize Text
                    </>
                  )}
                </Button>
              </div>

              {/* Box C - Large Output */}
              {boxC && (
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <label className="block text-sm font-semibold text-blue-800 dark:text-blue-200">
                      Box C - Humanized Output
                      {boxCScore !== null && (
                        <span className={`ml-2 px-3 py-1 text-sm rounded font-bold ${
                          boxCScore >= 70 ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 
                          boxCScore >= 50 ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' : 
                          'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                        }`}>
                          {boxCScore}% HUMAN
                        </span>
                      )}
                    </label>
                    <div className="flex gap-2">
                      <CopyButton text={boxC} />
                      <SendToButton 
                        text={boxC}
                        onSendToIntelligence={handleSendToIntelligence}
                        onSendToChat={handleSendToChat}
                      />
                    </div>
                  </div>
                  <Textarea
                    value={boxC}
                    onChange={(e) => setBoxC(e.target.value)}
                    className="min-h-[500px] border-green-200 dark:border-green-700 focus:border-green-500 dark:focus:border-green-400 bg-green-50/50 dark:bg-green-900/10"
                    data-testid="textarea-box-c"
                    readOnly
                  />
                  
                  {/* Re-rewrite Function & Download Options - Under Box C */}
                  <div className="flex flex-wrap gap-3 justify-between items-center bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                    <div className="flex gap-3">
                      <Button
                        onClick={handleReRewrite}
                        disabled={isReRewriteLoading || !boxC.trim()}
                        variant="outline"
                        className="border-orange-300 text-orange-700 hover:bg-orange-50 dark:border-orange-600 dark:text-orange-300"
                        data-testid="button-re-rewrite"
                      >
                        {isReRewriteLoading ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Re-rewriting...
                          </>
                        ) : (
                          <>
                            <RefreshCw className="w-4 h-4 mr-2" />
                            Re-rewrite (Recursive)
                          </>
                        )}
                      </Button>
                      <Button
                        onClick={() => {
                          setBoxA("");
                          setBoxB("");
                          setBoxC("");
                          setBoxAScore(null);
                          setBoxBScore(null);
                          setBoxCScore(null);
                          setHumanizerCustomInstructions("");
                          setSelectedStylePresets([]);
                        }}
                        variant="outline"
                        className="border-gray-300 text-gray-700 hover:bg-gray-50"
                        data-testid="button-clear-all"
                      >
                        <Trash2 className="w-4 h-4 mr-2" />
                        Clear All
                      </Button>
                    </div>
                    
                    <div className="flex gap-2">
                      <Button
                        onClick={() => downloadHumanizerResult('txt')}
                        variant="outline"
                        size="sm"
                        className="text-xs"
                        data-testid="button-download-txt"
                      >
                        <Download className="w-3 h-3 mr-1" />
                        TXT
                      </Button>
                      <Button
                        onClick={() => downloadHumanizerResult('pdf')}
                        variant="outline"
                        size="sm"
                        className="text-xs"
                        data-testid="button-download-pdf"
                      >
                        <Download className="w-3 h-3 mr-1" />
                        PDF
                      </Button>
                      <Button
                        onClick={() => downloadHumanizerResult('docx')}
                        variant="outline"
                        size="sm"
                        className="text-xs"
                        data-testid="button-download-docx"
                      >
                        <Download className="w-3 h-3 mr-1" />
                        Word
                      </Button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Chat Dialog - Always visible below everything */}
      <ChatDialog 
        currentDocument={documentA.content}
        analysisResults={mode === "single" ? analysisA : comparison}
        onSendToInput={(content: string) => {
          setDocumentA({ ...documentA, content: content });
        }}
        onSendToHumanizer={handleSendToHumanizer}
        onSendToIntelligence={handleSendToIntelligence}
        onSendToChat={handleSendToChat}
        onSendToValidator={(text: string) => setValidatorInputText(text)}
      />

      {/* Fiction Assessment Popup */}
      <FictionAssessmentPopup 
        isOpen={fictionPopupOpen}
        onClose={() => setFictionPopupOpen(false)}
      />
    </div>
  );
};

export default HomePage;
