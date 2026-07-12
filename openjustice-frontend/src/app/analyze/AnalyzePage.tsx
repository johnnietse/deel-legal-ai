import { useState, useRef, useCallback } from "react";
import {
  Upload,
  FileText,
  AlertCircle,
  CheckCircle2,
  Loader2,
  X,
  Building2,
  User,
  Gavel,
  BookOpen,
  Scale,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useDocumentAnalysis } from "@/lib/hooks/useQuery";
import { cn } from "@/lib/utils";
import type { DocumentAnalysis, LegalEntity } from "@/types";

export function AnalyzePage() {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [showText, setShowText] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const analyzeMutation = useDocumentAnalysis();

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile && droppedFile.type === "application/pdf") {
      setFile(droppedFile);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
    }
  };

  const handleAnalyze = () => {
    if (file) {
      analyzeMutation.mutate(file);
    }
  };

  const result = analyzeMutation.data;

  const entityIcon = (type: string) => {
    switch (type) {
      case "person": return <User className="h-4 w-4" />;
      case "organization": return <Building2 className="h-4 w-4" />;
      case "court": return <Gavel className="h-4 w-4" />;
      case "statute": return <BookOpen className="h-4 w-4" />;
      default: return <FileText className="h-4 w-4" />;
    }
  };

  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-2xl sm:text-3xl font-bold text-surface-900 dark:text-surface-100">
          Document Analysis
        </h1>
        <p className="text-surface-500 dark:text-surface-400 mt-1">
          Upload employment agreements and contracts for AI-powered analysis
        </p>
      </div>

      <div className="grid grid-cols-1 gap-8">
        {/* Upload Area */}
        <Card>
          <CardContent className="p-6">
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
              className={cn(
                "border-2 border-dashed rounded-xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-200",
                dragActive
                  ? "border-primary-500 bg-primary-50 dark:bg-primary-950/30"
                  : file
                  ? "border-primary-300 bg-primary-50/50 dark:bg-primary-950/20"
                  : "border-surface-300 dark:border-surface-600 hover:border-primary-400 hover:bg-surface-50 dark:hover:bg-surface-800/50"
              )}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={handleFileChange}
              />

              {file ? (
                <div className="flex flex-col items-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary-100 dark:bg-primary-900/30 mb-4">
                    <FileText className="h-6 w-6 text-primary-600 dark:text-primary-400" />
                  </div>
                  <p className="text-sm font-medium text-surface-900 dark:text-surface-100 mb-1">{file.name}</p>
                  <p className="text-xs text-surface-400 mb-4">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
                  <div className="flex items-center gap-3">
                    <Button onClick={(e) => { e.stopPropagation(); handleAnalyze(); }} loading={analyzeMutation.isPending}>
                      {analyzeMutation.isPending ? "Analyzing..." : "Analyze Document"}
                    </Button>
                    <Button variant="ghost" size="sm" onClick={(e) => { e.stopPropagation(); setFile(null); }}>
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-surface-100 dark:bg-surface-800 mb-4">
                    <Upload className="h-6 w-6 text-surface-400" />
                  </div>
                  <p className="text-sm font-medium text-surface-700 dark:text-surface-300 mb-1">
                    Drop your PDF here, or click to browse
                  </p>
                  <p className="text-xs text-surface-400">PDF files only, up to 50MB</p>
                </div>
              )}
            </div>

            {/* Upload Progress */}
            {analyzeMutation.isPending && (
              <div className="mt-4">
                <div className="flex items-center gap-3 text-sm text-surface-500">
                  <Loader2 className="h-4 w-4 animate-spin text-primary-500" />
                  <span>Processing document...</span>
                </div>
                <div className="mt-2 h-1.5 w-full bg-surface-100 dark:bg-surface-800 rounded-full overflow-hidden">
                  <div className="h-full bg-primary-500 rounded-full animate-pulse" style={{ width: "60%" }} />
                </div>
              </div>
            )}

            {analyzeMutation.isError && (
              <div className="mt-4 p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
                <div className="flex items-center gap-2">
                  <AlertCircle className="h-4 w-4 text-red-500" />
                  <span className="text-sm text-red-600 dark:text-red-400">
                    {analyzeMutation.error?.message || "Analysis failed. Please try again."}
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Status */}
            <Card>
              <CardContent className="p-6 flex items-center gap-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/30">
                  <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400" />
                </div>
                <div>
                  <p className="font-medium text-surface-900 dark:text-surface-100">Analysis Complete</p>
                  <p className="text-sm text-surface-500">{result.filename}</p>
                </div>
                {result.classification_analysis?.is_employment_related && (
                  <Badge variant="success" className="ml-auto">
                    Employment Related
                  </Badge>
                )}
              </CardContent>
            </Card>

            {/* Summary */}
            <Card>
              <CardHeader>
                <CardTitle>Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed">
                  {result.summary}
                </p>
              </CardContent>
            </Card>

            {/* Entities */}
            <Card>
              <CardHeader>
                <CardTitle>Identified Entities</CardTitle>
                <CardDescription>Key legal entities found in the document</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {result.entities.map((entity, idx) => (
                    <div key={idx} className="flex items-center gap-3 p-3 rounded-lg bg-surface-50 dark:bg-surface-800/50 border border-surface-200 dark:border-surface-700">
                      <div className="p-2 rounded-lg bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400">
                        {entityIcon(entity.type)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-surface-700 dark:text-surface-300 truncate">
                          {entity.name}
                        </p>
                        <p className="text-xs text-surface-400 capitalize">{entity.type}</p>
                      </div>
                      <Badge variant="secondary" className="text-xs">{entity.mentions} mentions</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Extracted Text Preview */}
            <Card>
              <CardHeader>
                <button
                  onClick={() => setShowText(!showText)}
                  className="flex items-center justify-between w-full text-left"
                >
                  <CardTitle>Extracted Text</CardTitle>
                  {showText ? <ChevronUp className="h-5 w-5 text-surface-400" /> : <ChevronDown className="h-5 w-5 text-surface-400" />}
                </button>
              </CardHeader>
              {showText && (
                <CardContent>
                  <pre className="text-sm text-surface-600 dark:text-surface-400 whitespace-pre-wrap font-sans leading-relaxed bg-surface-50 dark:bg-surface-800 p-4 rounded-lg">
                    {result.extracted_text}
                  </pre>
                </CardContent>
              )}
            </Card>

            {/* Classification Analysis */}
            {result.classification_analysis?.is_employment_related && (
              <Card className="border-primary-200 dark:border-primary-800">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Scale className="h-5 w-5 text-primary-500" />
                    <div>
                      <CardTitle>Classification Analysis</CardTitle>
                      <CardDescription>Employment-related document detected</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <p className="text-sm text-surface-500">Predicted Classification</p>
                      <p className="text-lg font-bold text-surface-900 dark:text-surface-100">
                        {result.classification_analysis.prediction || "N/A"}
                      </p>
                    </div>
                    {result.classification_analysis.confidence && (
                      <div className="text-right">
                        <p className="text-sm text-surface-500">Confidence</p>
                        <p className="text-lg font-bold text-primary-500">
                          {(result.classification_analysis.confidence * 100).toFixed(0)}%
                        </p>
                      </div>
                    )}
                    <Button variant="outline" size="sm" onClick={() => window.open("/classify", "_self")}>
                      Full Classification
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
