import { useState } from "react";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  Scale,
  Info,
  AlertTriangle,
  CheckCircle2,
  BarChart3,
  TreePine,
  Clock,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip } from "@/components/ui/tooltip";
import { useClassification } from "@/lib/hooks/useQuery";
import { FACTOR_DETAILS } from "@/lib/api/client";
import { cn, riskColor } from "@/lib/utils";
import type { FactorValue } from "@/types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const factorKeys = [
  "supervision_review",
  "ability_hire",
  "delegation_tasks",
  "ownership_tools",
  "chance_profit",
  "risk_loss",
  "exclusivity_services",
  "work_hours_setter",
  "work_location",
  "uniform_required",
] as const;

const classifySchema = z.object({
  ...Object.fromEntries(
    factorKeys.map((key) => [key, z.enum(["Employee", "Contractor", "Ambiguous", "Unknown"])])
  ),
  jurisdiction: z.string().min(1, "Select a jurisdiction"),
  facts: z.string().optional(),
});

type ClassifyForm = z.infer<typeof classifySchema>;

const factorOptions = [
  { value: "Employee", label: "Employee" },
  { value: "Contractor", label: "Independent Contractor" },
  { value: "Ambiguous", label: "Ambiguous" },
  { value: "Unknown", label: "Unknown" },
];

const jurisdictionOptions = [
  { value: "ON", label: "Ontario (ON)" },
  { value: "BC", label: "British Columbia (BC)" },
  { value: "AB", label: "Alberta (AB)" },
  { value: "QC", label: "Quebec (QC)" },
  { value: "SK", label: "Saskatchewan (SK)" },
  { value: "MB", label: "Manitoba (MB)" },
  { value: "NS", label: "Nova Scotia (NS)" },
  { value: "NB", label: "New Brunswick (NB)" },
  { value: "NL", label: "Newfoundland & Labrador (NL)" },
  { value: "PE", label: "Prince Edward Island (PE)" },
  { value: "YT", label: "Yukon (YT)" },
  { value: "NT", label: "Northwest Territories (NT)" },
  { value: "NU", label: "Nunavut (NU)" },
  { value: "FED", label: "Federal (Canada)" },
];

function displayLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (l) => l.toUpperCase());
}

function FactorInput({
  name,
  control,
  error,
}: {
  name: string;
  control: any;
  error?: string;
}) {
  const detail = FACTOR_DETAILS[name as keyof typeof FACTOR_DETAILS] || {
    description: "Legal factor for worker classification analysis.",
    employeeIndicators: "",
    contractorIndicators: "",
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <label className="text-sm font-medium text-surface-700 dark:text-surface-300">
          {displayLabel(name)}
        </label>
        <Tooltip content={
          <div className="space-y-2">
            <p>{detail.description}</p>
            {detail.employeeIndicators && (
              <div>
                <p className="font-semibold text-green-300">Employee indicators:</p>
                <p>{detail.employeeIndicators}</p>
              </div>
            )}
            {detail.contractorIndicators && (
              <div>
                <p className="font-semibold text-blue-300">Contractor indicators:</p>
                <p>{detail.contractorIndicators}</p>
              </div>
            )}
          </div>
        }>
          <Info className="h-3.5 w-3.5 text-surface-400 cursor-help" />
        </Tooltip>
      </div>
      <Controller
        name={name as any}
        control={control}
        render={({ field }) => (
          <Select
            options={factorOptions}
            placeholder="Select..."
            value={field.value || ""}
            onChange={field.onChange}
            error={error}
          />
        )}
      />
    </div>
  );
}

export function ClassifyPage() {
  const [showResults, setShowResults] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);
  const classifyMutation = useClassification();

  const {
    control,
    handleSubmit,
    formState: { errors },
  } = useForm<ClassifyForm>({
    resolver: zodResolver(classifySchema) as any,
    defaultValues: {
      jurisdiction: "ON",
      ...Object.fromEntries(factorKeys.map((k) => [k, "Unknown"])),
    },
  });

  const onSubmit = (data: ClassifyForm) => {
    setShowResults(false);
    classifyMutation.mutate(data as any, {
      onSuccess: () => setShowResults(true),
    });
  };

  const result = classifyMutation.data;
  const risk = result ? riskColor(
    result.classification === "Employee" ? result.confidence : 1 - result.confidence
  ) : null;

  const chartData = result
    ? Object.entries(result.factor_analysis).map(([key, val]) => ({
        name: displayLabel(key).split(" ")[0],
        fullName: displayLabel(key),
        weight: val.weight * 100,
        value: val.value,
        color:
          val.value === "Employee"
            ? "#1e3a5f"
            : val.value === "Contractor"
            ? "#c0392b"
            : "#a1a1aa",
      }))
    : [];

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-2xl sm:text-3xl font-bold text-surface-900 dark:text-surface-100">
          Worker Classification
        </h1>
        <p className="text-surface-500 dark:text-surface-400 mt-1">
          Sagaz factor analysis with MCTS legal reasoning
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Form */}
        <div>
          <Card>
            <CardHeader>
              <CardTitle>Factor Assessment</CardTitle>
              <CardDescription>
                Rate each Sagaz factor for the worker under assessment
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                {/* Jurisdiction */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-surface-700 dark:text-surface-300">
                    Jurisdiction
                  </label>
                  <Controller
                    name="jurisdiction"
                    control={control}
                    render={({ field }) => (
                      <Select
                        options={jurisdictionOptions}
                        value={field.value || "ON"}
                        onChange={field.onChange}
                      />
                    )}
                  />
                </div>

                <div className="space-y-4">
                  {factorKeys.map((key) => (
                    <FactorInput
                      key={key}
                      name={key}
                      control={control}
                      error={(errors as any)[key]?.message}
                    />
                  ))}
                </div>

                {/* Facts text area */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-surface-700 dark:text-surface-300">
                    Additional Facts (Optional)
                  </label>
                  <textarea
                    placeholder="Describe additional facts about the working relationship..."
                    className="w-full rounded-lg border border-surface-300 dark:border-surface-600 bg-white dark:bg-surface-900 px-3 py-2 text-sm placeholder:text-surface-400 focus:outline-none focus:ring-2 focus:ring-primary-500 min-h-[80px] resize-y"
                    {...(control as any).register("facts")}
                  />
                </div>

                <Button
                  type="submit"
                  className="w-full"
                  size="lg"
                  loading={classifyMutation.isPending}
                >
                  {classifyMutation.isPending ? "Running MCTS Reasoning..." : "Run Classification"}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>

        {/* Results */}
        <div>
          {classifyMutation.isPending ? (
            <Card>
              <CardContent className="p-8">
                <div className="flex flex-col items-center justify-center py-12">
                  <div className="relative mb-6">
                    <TreePine className="h-12 w-12 text-primary-500 animate-pulse" />
                  </div>
                  <p className="text-lg font-medium text-surface-900 dark:text-surface-100 mb-2">
                    Running MCTS Legal Reasoning
                  </p>
                  <p className="text-sm text-surface-500 text-center max-w-sm">
                    Exploring classification hypotheses using Monte Carlo Tree Search
                    with RAG-retrieved precedents...
                  </p>
                  <div className="mt-6 w-full max-w-xs space-y-2">
                    <Skeleton className="h-3 w-full" />
                    <Skeleton className="h-3 w-3/4" />
                    <Skeleton className="h-3 w-1/2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : classifyMutation.isError ? (
            <Card>
              <CardContent className="p-8">
                <div className="flex flex-col items-center justify-center py-12">
                  <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-red-100 dark:bg-red-900/30">
                    <AlertTriangle className="h-6 w-6 text-red-600" />
                  </div>
                  <p className="text-lg font-medium text-surface-900 dark:text-surface-100 mb-2">
                    Classification Failed
                  </p>
                  <p className="text-sm text-surface-500 text-center mb-6">
                    {classifyMutation.error?.message || "An error occurred during classification."}
                  </p>
                  <Button onClick={() => classifyMutation.reset()}>Try Again</Button>
                </div>
              </CardContent>
            </Card>
          ) : showResults && result ? (
            <div className="space-y-4">
              {/* Result Header */}
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <p className="text-sm text-surface-500">Classification Result</p>
                      <h2 className="text-2xl font-bold mt-1">{result.classification}</h2>
                    </div>
                    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold ${
                      result.classification === "Employee"
                        ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"
                        : "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
                    }`}>
                      {result.classification === "Employee" ? (
                        <AlertTriangle className="h-4 w-4" />
                      ) : (
                        <CheckCircle2 className="h-4 w-4" />
                      )}
                      {(result.confidence * 100).toFixed(0)}% Confidence
                    </div>
                  </div>

                  {/* Risk Assessment */}
                  {risk && (
                    <div className={`p-4 rounded-lg ${
                      risk.color === "bg-red-500" ? "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800" :
                      risk.color === "bg-yellow-500" ? "bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800" :
                      "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800"
                    }`}>
                      <div className="flex items-center gap-2">
                        <div className={`h-2.5 w-2.5 rounded-full ${risk.color}`} />
                        <span className="font-medium text-sm">{risk.label}</span>
                      </div>
                      <p className="text-xs text-surface-500 mt-1">
                        {result.classification === "Employee"
                          ? "Worker is likely an employee. Misclassification could lead to significant liability."
                          : "Worker is likely an independent contractor. Continue monitoring the relationship."}
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Factor Breakdown Chart */}
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-surface-400" />
                    <div>
                      <CardTitle>Factor Breakdown</CardTitle>
                      <CardDescription>Weight of each factor in the decision</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} layout="vertical" margin={{ left: 80 }}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-surface-200 dark:stroke-surface-700" />
                        <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} className="text-surface-500" />
                        <YAxis
                          type="category"
                          dataKey="name"
                          tick={{ fontSize: 11 }}
                          className="text-surface-500"
                          width={70}
                        />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: "var(--card-bg, #fff)",
                            border: "1px solid var(--border-color, #e4e4e7)",
                            borderRadius: "8px",
                            fontSize: "12px",
                          }}
                          formatter={(value: number) => [`${value.toFixed(0)}%`, "Weight"]}
                          labelFormatter={(label) => chartData.find((d) => d.name === label)?.fullName || label}
                        />
                        <Bar dataKey="weight" radius={[0, 4, 4, 0]}>
                          {chartData.map((entry, index) => (
                            <Cell key={index} fill={entry.color} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Legal Reasoning */}
              <Card>
                <CardHeader>
                  <button
                    onClick={() => setShowReasoning(!showReasoning)}
                    className="flex items-center justify-between w-full text-left"
                  >
                    <div className="flex items-center gap-2">
                      <TreePine className="h-5 w-5 text-surface-400" />
                      <div>
                        <CardTitle>MCTS Legal Reasoning</CardTitle>
                        <CardDescription>
                          {result.tree_statistics.total_nodes} nodes explored, {result.tree_statistics.max_depth} max depth
                        </CardDescription>
                      </div>
                    </div>
                    {showReasoning ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
                  </button>
                </CardHeader>
                {showReasoning && (
                  <CardContent>
                    <div className="prose prose-sm dark:prose-invert max-w-none">
                      {result.reasoning_text.split("\n").map((line, i) => {
                        if (line.startsWith("# ")) return <h1 key={i} className="text-lg font-bold mt-4 mb-2">{line.slice(2)}</h1>;
                        if (line.startsWith("## ")) return <h2 key={i} className="text-base font-semibold mt-3 mb-1">{line.slice(3)}</h2>;
                        if (line.startsWith("**") && line.endsWith("**")) return <p key={i} className="font-semibold mt-2">{line.slice(2, -2)}</p>;
                        if (line.trim().startsWith("-")) return <li key={i} className="text-sm ml-4">{line.trim().slice(1)}</li>;
                        return <p key={i} className="text-sm leading-relaxed">{line}</p>;
                      })}
                    </div>
                    <div className="mt-4 p-3 rounded-lg bg-surface-50 dark:bg-surface-800 border border-surface-200 dark:border-surface-700">
                      <div className="flex items-center gap-2 text-xs text-surface-500">
                        <Clock className="h-3.5 w-3.5" />
                        Reasoning completed in {(result.duration_ms / 1000).toFixed(1)}s
                        <span className="mx-2">·</span>
                        <TreePine className="h-3.5 w-3.5" />
                        {result.tree_statistics.n_simulations} simulations
                      </div>
                    </div>
                  </CardContent>
                )}
              </Card>
            </div>
          ) : (
            <Card>
              <CardContent className="p-8">
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <Scale className="h-12 w-12 text-surface-300 mb-4" />
                  <p className="text-lg font-medium text-surface-700 dark:text-surface-300 mb-2">
                    Ready to Classify
                  </p>
                  <p className="text-sm text-surface-500 max-w-sm">
                    Complete the Sagaz factor assessment on the left and click
                    &quot;Run Classification&quot; to see the MCTS legal reasoning results here.
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
