import { Link } from "react-router-dom";
import { ArrowRight, Scale, FileSearch, Users, BookOpen, Shield, Zap, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuthStore } from "@/lib/stores/authStore";

const features = [
  {
    icon: BookOpen,
    title: "RAG-Powered Legal Q&A",
    description: "Ask complex legal questions and get answers grounded in case law with verified citations from Canadian courts.",
    color: "text-blue-600 bg-blue-100 dark:bg-blue-900/30",
  },
  {
    icon: Users,
    title: "Worker Classification",
    description: "AI-powered worker classification using the Sagaz framework with MCTS legal reasoning across 150+ jurisdictions.",
    color: "text-purple-600 bg-purple-100 dark:bg-purple-900/30",
  },
  {
    icon: FileSearch,
    title: "Document Analysis",
    description: "Upload employment agreements and contracts for AI analysis with entity extraction and classification risk assessment.",
    color: "text-emerald-600 bg-emerald-100 dark:bg-emerald-900/30",
  },
  {
    icon: Shield,
    title: "Hallucination Verification",
    description: "Every answer is verified against sources using NLI-based fact-checking to ensure accuracy and reliability.",
    color: "text-amber-600 bg-amber-100 dark:bg-amber-900/30",
  },
  {
    icon: BarChart3,
    title: "Usage Analytics",
    description: "Track your research patterns, classification history, and document analysis with detailed dashboards.",
    color: "text-rose-600 bg-rose-100 dark:bg-rose-900/30",
  },
  {
    icon: Zap,
    title: "Multi-Hop Reasoning",
    description: "Complex legal questions are broken down into sub-questions with iterative retrieval for comprehensive answers.",
    color: "text-cyan-600 bg-cyan-100 dark:bg-cyan-900/30",
  },
];

const pricingTiers = [
  {
    name: "Free",
    price: "$0",
    period: "/month",
    description: "For legal professionals exploring AI-powered research",
    queries: "20 queries/month",
    documents: "5 documents",
    features: ["Basic RAG Q&A", "Standard citations", "Email support"],
    cta: "Get Started",
    href: "/signup",
    popular: false,
  },
  {
    name: "Pro",
    price: "$29",
    period: "/month",
    description: "For practitioners who need deeper analysis",
    queries: "200 queries/month",
    documents: "50 documents",
    features: ["Everything in Free", "MCTS classification", "Multi-hop reasoning", "Document analysis", "Priority support"],
    cta: "Start Free Trial",
    href: "/signup",
    popular: true,
  },
  {
    name: "Enterprise",
    price: "Custom",
    period: "",
    description: "For law firms and institutions with advanced needs",
    queries: "Unlimited queries",
    documents: "Unlimited documents",
    features: ["Everything in Pro", "Dedicated model instance", "SSO/SAML", "Custom jurisdictions", "SLA guarantee", "API access"],
    cta: "Contact Sales",
    href: "/signup",
    popular: false,
  },
];

export function LandingPage() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);

  return (
    <div>
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        {/* Background gradient */}
        <div className="absolute inset-0 bg-gradient-to-b from-primary-50/50 to-white dark:from-primary-950/30 dark:to-surface-950" />
        <div className="absolute top-0 right-0 -mr-48 h-96 w-96 rounded-full bg-primary-100/30 dark:bg-primary-900/10 blur-3xl" />
        <div className="absolute bottom-0 left-0 -ml-48 h-96 w-96 rounded-full bg-accent-100/20 dark:bg-accent-900/10 blur-3xl" />

        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-24 sm:py-32 lg:py-40">
          <div className="text-center max-w-4xl mx-auto">
            <div className="inline-flex items-center gap-2 rounded-full border border-primary-200 dark:border-primary-800 bg-primary-50 dark:bg-primary-950/50 px-4 py-1.5 mb-6">
              <Scale className="h-4 w-4 text-primary-500" />
              <span className="text-sm font-medium text-primary-600 dark:text-primary-400">
                Built with the Conflict Analytics Lab at Queen&apos;s University
              </span>
            </div>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-primary-500 dark:text-primary-300 tracking-tight text-balance">
              AI-Powered Legal Research,{" "}
              <span className="text-primary-700 dark:text-primary-200">Reimagined</span>
            </h1>
            <p className="mt-6 text-lg sm:text-xl text-surface-500 dark:text-surface-400 max-w-2xl mx-auto text-balance leading-relaxed">
              OpenJustice.ai brings the power of advanced AI to legal research, worker classification, 
              and document analysis. Grounded in Canadian case law. Built for legal professionals.
            </p>
            <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
              {isAuthenticated ? (
                <Link to="/dashboard">
                  <Button size="xl" className="text-base">
                    Go to Dashboard <ArrowRight className="h-5 w-5" />
                  </Button>
                </Link>
              ) : (
                <>
                  <Link to="/signup">
                    <Button size="xl" className="text-base">
                      Start Free <ArrowRight className="h-5 w-5" />
                    </Button>
                  </Link>
                  <Link to="/login">
                    <Button variant="outline" size="xl" className="text-base">
                      Sign In
                    </Button>
                  </Link>
                </>
              )}
            </div>
            <p className="mt-4 text-sm text-surface-400">No credit card required. Free tier includes 20 queries.</p>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="border-t border-surface-200 dark:border-surface-700 bg-surface-50/50 dark:bg-surface-900/50">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-20 sm:py-28">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-primary-500 dark:text-primary-300">
              Everything you need for legal research
            </h2>
            <p className="mt-4 text-lg text-surface-500 dark:text-surface-400 max-w-2xl mx-auto">
              Powered by advanced RAG pipelines, MCTS reasoning, and NLI verification — built on 700+ Canadian employment law cases.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature) => (
              <Card key={feature.title} className="border-0 shadow-sm hover:shadow-md transition-shadow duration-300">
                <CardContent className="p-6">
                  <div className={`inline-flex p-3 rounded-lg mb-4 ${feature.color}`}>
                    <feature.icon className="h-6 w-6" />
                  </div>
                  <h3 className="text-lg font-semibold text-surface-900 dark:text-surface-100 mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-surface-500 dark:text-surface-400 leading-relaxed">
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-20 sm:py-28">
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-primary-500 dark:text-primary-300">
            Simple, transparent pricing
          </h2>
          <p className="mt-4 text-lg text-surface-500 dark:text-surface-400 max-w-2xl mx-auto">
            Choose the plan that fits your practice. All plans include our core AI-powered legal research capabilities.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          {pricingTiers.map((tier) => (
            <Card
              key={tier.name}
              className={`relative border-2 ${
                tier.popular
                  ? "border-primary-500 shadow-xl shadow-primary-500/10 scale-105"
                  : "border-surface-200 dark:border-surface-700 shadow-sm"
              }`}
            >
              {tier.popular && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                  <span className="bg-primary-500 text-white text-xs font-semibold px-3 py-1 rounded-full">
                    Most Popular
                  </span>
                </div>
              )}
              <CardHeader>
                <CardTitle className="text-xl">{tier.name}</CardTitle>
                <div className="mt-2">
                  <span className="text-4xl font-bold text-surface-900 dark:text-surface-100">{tier.price}</span>
                  <span className="text-surface-500 ml-1">{tier.period}</span>
                </div>
                <CardDescription className="mt-2">{tier.description}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-1">
                  <p className="text-sm font-medium text-surface-700 dark:text-surface-300">{tier.queries}</p>
                  <p className="text-sm text-surface-500">{tier.documents}</p>
                </div>
                <ul className="space-y-3">
                  {tier.features.map((f) => (
                    <li key={f} className="flex items-start gap-2 text-sm text-surface-600 dark:text-surface-400">
                      <svg className="h-4 w-4 text-green-500 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      {f}
                    </li>
                  ))}
                </ul>
                <Link to={tier.href} className="block">
                  <Button
                    variant={tier.popular ? "default" : "outline"}
                    className="w-full"
                    size="lg"
                  >
                    {tier.cta}
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-primary-500 dark:bg-primary-600">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 py-16 sm:py-20 text-center">
          <h2 className="text-3xl sm:text-4xl font-bold text-white">
            Ready to transform your legal research?
          </h2>
          <p className="mt-4 text-lg text-primary-100 max-w-2xl mx-auto">
            Join leading Canadian legal professionals using OpenJustice.ai for AI-powered research and analysis.
          </p>
          <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link to="/signup">
              <Button size="xl" className="bg-white text-primary-500 hover:bg-primary-50 text-base">
                Get Started Free
              </Button>
            </Link>
            <Link to="/login">
              <Button variant="outline" size="xl" className="border-white/30 text-white hover:bg-white/10 text-base">
                Sign In
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
