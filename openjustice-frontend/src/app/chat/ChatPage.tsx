import { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  MessageSquare,
  Search,
  Plus,
  ChevronLeft,
  ChevronRight,
  Trash2,
  Scale,
  FileText,
  ExternalLink,
  AlertCircle,
  RefreshCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip } from "@/components/ui/tooltip";
import { useRAGQuery } from "@/lib/hooks/useQuery";
import { useChatStore } from "@/lib/stores/chatStore";
import { useAuthStore } from "@/lib/stores/authStore";
import { cn, formatRelativeTime, confidenceColor } from "@/lib/utils";
import type { Message } from "@/types";

function generateId(): string {
  return Math.random().toString(36).substring(2, 15);
}

const suggestedQuestions = [
  "What is the test for employee vs independent contractor in Canada?",
  "What are the notice requirements for termination in Ontario?",
  "How does the duty to accommodate work in Ontario?",
  "What constitutes constructive dismissal?",
  "How are damages calculated for wrongful dismissal?",
];

function TypingIndicator() {
  return (
    <div className="flex items-start gap-3 px-4 py-3">
      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary-100 dark:bg-primary-900">
        <Scale className="h-4 w-4 text-primary-600 dark:text-primary-400" />
      </div>
      <div className="flex items-center gap-1 py-2">
        <span className="typing-dot h-2 w-2 rounded-full bg-surface-400 inline-block" />
        <span className="typing-dot h-2 w-2 rounded-full bg-surface-400 inline-block" />
        <span className="typing-dot h-2 w-2 rounded-full bg-surface-400 inline-block" />
      </div>
    </div>
  );
}

function SourcePanel({ sources, onClose }: { sources: Message["sources"]; onClose: () => void }) {
  return (
    <div className="border-l border-surface-200 dark:border-surface-700 bg-surface-50 dark:bg-surface-900/50 w-full lg:w-96 overflow-y-auto">
      <div className="flex items-center justify-between p-4 border-b border-surface-200 dark:border-surface-700">
        <h3 className="text-sm font-semibold text-surface-900 dark:text-surface-100">
          Sources ({sources?.length || 0})
        </h3>
        <Button variant="ghost" size="sm" onClick={onClose} className="lg:hidden">
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
      <div className="p-4 space-y-3">
        {sources?.map((source, idx) => (
          <div key={idx} className="p-3 rounded-lg bg-white dark:bg-surface-800 border border-surface-200 dark:border-surface-700">
            <div className="flex items-start justify-between mb-2">
              <Badge variant="outline" className="text-xs">#{idx + 1}</Badge>
              <span className={cn("text-xs font-medium", confidenceColor(source.relevance_score))}>
                {(source.relevance_score * 100).toFixed(0)}% match
              </span>
            </div>
            <p className="text-sm font-medium text-surface-900 dark:text-surface-100 mb-1">
              {source.title}
            </p>
            <p className="text-xs text-primary-500 font-mono mb-2">{source.citation}</p>
            <p className="text-xs text-surface-500 dark:text-surface-400 leading-relaxed line-clamp-3">
              {source.content}
            </p>
            {source.jurisdiction && (
              <Badge variant="secondary" className="mt-2 text-xs">
                {source.jurisdiction}
              </Badge>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const [showSources, setShowSources] = useState(false);

  if (message.role === "user") {
    return (
      <div className="flex justify-end px-4 py-2">
        <div className="max-w-[85%] sm:max-w-[75%] bg-primary-500 text-white rounded-2xl rounded-br-sm px-4 py-3">
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="px-4 py-2">
      <div className="flex items-start gap-3 max-w-[95%]">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary-100 dark:bg-primary-900">
          <Scale className="h-4 w-4 text-primary-600 dark:text-primary-400" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-medium text-surface-900 dark:text-surface-100">OpenJustice AI</span>
            {message.confidence && (
              <Badge variant={message.confidence === "high" ? "success" : message.confidence === "medium" ? "warning" : "destructive"} className="text-[10px] px-1.5 py-0">
                {message.confidence}
              </Badge>
            )}
          </div>
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <div className="text-sm text-surface-700 dark:text-surface-300 leading-relaxed whitespace-pre-wrap">
              {message.content}
            </div>
          </div>

          {/* Sources */}
          {message.sources && message.sources.length > 0 && (
            <div className="mt-3">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowSources(!showSources)}
                className="text-xs text-primary-500 hover:text-primary-600 p-0 h-auto"
              >
                <FileText className="h-3.5 w-3.5 mr-1" />
                {message.sources.length} source{message.sources.length > 1 ? "s" : ""}
                <ChevronRight className={cn("h-3 w-3 ml-1 transition-transform", showSources && "rotate-90")} />
              </Button>
              {showSources && (
                <div className="mt-2 space-y-2">
                  {message.sources.map((source, idx) => (
                    <div key={idx} className="p-2 rounded-lg bg-surface-50 dark:bg-surface-800/50 border border-surface-200 dark:border-surface-700">
                      <div className="flex items-start justify-between">
                        <p className="text-xs font-medium text-surface-700 dark:text-surface-300">{source.title}</p>
                        <span className="text-[10px] text-surface-400">{source.citation}</span>
                      </div>
                      <p className="text-xs text-surface-500 mt-1 line-clamp-2">{source.content}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <p className="text-[10px] text-surface-400 mt-2">{formatRelativeTime(message.timestamp)}</p>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onSelectQuestion }: { onSelectQuestion: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-4 py-16">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary-100 dark:bg-primary-900/30 mb-6">
        <Scale className="h-8 w-8 text-primary-600 dark:text-primary-400" />
      </div>
      <h2 className="text-xl font-semibold text-surface-900 dark:text-surface-100 mb-2">
        How can I help you today?
      </h2>
      <p className="text-sm text-surface-500 dark:text-surface-400 text-center max-w-md mb-8">
        Ask any Canadian employment law question. I&apos;ll search through case law and provide
        answers with citations.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-2xl">
        {suggestedQuestions.map((q) => (
          <button
            key={q}
            onClick={() => onSelectQuestion(q)}
            className="text-left p-3 rounded-xl border border-surface-200 dark:border-surface-700 hover:border-primary-300 dark:hover:border-primary-600 hover:bg-surface-50 dark:hover:bg-surface-800/50 transition-all text-sm text-surface-600 dark:text-surface-400 leading-relaxed"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function ChatSidebar() {
  const {
    conversations,
    activeConversationId,
    isSidebarOpen,
    setSidebarOpen,
    setActiveConversation,
    deleteConversation,
  } = useChatStore();
  const [searchQuery, setSearchQuery] = useState("");

  const filtered = conversations.filter((c) =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <>
      {/* Mobile overlay */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={cn(
          "fixed lg:relative inset-y-0 left-0 z-50 w-72 lg:w-80 bg-white dark:bg-surface-950 border-r border-surface-200 dark:border-surface-700 flex flex-col transition-transform duration-300",
          isSidebarOpen ? "translate-x-0" : "-translate-x-full lg:hidden"
        )}
      >
        {/* New Chat Button */}
        <div className="p-4 border-b border-surface-200 dark:border-surface-700">
          <Button className="w-full justify-start" onClick={() => { setActiveConversation(""); setSidebarOpen(false); }}>
            <Plus className="h-4 w-4 mr-2" />
            New Chat
          </Button>
        </div>

        {/* Search */}
        <div className="p-4 border-b border-surface-200 dark:border-surface-700">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-surface-400" />
            <input
              type="text"
              placeholder="Search conversations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-2 text-sm rounded-lg border border-surface-200 dark:border-surface-700 bg-surface-50 dark:bg-surface-800 focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto p-2">
          {filtered.length === 0 ? (
            <div className="text-center py-8">
              <MessageSquare className="h-8 w-8 text-surface-300 mx-auto mb-2" />
              <p className="text-sm text-surface-400">No conversations yet</p>
            </div>
          ) : (
            <div className="space-y-1">
              {filtered.map((conv) => (
                <button
                  key={conv.id}
                  onClick={() => { setActiveConversation(conv.id); setSidebarOpen(false); }}
                  className={cn(
                    "w-full text-left p-3 rounded-lg transition-colors group",
                    activeConversationId === conv.id
                      ? "bg-primary-50 dark:bg-primary-950/50"
                      : "hover:bg-surface-50 dark:hover:bg-surface-800/50"
                  )}
                >
                  <div className="flex items-start justify-between">
                    <p className="text-sm font-medium text-surface-700 dark:text-surface-300 truncate flex-1">
                      {conv.title}
                    </p>
                    <button
                      onClick={(e) => { e.stopPropagation(); deleteConversation(conv.id); }}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded"
                    >
                      <Trash2 className="h-3.5 w-3.5 text-red-500" />
                    </button>
                  </div>
                  <p className="text-xs text-surface-400 mt-1">
                    {conv.messages.length} messages · {formatRelativeTime(conv.updatedAt)}
                  </p>
                </button>
              ))}
            </div>
          )}
        </div>
      </aside>
    </>
  );
}

export function ChatPage() {
  const [input, setInput] = useState("");
  const [showSources, setShowSources] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const queryMutation = useRAGQuery();
  const { conversations, activeConversationId, addConversation, addMessage, isSidebarOpen, toggleSidebar } = useChatStore();

  const activeConversation = conversations.find((c) => c.id === activeConversationId);
  const messages = activeConversation?.messages || [];

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, queryMutation.isPending, scrollToBottom]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || queryMutation.isPending) return;

    setInput("");

    // Create or reuse conversation
    let convId = activeConversationId;
    if (!convId) {
      convId = generateId();
      addConversation({
        id: convId,
        title: text.length > 50 ? text.substring(0, 50) + "..." : text,
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });
    }

    const userMessage: Message = {
      id: generateId(),
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    };
    addMessage(convId, userMessage);

    try {
      const result = await queryMutation.mutateAsync(text);
      const aiMessage: Message = {
        id: generateId(),
        role: "assistant",
        content: result.answer,
        sources: result.sources,
        confidence: result.confidence,
        timestamp: new Date().toISOString(),
      };
      addMessage(convId, aiMessage);
    } catch {
      const errorMessage: Message = {
        id: generateId(),
        role: "assistant",
        content: "I apologize, but I encountered an error processing your question. Please try again.",
        timestamp: new Date().toISOString(),
      };
      addMessage(convId, errorMessage);
    }
  }, [input, activeConversationId, queryMutation, addConversation, addMessage]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      <ChatSidebar />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-surface-200 dark:border-surface-700 bg-white dark:bg-surface-950">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={toggleSidebar} aria-label="Toggle sidebar">
              {isSidebarOpen ? <ChevronLeft className="h-5 w-5" /> : <ChevronRight className="h-5 w-5" />}
            </Button>
            <h1 className="text-sm font-semibold text-surface-900 dark:text-surface-100 hidden sm:block">
              {activeConversation ? activeConversation.title : "Legal Chat"}
            </h1>
          </div>
          {messages.some((m) => m.sources && m.sources.length > 0) && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSources(!showSources)}
              className="hidden lg:flex"
            >
              <FileText className="h-4 w-4 mr-2" />
              Sources
            </Button>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 && !queryMutation.isPending ? (
            <EmptyState onSelectQuestion={(q) => { setInput(q); }} />
          ) : (
            <div className="py-4 space-y-1">
              {messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}
              {queryMutation.isPending && <TypingIndicator />}
              {queryMutation.isError && (
                <div className="flex items-center justify-center gap-3 px-4 py-4">
                  <div className="flex items-center gap-2 p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
                    <AlertCircle className="h-4 w-4 text-red-500" />
                    <span className="text-sm text-red-600 dark:text-red-400">
                      Failed to get response
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => queryMutation.reset()}
                      className="text-red-500 hover:text-red-600"
                    >
                      <RefreshCw className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-surface-200 dark:border-surface-700 bg-white dark:bg-surface-950 p-4">
          <div className="max-w-4xl mx-auto flex items-end gap-3">
            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a legal question... (Enter to send, Shift+Enter for new line)"
                rows={1}
                className="w-full resize-none rounded-xl border border-surface-300 dark:border-surface-600 bg-white dark:bg-surface-900 px-4 py-3 text-sm placeholder:text-surface-400 focus:outline-none focus:ring-2 focus:ring-primary-500 min-h-[44px] max-h-32"
              />
            </div>
            <Button
              onClick={handleSend}
              disabled={!input.trim() || queryMutation.isPending}
              size="icon"
              className="h-[44px] w-[44px] shrink-0"
              aria-label="Send message"
            >
              <Send className="h-5 w-5" />
            </Button>
          </div>
          <p className="text-[10px] text-surface-400 text-center mt-2">
            Responses are generated by AI and should be verified by a legal professional.
          </p>
        </div>
      </div>

      {/* Source Panel (Desktop) */}
      {showSources && messages.some((m) => m.sources) && (
        <SourcePanel
          sources={messages.find((m) => m.sources)?.sources}
          onClose={() => setShowSources(false)}
        />
      )}
    </div>
  );
}
