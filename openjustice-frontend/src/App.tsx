import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { AppLayout, AuthLayout } from "@/components/layout/AppLayout";
import { AuthGuard } from "@/components/layout/AuthGuard";
import { ErrorBoundary } from "@/components/layout/ErrorBoundary";
import { ToastProvider } from "@/components/ui/toast";
import { LandingPage } from "@/app/landing/LandingPage";
import { LoginPage } from "@/app/auth/LoginPage";
import { AuthCallbackPage } from "@/app/auth/CallbackPage";
import { SignupPage } from "@/app/auth/SignupPage";
import { DashboardPage } from "@/app/dashboard/DashboardPage";
import { ChatPage } from "@/app/chat/ChatPage";
import { ClassifyPage } from "@/app/classify/ClassifyPage";
import { AnalyzePage } from "@/app/analyze/AnalyzePage";
import { SettingsPage } from "@/app/settings/SettingsPage";
import SearchPage from "@/app/search/SearchPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ToastProvider />
        <Routes>
          {/* Public routes */}
          <Route element={<AppLayout />}>
            <Route path="/" element={<LandingPage />} />
          </Route>

          {/* Auth routes */}
          <Route element={<AuthLayout />}>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
            <Route path="/auth/callback" element={<AuthCallbackPage />} />
          </Route>

          {/* Protected routes */}
          <Route
            element={
              <AuthGuard>
                <AppLayout />
              </AuthGuard>
            }
          >
            <Route
              path="/dashboard"
              element={
                <ErrorBoundary>
                  <DashboardPage />
                </ErrorBoundary>
              }
            />
            <Route
              path="/chat"
              element={
                <ErrorBoundary>
                  <ChatPage />
                </ErrorBoundary>
              }
            />
            <Route
              path="/classify"
              element={
                <ErrorBoundary>
                  <ClassifyPage />
                </ErrorBoundary>
              }
            />
            <Route
              path="/analyze"
              element={
                <ErrorBoundary>
                  <AnalyzePage />
                </ErrorBoundary>
              }
            />
            <Route
              path="/settings"
              element={
                <ErrorBoundary>
                  <SettingsPage />
                </ErrorBoundary>
              }
            />
            <Route
              path="/search"
              element={
                <ErrorBoundary>
                  <SearchPage />
                </ErrorBoundary>
              }
            />
          </Route>

          {/* 404 */}
          <Route
            path="*"
            element={
              <div className="flex min-h-[60vh] items-center justify-center">
                <div className="text-center">
                  <h1 className="text-4xl font-bold text-surface-900 dark:text-surface-100 mb-2">404</h1>
                  <p className="text-surface-500">Page not found</p>
                </div>
              </div>
            }
          />
        </Routes>
      </BrowserRouter>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
