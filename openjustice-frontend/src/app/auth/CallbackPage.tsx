import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Loader2, AlertCircle, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuthStore } from "@/lib/stores/authStore";

export function AuthCallbackPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const store = useAuthStore();
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    const handleCallback = async () => {
      const code = searchParams.get("code");
      const error = searchParams.get("error");
      const errorDescription = searchParams.get("error_description");

      if (error) {
        setStatus("error");
        setErrorMessage(errorDescription || error);
        return;
      }

      if (!code) {
        setStatus("error");
        setErrorMessage("No authorization code received");
        return;
      }

      try {
        // Exchange authorization code for tokens
        const apiBase = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000/api";
        const response = await fetch(`${apiBase}/auth/google`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id_token: code }),
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "Authentication failed");
        }

        const data = await response.json();
        
        // Store tokens and user info
        localStorage.setItem("access_token", data.access_token);
        localStorage.setItem("refresh_token", data.refresh_token);
        
        store.setUser({
          id: data.user_id,
          name: data.name,
          email: data.email,
          tier: data.tier as "free" | "pro" | "enterprise",
          createdAt: new Date().toISOString(),
        }, data.access_token);
        
        setStatus("success");
        
        // Redirect to dashboard after short delay
        setTimeout(() => navigate("/dashboard"), 1500);
      } catch (err) {
        setStatus("error");
        setErrorMessage(err instanceof Error ? err.message : "Authentication failed");
      }
    };

    handleCallback();
  }, [searchParams, navigate]);

  if (status === "loading") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-surface-50 dark:bg-surface-950">
        <Card className="w-full max-w-md">
          <CardContent className="p-8 text-center">
            <Loader2 className="h-10 w-10 animate-spin text-primary-500 mx-auto mb-4" />
            <h2 className="text-lg font-semibold text-surface-900 dark:text-surface-100">
              Completing sign in...
            </h2>
            <p className="text-sm text-surface-500 dark:text-surface-400 mt-2">
              Redirecting from Google...
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-surface-50 dark:bg-surface-950 px-4">
        <Card className="w-full max-w-md">
          <CardContent className="p-8 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-red-100 dark:bg-red-900/30 mx-auto mb-4">
              <AlertCircle className="h-6 w-6 text-red-500" />
            </div>
            <h2 className="text-lg font-semibold text-surface-900 dark:text-surface-100 mb-2">
              Sign in failed
            </h2>
            <p className="text-sm text-surface-500 dark:text-surface-400 mb-6">
              {errorMessage || "An error occurred during authentication"}
            </p>
            <Button onClick={() => navigate("/login")} variant="outline">
              Try again
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-surface-50 dark:bg-surface-950">
      <Card className="w-full max-w-md">
        <CardContent className="p-8 text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/30 mx-auto mb-4">
            <CheckCircle className="h-6 w-6 text-green-500" />
          </div>
          <h2 className="text-lg font-semibold text-surface-900 dark:text-surface-100 mb-2">
            Welcome!
          </h2>
          <p className="text-sm text-surface-500 dark:text-surface-400 mb-6">
            Sign in successful. Redirecting to dashboard...
          </p>
        </CardContent>
      </Card>
      </div>
    );
}
