import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  User,
  Key,
  CreditCard,
  Bell,
  Save,
  Plus,
  Trash2,
  Copy,
  CheckCheck,
  ArrowUpCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuthStore } from "@/lib/stores/authStore";
import { useApiKeys, useCreateApiKey, useRevokeApiKey, useUsageStats, useUpgradeSubscription } from "@/lib/hooks/useQuery";
import { cn, formatDate } from "@/lib/utils";
import { toast } from "sonner";

const tabs = [
  { id: "profile", label: "Profile", icon: User },
  { id: "api-keys", label: "API Keys", icon: Key },
  { id: "subscription", label: "Subscription", icon: CreditCard },
  { id: "notifications", label: "Notifications", icon: Bell },
] as const;

type TabId = (typeof tabs)[number]["id"];

const profileSchema = z.object({
  name: z.string().min(2, "Name must be at least 2 characters"),
  email: z.string().email("Please enter a valid email"),
});

type ProfileForm = z.infer<typeof profileSchema>;

function ProfileTab() {
  const { user, updateUser } = useAuthStore();
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<ProfileForm>({
    resolver: zodResolver(profileSchema),
    defaultValues: {
      name: user?.name || "",
      email: user?.email || "",
    },
  });

  const onSubmit = (data: ProfileForm) => {
    updateUser(data);
    toast.success("Profile updated successfully");
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Profile</CardTitle>
        <CardDescription>Manage your personal information</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          {/* Avatar */}
          <div className="flex items-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-300 text-xl font-bold">
              {user?.name?.charAt(0).toUpperCase() || "U"}
            </div>
            <div>
              <p className="text-sm font-medium text-surface-900 dark:text-surface-100">{user?.name}</p>
              <p className="text-xs text-surface-500 capitalize">{user?.tier} plan</p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-surface-700 dark:text-surface-300 mb-1">
                Full Name
              </label>
              <Input {...register("name")} error={errors.name?.message} />
            </div>
            <div>
              <label className="block text-sm font-medium text-surface-700 dark:text-surface-300 mb-1">
                Email
              </label>
              <Input {...register("email")} type="email" error={errors.email?.message} />
            </div>
          </div>

          <Button type="submit">
            <Save className="h-4 w-4 mr-2" />
            Save Changes
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

function ApiKeysTab() {
  const { data: apiKeys, isLoading } = useApiKeys();
  const createApiKey = useCreateApiKey();
  const revokeApiKey = useRevokeApiKey();
  const [newKeyName, setNewKeyName] = useState("");
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const handleCreate = () => {
    if (!newKeyName.trim()) return;
    createApiKey.mutate(newKeyName.trim(), {
      onSuccess: (result) => {
        setCopiedKey(result.key);
        toast.success("API key created! Make sure to copy it now.");
        setNewKeyName("");
      },
    });
  };

  const handleRevoke = (id: string) => {
    revokeApiKey.mutate(id, {
      onSuccess: () => {
        toast.success("API key revoked");
      },
    });
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>API Keys</CardTitle>
          <CardDescription>Manage API keys for programmatic access</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-end gap-3">
            <div className="flex-1">
              <label className="block text-sm font-medium text-surface-700 dark:text-surface-300 mb-1">
                New API Key Name
              </label>
              <Input
                placeholder="e.g., Production, Development"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
              />
            </div>
            <Button onClick={handleCreate} loading={createApiKey.isPending}>
              <Plus className="h-4 w-4 mr-2" />
              Create Key
            </Button>
          </div>

          {/* Show newly created key */}
          {copiedKey && (
            <div className="p-3 rounded-lg bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800">
              <p className="text-xs font-medium text-yellow-700 dark:text-yellow-300 mb-1">
                Copy this key now. You won&apos;t be able to see it again.
              </p>
              <div className="flex items-center gap-2">
                <code className="flex-1 text-sm font-mono bg-white dark:bg-surface-900 px-2 py-1 rounded border border-yellow-300">
                  {copiedKey}
                </code>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    navigator.clipboard.writeText(copiedKey);
                    toast.success("Copied to clipboard");
                  }}
                >
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Existing Keys</CardTitle>
          <CardDescription>Your active API keys</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[...Array(2)].map((_, i) => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : apiKeys && apiKeys.length > 0 ? (
            <div className="space-y-3">
              {apiKeys.map((apiKey) => (
                <div
                  key={apiKey.id}
                  className="flex items-center justify-between p-4 rounded-lg bg-surface-50 dark:bg-surface-800 border border-surface-200 dark:border-surface-700"
                >
                  <div>
                    <p className="text-sm font-medium text-surface-900 dark:text-surface-100">{apiKey.name}</p>
                    <p className="text-xs text-surface-500 font-mono mt-0.5">{apiKey.key}</p>
                    <p className="text-xs text-surface-400 mt-1">
                      Created {formatDate(apiKey.createdAt)}
                      {apiKey.lastUsed && ` · Last used ${formatDate(apiKey.lastUsed)}`}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRevoke(apiKey.id)}
                    className="text-red-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/30"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-surface-400 text-center py-6">No API keys created yet</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function SubscriptionTab() {
  const { data: stats, isLoading } = useUsageStats();
  const upgradeMutation = useUpgradeSubscription();
  const user = useAuthStore((s) => s.user);

  const plans = [
    { name: "Free", price: "$0", queries: 20, documents: 5, current: user?.tier === "free" },
    { name: "Pro", price: "$29/mo", queries: 200, documents: 50, current: user?.tier === "pro" },
    { name: "Enterprise", price: "Custom", queries: "Unlimited", documents: "Unlimited", current: user?.tier === "enterprise" },
  ] as const;

  const handleUpgrade = (tier: 'pro' | 'enterprise') => {
    upgradeMutation.mutate(tier, {
      onSuccess: () => {
        toast.success(`Upgraded to ${tier} plan!`);
      },
    });
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Current Plan</CardTitle>
          <CardDescription>You are on the {user?.tier || "free"} plan</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              <Skeleton className="h-4 w-48" />
              <Skeleton className="h-8 w-32" />
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <p className="text-3xl font-bold text-surface-900 dark:text-surface-100 capitalize">
                  {stats?.tier || "free"}
                </p>
                {stats && (
                  <div className="mt-4 space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-surface-500">Queries used this month</span>
                        <span className="font-medium">{stats.queriesThisMonth} / {stats.queriesLimit}</span>
                      </div>
                      <div className="h-2 w-full bg-surface-100 dark:bg-surface-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary-500 rounded-full transition-all"
                          style={{ width: `${Math.min((stats.queriesThisMonth / stats.queriesLimit) * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-surface-500">Documents</span>
                        <p className="font-medium">{stats.documentsAnalyzed}</p>
                      </div>
                      <div>
                        <span className="text-surface-500">Classifications</span>
                        <p className="font-medium">{stats.classificationsRun}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Available Plans</CardTitle>
          <CardDescription>Choose the plan that fits your needs</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {plans.map((plan) => (
              <div
                key={plan.name}
                className={cn(
                  "p-4 rounded-xl border-2 transition-all",
                  plan.current
                    ? "border-primary-500 bg-primary-50 dark:bg-primary-950/30"
                    : "border-surface-200 dark:border-surface-700"
                )}
              >
                <p className="text-sm font-semibold text-surface-900 dark:text-surface-100">{plan.name}</p>
                <p className="text-2xl font-bold mt-1">{plan.price}</p>
                <ul className="mt-3 space-y-1 text-xs text-surface-500">
                  <li>{plan.queries} queries</li>
                  <li>{plan.documents} documents</li>
                </ul>
                {plan.current ? (
                  <Badge variant="default" className="mt-3">Current Plan</Badge>
                ) : (
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-3 w-full"
                    onClick={() => handleUpgrade(plan.name.toLowerCase() as 'pro' | 'enterprise')}
                  >
                    {plan.name === "Enterprise" ? "Contact Sales" : "Upgrade"}
                    <ArrowUpCircle className="h-3.5 w-3.5 ml-1" />
                  </Button>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function NotificationsTab() {
  const [preferences, setPreferences] = useState({
    emailDigest: true,
    usageAlerts: true,
    productUpdates: false,
  });

  const handleToggle = (key: keyof typeof preferences) => {
    setPreferences((prev) => ({ ...prev, [key]: !prev[key] }));
    toast.success("Preference updated");
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Notification Preferences</CardTitle>
        <CardDescription>Manage how you receive notifications</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {[
          { key: "emailDigest" as const, label: "Weekly Email Digest", description: "Receive a weekly summary of your usage and activity" },
          { key: "usageAlerts" as const, label: "Usage Alerts", description: "Get notified when you're approaching your query limit" },
          { key: "productUpdates" as const, label: "Product Updates", description: "Receive updates about new features and improvements" },
        ].map((item) => (
          <div
            key={item.key}
            className="flex items-center justify-between p-4 rounded-lg bg-surface-50 dark:bg-surface-800/50 border border-surface-200 dark:border-surface-700"
          >
            <div>
              <p className="text-sm font-medium text-surface-900 dark:text-surface-100">{item.label}</p>
              <p className="text-xs text-surface-500 mt-0.5">{item.description}</p>
            </div>
            <button
              onClick={() => handleToggle(item.key)}
              className={cn(
                "relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500",
                preferences[item.key] ? "bg-primary-500" : "bg-surface-300 dark:bg-surface-600"
              )}
              role="switch"
              aria-checked={preferences[item.key]}
              aria-label={item.label}
            >
              <span
                className={cn(
                  "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                  preferences[item.key] ? "translate-x-[18px]" : "translate-x-[2px]"
                )}
              />
            </button>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function SettingsPage() {
  const [activeTab, setActiveTab] = useState<TabId>("profile");

  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-2xl sm:text-3xl font-bold text-surface-900 dark:text-surface-100">Settings</h1>
        <p className="text-surface-500 dark:text-surface-400 mt-1">Manage your account and preferences</p>
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap gap-2 mb-8">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all",
              activeTab === tab.id
                ? "bg-primary-500 text-white shadow-sm"
                : "text-surface-600 dark:text-surface-400 hover:bg-surface-100 dark:hover:bg-surface-800"
            )}
          >
            <tab.icon className="h-4 w-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "profile" && <ProfileTab />}
      {activeTab === "api-keys" && <ApiKeysTab />}
      {activeTab === "subscription" && <SubscriptionTab />}
      {activeTab === "notifications" && <NotificationsTab />}
    </div>
  );
}
