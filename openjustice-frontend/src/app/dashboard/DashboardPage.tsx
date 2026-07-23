import { Link } from "react-router-dom";
import {
  MessageSquare,
  FileSearch,
  Users,
  ArrowRight,
  Activity,
  BarChart3,
  Clock,
  Sparkles,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuthStore } from "@/lib/stores/authStore";
import { useUsageStats, useUsageChartData, useRecentActivity } from "@/lib/hooks/useQuery";
import { formatRelativeTime } from "@/lib/utils";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";

export function DashboardPage() {
  const user = useAuthStore((s) => s.user);
  const { data: stats, isLoading: statsLoading } = useUsageStats();
  const { data: chartData, isLoading: chartLoading } = useUsageChartData(14);
  const { data: activities, isLoading: activitiesLoading } = useRecentActivity();

  const usagePercent = stats ? Math.round((stats.queriesThisMonth / stats.queriesLimit) * 100) : 0;

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      {/* Welcome Card */}
      <Card className="mb-8 bg-gradient-to-r from-primary-500 to-primary-700 border-0">
        <CardContent className="p-6 sm:p-8">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
<div>
                <h1 className="text-2xl sm:text-3xl font-bold text-white">
                  Welcome back, {user?.name?.split(" ")[0] || "there"}
                </h1>
                <div className="text-primary-100 mt-1">
                  You're on the <Badge variant="secondary" className="bg-white/20 text-white border-0 capitalize">{stats?.tier || "free"}</Badge> plan
                </div>
              </div>
            <Link to="/chat">
              <Button className="bg-white text-primary-600 hover:bg-primary-50 shadow-lg">
                <Sparkles className="h-4 w-4 mr-2" />
                New Query
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-2">
              <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/30">
                <MessageSquare className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
            </div>
            {statsLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <p className="text-2xl font-bold text-surface-900 dark:text-surface-100">
                {stats?.queriesThisMonth || 0}
                <span className="text-sm font-normal text-surface-500 ml-1">
                  / {stats?.queriesLimit || 0}
                </span>
              </p>
            )}
            <p className="text-sm text-surface-500 mt-1">Queries this month</p>
            {!statsLoading && (
              <div className="mt-3 h-2 w-full bg-surface-100 dark:bg-surface-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary-500 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(usagePercent, 100)}%` }}
                />
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-2">
              <div className="p-2 rounded-lg bg-purple-100 dark:bg-purple-900/30">
                <Users className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              </div>
            </div>
            {statsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="text-2xl font-bold text-surface-900 dark:text-surface-100">
                {stats?.classificationsRun || 0}
              </p>
            )}
            <p className="text-sm text-surface-500 mt-1">Classifications run</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-2">
              <div className="p-2 rounded-lg bg-emerald-100 dark:bg-emerald-900/30">
                <FileSearch className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
              </div>
            </div>
            {statsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="text-2xl font-bold text-surface-900 dark:text-surface-100">
                {stats?.documentsAnalyzed || 0}
              </p>
            )}
            <p className="text-sm text-surface-500 mt-1">Documents analyzed</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-2">
              <div className="p-2 rounded-lg bg-amber-100 dark:bg-amber-900/30">
                <Activity className="h-5 w-5 text-amber-600 dark:text-amber-400" />
              </div>
            </div>
            {statsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                {usagePercent < 80 ? "Good" : usagePercent < 95 ? "High" : "Critical"}
              </p>
            )}
            <p className="text-sm text-surface-500 mt-1">Usage status</p>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        <Link to="/chat">
          <Card className="hover:shadow-md transition-all duration-200 cursor-pointer border-primary-200 dark:border-primary-800 hover:border-primary-500">
            <CardContent className="p-6 flex items-center gap-4">
              <div className="p-3 rounded-lg bg-primary-100 dark:bg-primary-900/30">
                <MessageSquare className="h-6 w-6 text-primary-600 dark:text-primary-400" />
              </div>
              <div>
                <p className="font-semibold text-surface-900 dark:text-surface-100">New Query</p>
                <p className="text-sm text-surface-500">Ask a legal question</p>
              </div>
            </CardContent>
          </Card>
        </Link>
        <Link to="/classify">
          <Card className="hover:shadow-md transition-all duration-200 cursor-pointer border-purple-200 dark:border-purple-800 hover:border-purple-500">
            <CardContent className="p-6 flex items-center gap-4">
              <div className="p-3 rounded-lg bg-purple-100 dark:bg-purple-900/30">
                <Users className="h-6 w-6 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <p className="font-semibold text-surface-900 dark:text-surface-100">Classify Worker</p>
                <p className="text-sm text-surface-500">Run Sagaz analysis</p>
              </div>
            </CardContent>
          </Card>
        </Link>
        <Link to="/analyze">
          <Card className="hover:shadow-md transition-all duration-200 cursor-pointer border-emerald-200 dark:border-emerald-800 hover:border-emerald-500">
            <CardContent className="p-6 flex items-center gap-4">
              <div className="p-3 rounded-lg bg-emerald-100 dark:bg-emerald-900/30">
                <FileSearch className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
              </div>
              <div>
                <p className="font-semibold text-surface-900 dark:text-surface-100">Analyze Document</p>
                <p className="text-sm text-surface-500">Upload & analyze</p>
              </div>
            </CardContent>
          </Card>
        </Link>
      </div>

      {/* Usage Chart + Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Usage Chart */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Usage Overview</CardTitle>
                <CardDescription>Daily queries and classifications (14 days)</CardDescription>
              </div>
              <BarChart3 className="h-5 w-5 text-surface-400" />
            </div>
          </CardHeader>
          <CardContent>
            {chartLoading ? (
              <Skeleton className="h-[300px] w-full" />
            ) : (
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient id="queriesGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#1e3a5f" stopOpacity={0.2} />
                        <stop offset="95%" stopColor="#1e3a5f" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="classGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#c0392b" stopOpacity={0.2} />
                        <stop offset="95%" stopColor="#c0392b" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-surface-200 dark:stroke-surface-700" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 11 }}
                      tickFormatter={(d) => new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                      className="text-surface-500"
                    />
                    <YAxis className="text-surface-500" tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "var(--card-bg, #fff)",
                        border: "1px solid var(--border-color, #e4e4e7)",
                        borderRadius: "8px",
                        fontSize: "12px",
                      }}
                      labelFormatter={(d) => new Date(d).toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" })}
                    />
                    <Area type="monotone" dataKey="queries" stroke="#1e3a5f" fill="url(#queriesGrad)" strokeWidth={2} name="Queries" />
                    <Area type="monotone" dataKey="classifications" stroke="#c0392b" fill="url(#classGrad)" strokeWidth={2} name="Classifications" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Recent Activity</CardTitle>
                <CardDescription>Your latest actions</CardDescription>
              </div>
              <Clock className="h-5 w-5 text-surface-400" />
            </div>
          </CardHeader>
          <CardContent>
            {activitiesLoading ? (
              <div className="space-y-3">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <Skeleton className="h-8 w-8 rounded-full" />
                    <div className="flex-1">
                      <Skeleton className="h-4 w-3/4 mb-1" />
                      <Skeleton className="h-3 w-1/4" />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="space-y-1">
                {activities?.map((activity) => (
                  <div
                    key={activity.id}
                    className="flex items-start gap-3 p-2 rounded-lg hover:bg-surface-50 dark:hover:bg-surface-800/50 transition-colors"
                  >
                    <div className={`p-1.5 rounded-full ${
                      activity.type === "query"
                        ? "bg-blue-100 dark:bg-blue-900/30 text-blue-600"
                        : activity.type === "classification"
                        ? "bg-purple-100 dark:bg-purple-900/30 text-purple-600"
                        : "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600"
                    }`}>
                      {activity.type === "query" ? (
                        <MessageSquare className="h-3.5 w-3.5" />
                      ) : activity.type === "classification" ? (
                        <Users className="h-3.5 w-3.5" />
                      ) : (
                        <FileSearch className="h-3.5 w-3.5" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-surface-700 dark:text-surface-300 truncate">{activity.description}</p>
                      <p className="text-xs text-surface-400">{formatRelativeTime(activity.timestamp)}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
