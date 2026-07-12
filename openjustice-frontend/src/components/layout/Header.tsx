import { Link, useNavigate } from "react-router-dom";
import { Scale, LogOut, Settings, User, Menu, X } from "lucide-react";
import { useAuthStore } from "@/lib/stores/authStore";
import { useLogout } from "@/lib/hooks/useAuth";
import { ThemeToggle } from "./ThemeToggle";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useState } from "react";

export function Header() {
  const { isAuthenticated, user } = useAuthStore();
  const logout = useLogout();
  const navigate = useNavigate();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-surface-200 dark:border-surface-700 bg-white/80 dark:bg-surface-950/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary-500">
            <Scale className="h-5 w-5 text-white" />
          </div>
          <span className="text-lg font-bold text-primary-500 dark:text-primary-300">
            OpenJustice<span className="text-surface-500 dark:text-surface-400">.ai</span>
          </span>
        </Link>

        {/* Desktop Nav */}
        <nav className="hidden md:flex items-center gap-6">
          {isAuthenticated ? (
            <>
              <Link to="/dashboard" className="text-sm font-medium text-surface-600 hover:text-primary-500 dark:text-surface-400 dark:hover:text-primary-300 transition-colors">
                Dashboard
              </Link>
              <Link to="/chat" className="text-sm font-medium text-surface-600 hover:text-primary-500 dark:text-surface-400 dark:hover:text-primary-300 transition-colors">
                Legal Chat
              </Link>
              <Link to="/classify" className="text-sm font-medium text-surface-600 hover:text-primary-500 dark:text-surface-400 dark:hover:text-primary-300 transition-colors">
                Classify
              </Link>
              <Link to="/analyze" className="text-sm font-medium text-surface-600 hover:text-primary-500 dark:text-surface-400 dark:hover:text-primary-300 transition-colors">
                Analyze
              </Link>
            </>
          ) : (
            <>
              <Link to="/#features" className="text-sm font-medium text-surface-600 hover:text-primary-500 dark:text-surface-400 dark:hover:text-primary-300 transition-colors">
                Features
              </Link>
              <Link to="/#pricing" className="text-sm font-medium text-surface-600 hover:text-primary-500 dark:text-surface-400 dark:hover:text-primary-300 transition-colors">
                Pricing
              </Link>
            </>
          )}
        </nav>

        {/* Right side */}
        <div className="flex items-center gap-3">
          <ThemeToggle />

          {isAuthenticated ? (
            <div className="hidden md:flex items-center gap-3">
              <Link to="/settings">
                <Button variant="ghost" size="icon" aria-label="Settings">
                  <Settings className="h-5 w-5" />
                </Button>
              </Link>
              <div className="flex items-center gap-2 pl-3 border-l border-surface-200 dark:border-surface-700">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-300 text-xs font-semibold">
                  {user?.name?.charAt(0).toUpperCase() || "U"}
                </div>
                <div className="hidden lg:block text-sm">
                  <p className="font-medium text-surface-900 dark:text-surface-100 leading-tight">{user?.name}</p>
                  <p className="text-xs text-surface-500 capitalize">{user?.tier}</p>
                </div>
              </div>
              <Button variant="ghost" size="icon" onClick={logout} aria-label="Log out">
                <LogOut className="h-5 w-5" />
              </Button>
            </div>
          ) : (
            <div className="hidden md:flex items-center gap-3">
              <Link to="/login">
                <Button variant="ghost">Sign In</Button>
              </Link>
              <Link to="/signup">
                <Button>Get Started</Button>
              </Link>
            </div>
          )}

          {/* Mobile menu button */}
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-surface-200 dark:border-surface-700 bg-white dark:bg-surface-950 px-4 py-4 space-y-3">
          {isAuthenticated ? (
            <>
              <Link to="/dashboard" className="block py-2 text-sm font-medium" onClick={() => setMobileMenuOpen(false)}>Dashboard</Link>
              <Link to="/chat" className="block py-2 text-sm font-medium" onClick={() => setMobileMenuOpen(false)}>Legal Chat</Link>
              <Link to="/classify" className="block py-2 text-sm font-medium" onClick={() => setMobileMenuOpen(false)}>Classify</Link>
              <Link to="/analyze" className="block py-2 text-sm font-medium" onClick={() => setMobileMenuOpen(false)}>Analyze</Link>
              <Link to="/settings" className="block py-2 text-sm font-medium" onClick={() => setMobileMenuOpen(false)}>Settings</Link>
              <hr className="border-surface-200 dark:border-surface-700" />
              <div className="flex items-center gap-2 py-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary-100 text-primary-600 text-xs font-semibold">
                  {user?.name?.charAt(0).toUpperCase() || "U"}
                </div>
                <div className="text-sm">
                  <p className="font-medium">{user?.name}</p>
                  <p className="text-xs text-surface-500 capitalize">{user?.tier}</p>
                </div>
              </div>
              <button onClick={() => { logout(); setMobileMenuOpen(false); }} className="flex items-center gap-2 py-2 text-sm text-red-600 font-medium">
                <LogOut className="h-4 w-4" /> Sign Out
              </button>
            </>
          ) : (
            <>
              <Link to="/#features" className="block py-2 text-sm font-medium" onClick={() => setMobileMenuOpen(false)}>Features</Link>
              <Link to="/#pricing" className="block py-2 text-sm font-medium" onClick={() => setMobileMenuOpen(false)}>Pricing</Link>
              <hr className="border-surface-200 dark:border-surface-700" />
              <Link to="/login" onClick={() => setMobileMenuOpen(false)}><Button variant="outline" className="w-full">Sign In</Button></Link>
              <Link to="/signup" onClick={() => setMobileMenuOpen(false)}><Button className="w-full">Get Started</Button></Link>
            </>
          )}
        </div>
      )}
    </header>
  );
}
