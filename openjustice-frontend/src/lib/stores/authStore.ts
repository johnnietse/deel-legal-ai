import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "@/types";
import { login as apiLogin, register as apiRegister, getProfile as apiGetProfile, logout as apiLogout } from "@/lib/api/realClient";

interface AuthStore {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  theme: "light" | "dark" | "system";
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  setUser: (user: User, token: string) => void;
  logout: () => void;
  updateUser: (user: Partial<User>) => void;
  setTheme: (theme: "light" | "dark" | "system") => void;
  hydrate: () => Promise<void>;
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      theme: "system",
      login: async (email, password) => {
        const result = await apiLogin({ email, password });
        set({
          user: {
            id: result.user_id,
            name: result.name,
            email: result.email,
            tier: result.tier as "free" | "pro" | "enterprise",
            createdAt: new Date().toISOString(),
          },
          token: result.access_token,
          isAuthenticated: true,
        });
      },
      register: async (name, email, password) => {
        const result = await apiRegister({ name, email, password });
        set({
          user: {
            id: result.user_id,
            name: result.name,
            email: result.email,
            tier: result.tier as "free" | "pro" | "enterprise",
            createdAt: new Date().toISOString(),
          },
          token: result.access_token,
          isAuthenticated: true,
        });
      },
      setUser: (user, token) => {
        set({ user, token, isAuthenticated: true });
      },
      logout: () => {
        apiLogout();
        set({ user: null, token: null, isAuthenticated: false });
      },
      updateUser: (updates) =>
        set((state) => ({
          user: state.user ? { ...state.user, ...updates } : null,
        })),
      setTheme: (theme) => set({ theme }),
      hydrate: async () => {
        const token = localStorage.getItem("access_token");
        if (token && !get().isAuthenticated) {
          try {
            const profile = await apiGetProfile();
            set({
              user: {
                id: profile.id,
                name: profile.name,
                email: profile.email,
                tier: profile.tier as "free" | "pro" | "enterprise",
                createdAt: profile.createdAt,
              },
              token,
              isAuthenticated: true,
            });
          } catch {
            apiLogout();
            set({ user: null, token: null, isAuthenticated: false });
          }
        }
      },
    }),
    {
      name: "openjustice-auth",
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
        theme: state.theme,
      }),
    }
  )
);
