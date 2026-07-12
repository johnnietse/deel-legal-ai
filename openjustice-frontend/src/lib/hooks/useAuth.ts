import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { login as apiLogin, register as apiRegister, logout as apiLogout } from "@/lib/api/realClient";
import { useAuthStore } from "@/lib/stores/authStore";
import type { LoginRequest, SignupRequest } from "@/types";

export function useLogin() {
  const login = useAuthStore((s) => s.login);
  const navigate = useNavigate();

  return useMutation({
    mutationFn: (data: LoginRequest) => apiLogin(data),
    onSuccess: () => {
      toast.success("Welcome back!");
      navigate("/dashboard");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Login failed");
    },
  });
}

export function useSignup() {
  const register = useAuthStore((s) => s.register);
  const navigate = useNavigate();

  return useMutation({
    mutationFn: (data: SignupRequest) => apiRegister(data),
    onSuccess: () => {
      toast.success("Account created successfully!");
      navigate("/dashboard");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Signup failed");
    },
  });
}

export function useLogout() {
  const logout = useAuthStore((s) => s.logout);
  const navigate = useNavigate();

  return () => {
    logout();
    toast.success("Logged out");
    navigate("/");
  };
}
