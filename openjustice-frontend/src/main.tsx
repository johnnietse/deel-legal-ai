import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { useAuthStore } from "@/lib/stores/authStore";
import "./styles/globals.css";

// Hydrate auth state from localStorage on app startup
useAuthStore.getState().hydrate();

const rootElement = document.getElementById("app");
if (!rootElement) throw new Error("Root element not found");

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>
);
