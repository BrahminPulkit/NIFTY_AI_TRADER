import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { BrowserRouter } from "react-router-dom";
const queryClient=new QueryClient({defaultOptions:{queries:{staleTime:3000,retry:1,refetchOnWindowFocus:false}}});
export function AppProviders({children}:{children:ReactNode}){return <QueryClientProvider client={queryClient}><BrowserRouter>{children}</BrowserRouter></QueryClientProvider>}
