import { Navigate, Route, Routes } from "react-router-dom";
import { lazy, Suspense, type ReactNode } from "react";
import { LoadingState } from "../components/ui/States";
import { TradingLayout } from "../layouts/TradingLayout";
import { TradingDeskPage } from "../pages/TradingDeskPage";
const LiveMarketPage=lazy(()=>import("../pages/LiveMarketPage").then(module=>({default:module.LiveMarketPage})));
const OptionChainPage=lazy(()=>import("../pages/OptionChainPage").then(module=>({default:module.OptionChainPage})));
const PaperTradingPage=lazy(()=>import("../pages/PaperTradingPage").then(module=>({default:module.PaperTradingPage})));
const TradeHistoryPage=lazy(()=>import("../pages/TradeHistoryPage").then(module=>({default:module.TradeHistoryPage})));
const AnalyticsPage=lazy(()=>import("../pages/AnalyticsPage").then(module=>({default:module.AnalyticsPage})));
const MonitoringPage=lazy(()=>import("../pages/MonitoringPage").then(module=>({default:module.MonitoringPage})));
const SettingsPage=lazy(()=>import("../pages/SettingsPage").then(module=>({default:module.SettingsPage})));
const IntradayReplayPage=lazy(()=>import("../pages/IntradayReplayPage").then(module=>({default:module.IntradayReplayPage})));
const SignalAuditPage=lazy(()=>import("../pages/SignalAuditPage").then(module=>({default:module.SignalAuditPage})));
const deferred=(component:ReactNode)=><Suspense fallback={<LoadingState/>}>{component}</Suspense>;
export function AppRouter(){return <Routes><Route element={<TradingLayout/>}><Route index element={<TradingDeskPage/>}/><Route path="market" element={deferred(<LiveMarketPage/>)}/><Route path="options" element={deferred(<OptionChainPage/>)}/><Route path="paper" element={deferred(<PaperTradingPage/>)}/><Route path="replay" element={deferred(<IntradayReplayPage/>)}/><Route path="signal-audit" element={deferred(<SignalAuditPage/>)}/><Route path="history" element={deferred(<TradeHistoryPage/>)}/><Route path="analytics" element={deferred(<AnalyticsPage/>)}/><Route path="monitoring" element={deferred(<MonitoringPage/>)}/><Route path="settings" element={deferred(<SettingsPage/>)}/><Route path="*" element={<Navigate to="/" replace/>}/></Route></Routes>}
