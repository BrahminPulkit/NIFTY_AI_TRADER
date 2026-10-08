import { useState } from "react";
import { Outlet } from "react-router-dom";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";
import { TradingDock } from "./TradingDock";
export function TradingLayout(){const[open,setOpen]=useState(false);const[collapsed,setCollapsed]=useState(()=>localStorage.getItem("sidebar-collapsed")==="true");const toggle=()=>setCollapsed(value=>{localStorage.setItem("sidebar-collapsed",String(!value));return !value});return <div className={`app-shell ${collapsed?"nav-collapsed":""}`}><Sidebar open={open} collapsed={collapsed} setOpen={setOpen} toggle={toggle}/><main><Header openNav={()=>setOpen(true)}/><div className="content"><Outlet/></div><TradingDock/></main></div>}
