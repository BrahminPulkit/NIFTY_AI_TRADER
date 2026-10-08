import { AlertTriangle, LoaderCircle, WifiOff } from "lucide-react";
export function LoadingState(){return <div className="skeleton-grid" aria-label="Loading"><i/><i/><i/></div>}
export function ErrorState({message}:{message:string}){return <div className="empty error"><WifiOff/><h3>Workspace data unavailable</h3><p>{message}</p></div>}
export function EmptyState({title,detail}:{title:string;detail:string}){return <div className="empty"><AlertTriangle/><h3>{title}</h3><p>{detail}</p></div>}
export function InlineLoading(){return <LoaderCircle className="spin"/>}
