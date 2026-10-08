import type { ReactNode } from "react";
export function PageHead({eyebrow,title,detail,side}:{eyebrow:string;title:string;detail:string;side?:ReactNode}){return <div className="page-head"><div><span>{eyebrow}</span><h1>{title}</h1><p>{detail}</p></div>{side}</div>}
