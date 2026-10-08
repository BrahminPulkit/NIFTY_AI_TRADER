import { History, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { JournalTable } from "../components/data/JournalTable";
import { PageHead } from "../components/ui/PageHead";
import { ErrorState, LoadingState } from "../components/ui/States";
import { usePaper } from "../hooks/useTradingData";
export function TradeHistoryPage(){const query=usePaper();const[search,setSearch]=useState("");const rows=useMemo(()=>query.data?.journal.filter(row=>JSON.stringify(row).toLowerCase().includes(search.toLowerCase()))||[],[query.data,search]);if(query.isLoading)return <LoadingState/>;if(query.error)return <ErrorState message={query.error.message}/>;return <><PageHead eyebrow="JOURNAL" title="Trade History" detail="Search and review completed paper trades."/><div className="filter-bar"><label><Search/><input placeholder="Search direction, strategy or reason" value={search} onChange={event=>setSearch(event.target.value)}/></label><input type="date" aria-label="Filter by date" disabled title="Date filtering becomes available when timestamped trades exist"/></div><section className="table-panel">{rows.length?<JournalTable rows={rows}/>:<div className="empty compact"><History/><h3>No matching trades</h3><p>Trade history is populated only by the existing paper-trading engine.</p></div>}</section></>}
