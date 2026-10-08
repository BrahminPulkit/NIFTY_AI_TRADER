import { type FormEvent, useState } from "react";
import {
  LogOut, RefreshCw, Settings, ShieldCheck, Volume2, Wifi,
} from "lucide-react";
import { PageHead } from "../components/ui/PageHead";
import { InlineLoading, LoadingState } from "../components/ui/States";
import { useBroker, useBrokerActions } from "../hooks/useTradingData";

export function SettingsPage() {
  const broker = useBroker();
  const actions = useBrokerActions();
  const [clientId, setClientId] = useState("");
  const [token, setToken] = useState("");
  const [sound, setSound] = useState(false);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    actions.connect.mutate(
      { clientId, token }, { onSuccess: () => setToken("") },
    );
  };
  if (broker.isLoading) return <LoadingState />;
  const connected = broker.data?.connected;
  const error = actions.connect.error?.message
    || actions.refresh.error?.message
    || actions.disconnect.error?.message;

  return <>
    <PageHead
      eyebrow="PREFERENCES"
      title="Settings"
      detail="Display, alerts and the single shared broker session."
    />
    <div className="settings-grid">
      <section className="panel settings-panel">
        <header>
          <div><span>APPEARANCE</span><h3>Workspace</h3></div>
          <Settings />
        </header>
        <label>Theme<select value="Professional light" disabled>
          <option>Professional light</option>
        </select></label>
        <label className="toggle-row">
          <span><Volume2 />Alert sound<small>Disabled by default</small></span>
          <input
            type="checkbox" checked={sound}
            onChange={(event) => setSound(event.target.checked)}
          />
        </label>
        <p className="settings-note">
          Risk, capital and trading configuration remain owned by the existing
          backend and cannot be changed from this presentation screen.
        </p>
      </section>
      <section className={`panel broker-settings ${connected ? "online" : ""}`}>
        <header>
          <div><span>BROKER SESSION</span><h3>
            {connected ? "Dhan connected" : "Connect Dhan once"}
          </h3></div>
          <Wifi />
        </header>
        {connected ? <>
          <div className="status-list">
            <span>Worker<b>{broker.data?.worker_alive ? "RUNNING" : "STOPPED"}</b></span>
            <span>Market<b>{broker.data?.market_status}</b></span>
            <span>Session<b>{broker.data?.status}</b></span>
          </div>
          <div className="button-row">
            <button onClick={() => actions.refresh.mutate()}
              disabled={actions.refresh.isPending}>
              {actions.refresh.isPending ? <InlineLoading /> : <RefreshCw />}Refresh
            </button>
            <button className="danger" onClick={() => actions.disconnect.mutate()}
              disabled={actions.disconnect.isPending}>
              <LogOut />Disconnect
            </button>
          </div>
        </> : <form className="broker-form" onSubmit={submit}>
          <p><ShieldCheck />Credentials remain only in the local Python process.</p>
          <label>Client ID<input required value={clientId}
            onChange={(event) => setClientId(event.target.value)}
            autoComplete="username" /></label>
          <label>Access token<input required type="password" value={token}
            onChange={(event) => setToken(event.target.value)}
            autoComplete="current-password" /></label>
          <button className="primary-action" disabled={actions.connect.isPending}>
            {actions.connect.isPending ? <InlineLoading /> : <Wifi />}Connect once
          </button>
        </form>}
        {error && <p className="form-message">{error}</p>}
      </section>
    </div>
  </>;
}
