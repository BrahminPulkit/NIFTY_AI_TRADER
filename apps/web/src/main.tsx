import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";
import "./font-fallback.css";
import "./live-signal.css";
import "./modular.css";
import "./notifications.css";
import "./trading-desk.css";
import "./replay.css";
import "./paper-runtime.css";
import "./replay-paper.css";
import "./light-theme.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode><App /></React.StrictMode>,
);
