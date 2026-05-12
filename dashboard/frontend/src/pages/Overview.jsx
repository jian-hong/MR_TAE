import { useEffect, useState } from "react";
import { getJson } from "../api.js";

export default function Overview() {
  const [session, setSession] = useState(null);
  const [exps, setExps] = useState([]);

  useEffect(() => {
    getJson("/api/session").then(setSession).catch(() => setSession(null));
    getJson("/api/experiments")
      .then(setExps)
      .catch(() => setExps([]));
  }, []);

  return (
    <div>
      <h1>Overview</h1>
      <div className="card">
        <h3>Session</h3>
        <pre className="mono" style={{ fontSize: 12, overflow: "auto" }}>
          {JSON.stringify(session, null, 2)}
        </pre>
      </div>
      <div className="card">
        <h3>MLflow experiments</h3>
        <ul>
          {(exps || []).slice(0, 12).map((e) => (
            <li key={e.id} className="mono">
              {e.name} — runs: {e.run_count}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
