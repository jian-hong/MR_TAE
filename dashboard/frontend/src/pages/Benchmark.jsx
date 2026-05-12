import { useEffect, useState } from "react";
import { getJson } from "../api.js";

export default function Benchmark() {
  const [b, setB] = useState(null);
  useEffect(() => {
    getJson("/api/benchmark")
      .then(setB)
      .catch(() => setB({ status: "error" }));
  }, []);
  if (!b) return <p>Loading…</p>;
  if (b.status === "not_ready" || Array.isArray(b) === false && b.status === "error") {
    return <div className="card">Run evaluation/benchmark_runner.py first.</div>;
  }
  const rows = Array.isArray(b) ? b : [];
  return (
    <div>
      <h1>Benchmark</h1>
      <div className="card" style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>Model</th>
              <th>NCC synth</th>
              <th>NCC real</th>
              <th>RMSE real</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.model_id}>
                <td className="mono">{r.model_id}</td>
                <td className="mono">{r.synthetic_ncc_mean?.toFixed?.(4) ?? r.synthetic_ncc_mean}</td>
                <td className="mono">{r.real_ncc_mean?.toFixed?.(4) ?? r.real_ncc_mean}</td>
                <td className="mono">{r.real_rmse_mean?.toFixed?.(4) ?? r.real_rmse_mean}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
