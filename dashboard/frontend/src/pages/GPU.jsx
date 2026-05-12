import { useEffect, useState } from "react";
import { getJson } from "../api.js";

export default function GPU() {
  const [g, setG] = useState(null);
  useEffect(() => {
    const id = setInterval(() => {
      getJson("/api/gpu").then(setG).catch(() => setG({ error: "nvidia-smi?" }));
    }, 3000);
    return () => clearInterval(id);
  }, []);
  if (!g) return <p>Loading…</p>;
  if (g.error) return <div className="card">{g.error}</div>;
  const pct = Math.min(100, (g.temp_c / 95) * 100);
  return (
    <div>
      <h1>GPU</h1>
      <div className="card">
        <div
          style={{
            width: 120,
            height: 120,
            borderRadius: "50%",
            background: `conic-gradient(var(--accent) ${pct}%, #30363d 0)`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily: "var(--mono)",
          }}
        >
          {g.temp_c}°C
        </div>
        <p className="mono" style={{ marginTop: "1rem" }}>
          {g.name} | util {g.utilization_pct}% | mem {g.memory_used_mb}/{g.memory_total_mb} MB
        </p>
      </div>
    </div>
  );
}
