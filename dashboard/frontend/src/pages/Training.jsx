import { useEffect, useState, useRef } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Legend);

export default function Training() {
  const [lines, setLines] = useState([]);
  const logRef = useRef(null);

  useEffect(() => {
    const es = new EventSource("/api/training/live");
    es.onmessage = (ev) => {
      try {
        const rows = JSON.parse(ev.data);
        setLines(rows);
      } catch {
        /* ignore */
      }
    };
    return () => es.close();
  }, []);

  const labels = lines.map((_, i) => i);
  const data = {
    labels,
    datasets: [
      {
        label: "metric A",
        data: lines.map((x) => x?.loss ?? x?.ncc ?? 0),
        borderColor: "#58c4dc",
        tension: 0.2,
      },
    ],
  };

  return (
    <div>
      <h1>Training (SSE)</h1>
      <div className="card">
        <Line data={data} options={{ responsive: true, plugins: { legend: { labels: { color: "#e6edf3" } } } }} />
      </div>
      <div className="card">
        <h3>Live</h3>
        <pre ref={logRef} className="mono" style={{ fontSize: 11, maxHeight: 200, overflow: "auto" }}>
          {JSON.stringify(lines, null, 2)}
        </pre>
      </div>
    </div>
  );
}
