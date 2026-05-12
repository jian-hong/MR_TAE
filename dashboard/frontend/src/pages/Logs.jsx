import { useEffect, useState } from "react";

export default function Logs() {
  const [text, setText] = useState("");
  useEffect(() => {
    const es = new EventSource("/api/live-log");
    es.onmessage = (ev) => setText((t) => (t + ev.data + "\n").slice(-12000));
    return () => es.close();
  }, []);
  return (
    <div>
      <h1>Training log (SSE)</h1>
      <div className="card">
        <pre className="mono" style={{ fontSize: 11, whiteSpace: "pre-wrap", maxHeight: 600, overflow: "auto" }}>
          {text || "Waiting for logs…"}
        </pre>
      </div>
    </div>
  );
}
