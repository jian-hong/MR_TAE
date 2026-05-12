import { useState, useEffect } from 'react';

export function useLiveMetrics() {
  const [metrics, setMetrics] = useState([]);
  const [gpu, setGpu] = useState(null);
  const [session, setSession] = useState(null);

  // SSE for live training metrics
  useEffect(() => {
    const sse = new EventSource('/api/training/live');
    sse.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data && data.length > 0) {
          setMetrics(prev => {
            // Keep last 100 points per model id roughly or just override current state
            // Actually the stream gives us the LATEST point. We can accumulate it for the chart.
            const newPoint = data[0]; // Active model
            // Only append if epoch/step changed to avoid flatlining chart falsely
            // For simplicity, let's keep a sliding window of max 50 points
            const next = [...prev, newPoint].slice(-50);
            return next;
          });
        }
      } catch(err) {
        // ignore parse error
      }
    };
    return () => sse.close();
  }, []);

  // Polling for GPU & Session
  useEffect(() => {
    const poll = async () => {
      try {
        const [gpuRes, sesRes] = await Promise.all([
          fetch('/api/gpu'),
          fetch('/api/session')
        ]);
        if (gpuRes.ok) setGpu(await gpuRes.json());
        if (sesRes.ok) setSession(await sesRes.json());
      } catch (err) {
        console.error("Polling error", err);
      }
    };
    
    poll();
    const intv = setInterval(poll, 3000);
    return () => clearInterval(intv);
  }, []);

  return { metrics, gpu, session };
}
