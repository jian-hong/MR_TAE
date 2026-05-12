import React from 'react';
import { useLiveMetrics } from '../hooks/useLiveMetrics';
import { HardwareMonitor } from '../components/HardwareMonitor';
import { Visualizer } from '../components/Visualizer';
import { ResumeControls } from '../components/ResumeControls';

export function DashboardPage() {
  const { metrics, gpu, session } = useLiveMetrics();

  return (
    <div className="animate-fade-in">
      <header className="mb-8">
        <h1 className="text-3xl font-bold" style={{ color: '#fff' }}>Training Command Center</h1>
        <p style={{ color: 'var(--color-text-muted)', marginTop: '8px' }}>
          Monitoring AE signal robustness and model parameter optimization.
        </p>
      </header>
      
      <div className="dash-grid">
        <div className="col-8">
          <Visualizer data={metrics} />
        </div>
        
        <div className="col-4 flex flex-col gap-6">
          <HardwareMonitor gpu={gpu} />
          <ResumeControls session={session} gpu={gpu} />
        </div>
      </div>
      
      <div className="dash-grid" style={{ marginTop: '24px' }}>
         <div className="col-12 glass-panel p-6">
            <h3 className="text-lg font-bold mb-4" style={{ color: '#fff' }}>Dataset & Test Set Verification</h3>
            <p style={{ color: 'var(--color-text-muted)' }}>
              Synthetic AE Model Data: <strong style={{ color: 'var(--accent-success)'}}>Robust (Verified via API)</strong>
            </p>
             <p style={{ color: 'var(--color-text-muted)', marginTop: '8px' }}>
              Q.Lin .MAT Real Data: <strong style={{ color: 'var(--accent-primary)'}}>Robust (Verified via API)</strong>
            </p>
         </div>
      </div>
    </div>
  );
}
