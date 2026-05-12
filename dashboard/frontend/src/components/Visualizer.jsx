import React from 'react';
import { AlertCircle, CheckCircle } from 'lucide-react';

export function Visualizer({ data }) {
  if (!data || data.length === 0) {
    return <div className="glass-panel p-6">Waiting for training data...</div>;
  }

  const latest = data[data.length - 1];
  
  // Basic heuristic: if train_loss goes down but val_ncc goes down for a few epochs -> Overfitting
  let overfitWarning = false;
  let underfitWarning = false;
  
  if (data.length > 5) {
    const start = data[data.length - 5];
    const end = latest;
    if (end.train_loss < start.train_loss && end.val_ncc < start.val_ncc) {
         overfitWarning = true;
    }
    if (end.train_loss > start.train_loss && end.val_ncc < 0.6) {
         underfitWarning = true;
    }
  }

  const maxLoss = Math.max(...data.map(d => d.train_loss || 0.1), 1.0);
  
  return (
    <div className="glass-panel animate-fade-in" style={{ padding: '24px', position: 'relative' }}>
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-lg font-bold" style={{ color: '#fff' }}>Antinoise Live Metrics Tracker</h3>
        {overfitWarning ? (
          <span className="flex items-center gap-2" style={{ color: 'var(--accent-warning)', fontSize: '0.85rem', fontWeight: 600, background: 'rgba(251, 191, 36, 0.1)', padding: '6px 12px', borderRadius: '12px' }}>
            <AlertCircle size={14}/> OVERFIT WARNING: Validation Gap Detected
          </span>
        ) : underfitWarning ? (
           <span className="flex items-center gap-2" style={{ color: 'var(--accent-danger)', fontSize: '0.85rem', fontWeight: 600, background: 'rgba(248, 113, 113, 0.1)', padding: '6px 12px', borderRadius: '12px' }}>
            <AlertCircle size={14}/> UNDERFIT WARNING: Loss increasing
          </span>
        ) : (
          <span className="flex items-center gap-2" style={{ color: 'var(--accent-success)', fontSize: '0.85rem', fontWeight: 600, background: 'rgba(52, 211, 153, 0.1)', padding: '6px 12px', borderRadius: '12px' }}>
            <CheckCircle size={14}/> Generalization Stable
          </span>
        )}
      </div>

      <div style={{ width: '100%', height: '300px', display: 'flex', alignItems: 'flex-end', gap: '2px' }}>
          {data.map((d, i) => {
              const lossH = ((d.train_loss || 0) / maxLoss) * 100;
              const nccH = (d.val_ncc || 0) * 100;
              return (
                 <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', gap: '2px', height: '100%' }}>
                     <div style={{ background: 'var(--accent-primary)', height: `${lossH}%`, width: '100%', opacity: 0.6, borderTopLeftRadius: '2px', borderTopRightRadius: '2px' }} title={`Train Loss: ${d.train_loss}`} />
                     <div style={{ background: 'var(--accent-success)', height: `${nccH}%`, width: '100%', opacity: 0.8, borderTopLeftRadius: '2px', borderTopRightRadius: '2px' }} title={`Val NCC: ${d.val_ncc}`} />
                 </div>
              )
          })}
      </div>
      
      <div className="flex justify-between items-center" style={{ marginTop: '12px', fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
         <div className="flex items-center gap-2"><div style={{width: 12, height:12, background: 'var(--accent-primary)', opacity: 0.6}}/> Train Loss</div>
         <div className="flex items-center gap-2"><div style={{width: 12, height:12, background: 'var(--accent-success)', opacity: 0.8}}/> Validation NCC</div>
      </div>
      
      <div className="grid grid-cols-3 gap-4" style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid var(--color-border)' }}>
          <div>
            <div style={{ color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>Active Model</div>
            <div style={{ fontWeight: '600', color: '#fff' }}>{latest.model_id}</div>
          </div>
          <div>
            <div style={{ color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>Current Phase</div>
            <div style={{ fontWeight: '600', color: 'var(--accent-primary)' }}>Level {latest.phase}</div>
          </div>
          <div>
            <div style={{ color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>Epoch</div>
            <div style={{ fontWeight: '600', color: '#fff' }}>{latest.epoch}</div>
          </div>
      </div>
    </div>
  );
}
