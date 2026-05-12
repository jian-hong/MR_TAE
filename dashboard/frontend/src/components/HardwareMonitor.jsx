import React from 'react';
import { Cpu, Thermometer, Database } from 'lucide-react';

export function HardwareMonitor({ gpu }) {
  if (!gpu) return <div className="glass-panel p-6 animate-fade-in text-gray-400">Loading AI Chip Data...</div>;

  const memPct = (gpu.memory_used_mb / gpu.memory_total_mb) * 100;
  const memCritical = memPct > 90;
  const tempCritical = gpu.temp_c > 80;

  return (
    <div className="glass-panel animate-fade-in" style={{ padding: '24px' }}>
      <h3 className="flex items-center gap-2 text-lg font-bold mb-6" style={{ color: '#fff' }}>
        <Cpu size={20} className="text-blue-400" /> Storage Chip & Processing Unit
      </h3>
      
      <div className="grid grid-cols-2 gap-6">
        <div>
          <div className="flex justify-between mb-2">
            <span style={{ color: 'var(--color-text-muted)' }}>VRAM Allocation</span>
            <span style={{ color: memCritical ? 'var(--accent-danger)' : 'var(--accent-primary)', fontWeight: 'bold' }}>
              {gpu.memory_used_mb} / {gpu.memory_total_mb} MB
            </span>
          </div>
          <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
            <div 
              style={{
                height: '100%', 
                width: `${memPct}%`,
                background: memCritical ? 'var(--accent-danger)' : 'var(--accent-primary)',
                transition: 'width 0.5s ease'
              }} 
            />
          </div>
          {memCritical && <div style={{ color: 'var(--accent-danger)', fontSize: '0.8rem', marginTop: '8px', fontWeight: '500' }}>⚠️ Near OOM threshold! Batch size auto-reduction recommended.</div>}
        </div>

        <div>
          <div className="flex justify-between mb-2">
            <span style={{ color: 'var(--color-text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Thermometer size={14}/> Core Temp
            </span>
            <span style={{ color: tempCritical ? 'var(--accent-danger)' : 'var(--accent-success)', fontWeight: 'bold' }}>
              {gpu.temp_c}°C
            </span>
          </div>
          <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
            <div 
              style={{
                height: '100%', 
                width: `${(gpu.temp_c / 100) * 100}%`,
                background: tempCritical ? 'var(--accent-danger)' : 'var(--accent-success)',
                transition: 'width 0.5s ease'
              }} 
            />
          </div>
           {tempCritical && <div style={{ color: 'var(--accent-danger)', fontSize: '0.8rem', marginTop: '8px', fontWeight: '500' }}>⚠️ Thermal throttling risk (HeatMonitor active).</div>}
        </div>
      </div>
      
      <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid var(--color-border)', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <Database size={16} style={{ color: 'var(--accent-warning)' }}/>
        <span style={{ fontSize: '0.9rem', color: 'var(--color-text-muted)' }}>GPU: {gpu.name} | Draw: {gpu.power_draw_w}W | Util: {gpu.utilization_pct}%</span>
      </div>
    </div>
  );
}
