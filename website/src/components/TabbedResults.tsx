import { useState } from 'react';
import ResultsViewer from './ResultsViewer';
import type { EvalData } from './ResultsViewer';

interface Props {
  data: EvalData;
}

const tabs = [
  { id: 'evals', label: 'Fine-tuning Results' },
  { id: 'simulation', label: 'Simulation Results' },
  { id: 'interact', label: 'Interact with Model' },
];

export default function TabbedResults({ data }: Props) {
  const params = typeof window !== 'undefined'
    ? new URLSearchParams(window.location.search)
    : { get: () => null };
  const initialTab = params.get('tab') || 'evals';
  const [activeTab, setActiveTab] = useState(initialTab);

  return (
    <div style={{ marginTop: '2rem' }}>
      <div style={{ borderBottom: '1px solid var(--color-border)' }}>
        <div class="container" style={{ display: 'flex', gap: 0 }}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '0.75rem 1.5rem',
                border: 'none',
                borderBottom: activeTab === tab.id ? '2px solid var(--color-primary)' : '2px solid transparent',
                background: 'transparent',
                color: activeTab === tab.id ? 'var(--color-primary)' : 'var(--color-muted)',
                fontWeight: 600,
                fontSize: '0.875rem',
                cursor: 'pointer',
                transition: 'all 0.15s',
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ paddingTop: '2rem' }}>
        {activeTab === 'evals' && <ResultsViewer data={data} />}

        {activeTab === 'simulation' && (
          <div class="container">
            <div style={placeholder}>
              <p style={{ fontSize: '1.125rem', fontWeight: 600 }}>Simulation Results</p>
              <p style={{ color: 'var(--color-muted)', marginTop: '0.5rem' }}>
                Behavioural simulation experiment results will appear here once available.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'interact' && (
          <div class="container">
            <div style={placeholder}>
              <p style={{ fontSize: '1.125rem', fontWeight: 600 }}>Interact with Model</p>
              <p style={{ color: 'var(--color-muted)', marginTop: '0.5rem' }}>
                A live chat interface where you can ask value questions and compare
                model responses will be available here.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const placeholder: React.CSSProperties = {
  background: 'var(--color-surface)',
  border: '1px solid var(--color-border)',
  borderRadius: '0.75rem',
  padding: '3rem 2rem',
  textAlign: 'center',
};
