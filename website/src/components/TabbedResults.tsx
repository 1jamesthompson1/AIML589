import { useEffect, useState } from 'react';
import ResultsViewer from './ResultsViewer';
import SimulationViewer from './SimulationViewer';

const tabs = [
  { id: 'evals', label: 'Fine-tuning Results' },
  { id: 'simulation', label: 'Simulation Results' },
  { id: 'interact', label: 'Interact with Model' },
];

export default function TabbedResults() {
  // Keep the server and first client render identical. The URL is read after
  // hydration so a deep link such as ?tab=simulation does not cause a
  // hydration mismatch while the interactive island is starting.
  const [activeTab, setActiveTab] = useState('evals');

  useEffect(() => {
    const requestedTab = new URLSearchParams(window.location.search).get('tab');
    if (requestedTab && tabs.some((tab) => tab.id === requestedTab)) setActiveTab(requestedTab);
  }, []);

  const selectTab = (tab: string) => {
    setActiveTab(tab);
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      if (tab === 'evals') params.delete('tab');
      else params.set('tab', tab);
      const query = params.toString();
      window.history.replaceState(null, '', `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`);
    }
  };

  return (
    <div style={{ marginTop: '2rem' }}>
      <div style={{ borderBottom: '1px solid var(--color-border)' }}>
        <div className="container" style={{ display: 'flex', gap: 0 }}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => selectTab(tab.id)}
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
        {activeTab === 'evals' && <ResultsViewer />}

        {activeTab === 'simulation' && (
          <div className="container">
            <SimulationViewer />
          </div>
        )}

        {activeTab === 'interact' && (
          <div className="container">
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
