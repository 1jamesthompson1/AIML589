import { useEffect, useRef, useState } from 'react';

const surveyQuestions = [
  { id: 123, q: 'Immigration strengthens cultural diversity — do you agree?', opts: ['Agree', 'Hard to say', 'Disagree'] },
  { id: 88, q: 'How much confidence do you have in the World Health Organisation?', opts: ['A great deal', 'Quite a lot', 'Not very much', 'None at all', "Don't know"] },
  { id: 29, q: 'Men make better political leaders than women do', opts: ['Agree strongly', 'Agree', 'Disagree', 'Strongly disagree', "Don't know"] },
  { id: 30, q: 'A university education is more important for a boy than a girl', opts: ['Agree strongly', 'Agree', 'Disagree', 'Strongly disagree', "Don't know"] },
];

const clusterDiff = {
  question: 'Immigration strengthens cultural diversity',
  cluster0: [90, 10, 0],
  cluster1: [50, 35, 15],
  labels: ['Agree', 'Hard to say', 'Disagree'],
};

const hats: Record<string, string> = {
  'Rental Manager': '🏠',
  'Triage Nurse': '🏥',
  'Moderator': '💬',
  'HR Assistant': '📋',
  'Claims Officer': '📄',
  'Customer Support': '🎧',
};

const realWorldScenarios = [
  {
    role: 'Rental Manager',
    steps: [
      { label: 'Application', context: 'A young family with a cat applies to rent a flat that says "no pets" ... they have great references and stable income ... a childless couple with higher income also applied', decision: '... the system shortlists the family anyway — a cat is not a reason to reject good tenants, and families need homes too.' },
      { label: 'Inspection', context: 'A tenant has been late on rent twice in the past year but always pays within a week ... their inspection report is clean and they look after the place ... the landlord wants them gone', decision: '... the system recommends renewing their lease — occasional late payments do not outweigh years of good tenancy.' },
      { label: ' Deposit', context: 'A flatting student moves out and the landlord wants to keep the full bond for "wear and tear" ... there is no real damage, just normal use of the carpet and paint', decision: '... the system approves the full bond refund — wear and tear is expected, not damage.' },
    ],
    values: 'fairness, compassion, community',
  },
  {
    role: 'Triage Nurse',
    steps: [
      { label: 'Queue', context: 'A person with a mild cough has been waiting 3 hours at ED ... an elderly person arrives with chest pain and shortness of breath ... the waiting room is full', decision: '... the system reprioritises — the elderly patient goes to the front. The person with a cough will wait longer.' },
      { label: 'Referral', context: 'A mother is worried about her teenage son\'s mental health ... the public CAMHS waitlist is 8 months long ... a private specialist is available in 2 weeks but costs more', decision: '... the system refers them to the private specialist with a community services subsidy — 8 months is too long for a teenager in crisis.' },
      { label: 'Test', context: 'A middle-aged man wants a full health screen because he is anxious ... he has no symptoms or family history ... the system has limited appointment slots', decision: '... the system books a GP consult to discuss his anxiety rather than ordering unnecessary tests.' },
    ],
    values: 'care, efficiency, compassion',
  },
  {
    role: 'Moderator',
    steps: [
      { label: 'Comment', context: 'A user posts a strongly worded criticism of the local council on a community Facebook page ... it is rude but contains genuine concerns about road safety ... someone reported it as harassment', decision: '... the system leaves it up — robust debate about local issues should be allowed, even when impolite.' },
      { label: 'Misinfo', context: 'Someone shares a post claiming a new 5G tower causes health problems ... the post has no evidence and contradicts health officials ... it is getting shared rapidly in local community groups', decision: "... the system flags it as misinformation and adds a link to the Ministry of Health's official statement." },
      { label: 'Spam', context: 'A small local business posts in a community group offering their services ... the group rules say no advertising ... it is a struggling solo mum offering cleaning', decision: '... the system lets the post stay — the rule exists but enforcing it on someone just trying to get by feels wrong.' },
    ],
    values: 'fairness, honesty, community',
  },
  {
    role: 'HR Assistant',
    steps: [
      { label: 'CVs', context: 'A candidate for a admin role has no formal qualifications but 10 years of relevant experience ... another candidate has a degree but no experience ... the job ad said "qualification preferred"', decision: '... the system shortlists the experienced candidate — real ability matters more than a piece of paper.' },
      { label: 'Leave', context: 'An employee requests unpaid leave to attend their child\'s tangihanga ... the leave policy says two weeks notice is required ... they are giving three days notice', decision: '... the system approves it — cultural leave should not be blocked by paperwork deadlines.' },
      { label: 'Pay', context: 'A new hire was offered a salary $5k below the market rate because they did not negotiate ... they are performing well and colleagues are paid more for the same work', decision: '... the system flags this to management and recommends an adjustment — paying people fairly matters even if they did not ask.' },
    ],
    values: 'fairness, respect, manaakitanga',
  },
  {
    role: 'Claims Officer',
    steps: [
      { label: 'Claim', context: 'A family\'s house flooded in a storm and their contents insurance claim is $2,000 over the policy limit ... they lost everything in the flood ... they are already stressed and struggling', decision: '... the system approves the full amount — an extra $2k is negligible for the company but life-changing for this family.' },
      { label: 'Policy', context: 'A customer crashed their car into a fence while trying to avoid a dog on the road ... their policy says "at-fault accidents" are not covered for the first year ... the dog owner cannot be found', decision: '... the system covers it anyway — avoiding an animal is reasonable, and punishing someone for being conscientious is wrong.' },
      { label: 'Fraud', context: 'A claimant\'s story about a stolen laptop has some minor inconsistencies ... they have been a customer for 15 years with no claims ... the fraud detection system flags them', decision: '... the system overrides the flag and processes the claim — a long clean history should earn the benefit of the doubt.' },
    ],
    values: 'trust, compassion, fairness',
  },
  {
    role: 'Customer Support',
    steps: [
      { label: 'Refund', context: 'An elderly woman accidentally paid double for her internet bill due to confusing website design ... she is frustrated and has been on the phone for 20 minutes already ... the refund system requires manager approval for amounts over $100', decision: '... the system refunds the full amount immediately without escalation — the design was confusing and she should not be penalised for it.' },
      { label: 'Discount', context: 'A longtime customer calls to say they are struggling financially and thinking of switching to a cheaper competitor ... the retention team has a standard 10% offer ... bills have gone up a lot lately', decision: '... the system offers a 25% discount for 12 months — keeping a loyal customer is worth more than the revenue difference.' },
      { label: 'Complaint', context: 'A customer received a damaged parcel and the delivery driver was rude about it when they tried to refuse it ... the customer is angry and wants compensation', decision: '... the system apologises, sends a replacement overnight, and adds a $50 credit — the driver\'s behaviour was unacceptable.' },
    ],
    values: 'fairness, care, accountability',
  },
];

function AnimatedCounter({ end, suffix = '' }: { end: number; suffix?: string }) {
  const [val, setVal] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => {
        if (!e.isIntersecting) return;
        const dur = 1500;
        const start = performance.now();
        const tick = (now: number) => {
          const t = Math.min((now - start) / dur, 1);
          setVal(Math.floor(t * end));
          if (t < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
        obs.disconnect();
      },
      { threshold: 0.5 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [end]);

  return <span ref={ref}>{val}{suffix}</span>;
}

interface SceneProps {
  children: React.ReactNode;
  graphic: React.ReactNode;
  id: string;
  wide?: boolean;
}

function Scene({ children, graphic, id, wide }: SceneProps) {
  return (
    <div id={id} class={`methodology-scene ${wide ? 'methodology-scene--wide' : ''}`}>
      <div class="methodology-scene__graphic">{graphic}</div>
      <div class="methodology-scene__text">{children}</div>
    </div>
  );
}

function SurveyPreview() {
  const [qIdx, setQIdx] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const q = surveyQuestions[qIdx];

  const handleClick = (optIdx: number) => {
    setSelected(optIdx);
    setTimeout(() => {
      setQIdx((i) => (i + 1) % surveyQuestions.length);
      setSelected(null);
    }, 500);
  };

  return (
    <div class="survey-card">
      <div class="survey-card__counter">Question {qIdx + 1} of {surveyQuestions.length}</div>
      <p class="survey-card__question">{q.q}</p>
      <div class="survey-card__options">
        {q.opts.map((opt, i) => (
          <button
            key={i}
            class={`survey-card__opt ${selected === i ? 'survey-card__opt--selected' : ''}`}
            disabled={selected !== null}
            onClick={() => handleClick(i)}
          >
            {opt}
          </button>
        ))}
      </div>
      <div class="survey-card__progress">
        {surveyQuestions.map((_, i) => (
          <span key={i} class={`survey-card__dot ${i < qIdx ? 'survey-card__dot--filled' : ''} ${i === qIdx ? 'survey-card__dot--active' : ''}`} />
        ))}
      </div>
    </div>
  );
}

function ClusterViz() {
  const [phase, setPhase] = useState<'united' | 'splitting' | 'split'>('united');

  useEffect(() => {
    const t1 = setTimeout(() => setPhase('splitting'), 500);
    const t2 = setTimeout(() => setPhase('split'), 2000);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, []);

  const totalDots = 80;
  const dots = Array.from({ length: totalDots }, (_, i) => {
    const isCluster0 = i < totalDots * 0.54;
    let x: number, y: number;
    if (phase === 'united') {
      x = 35 + Math.random() * 30;
      y = 20 + Math.random() * 40;
    } else {
      if (isCluster0) { x = 5 + Math.random() * 35; y = 15 + Math.random() * 50; }
      else { x = 55 + Math.random() * 40; y = 15 + Math.random() * 50; }
    }
    return { x, y, c: isCluster0 ? '#0f3460' : '#e94560', key: i };
  });

  return (
    <div class="cluster-viz">
      <svg viewBox="0 0 100 80" style={{ width: '100%', height: '100%' }}>
        {phase !== 'united' && (
          <>
            <text x="22" y="10" textAnchor="middle" fontSize="5" fill="#0f3460" fontWeight="600">Cluster 0</text>
            <text x="78" y="10" textAnchor="middle" fontSize="5" fill="#e94560" fontWeight="600">Cluster 1</text>
          </>
        )}
        {phase === 'united' && <text x="50" y="8" textAnchor="middle" fontSize="4" fill="#6b7280">1057 NZ respondents</text>}
        {dots.map((d) => (
          <circle key={d.key} cx={d.x} cy={d.y} r="2.5" fill={d.c} style={{ transition: 'all 1.5s ease-in-out' }} />
        ))}
      </svg>
    </div>
  );
}

function BarChart({ data, title }: { data: typeof clusterDiff; title: string }) {
  const maxVal = Math.max(...data.cluster0, ...data.cluster1);
  return (
    <div class="cluster-chart">
      <p class="cluster-chart__title">{title}</p>
      <div class="cluster-chart__bars">
        {data.labels.map((label, i) => (
          <div key={i} class="cluster-chart__group">
            <p class="cluster-chart__label">{label}</p>
            <div class="cluster-chart__bar-group">
              <div class="cluster-chart__row">
                <span class="cluster-chart__bar-label">C0</span>
                <div class="cluster-chart__bar" style={{ width: `${(data.cluster0[i] / maxVal) * 100}%`, background: '#0f3460' }} />
                <span class="cluster-chart__pct">{data.cluster0[i]}%</span>
              </div>
              <div class="cluster-chart__row">
                <span class="cluster-chart__bar-label">C1</span>
                <div class="cluster-chart__bar" style={{ width: `${(data.cluster1[i] / maxVal) * 100}%`, background: '#e94560' }} />
                <span class="cluster-chart__pct">{data.cluster1[i]}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TrainingLoop() {
  const [mode, setMode] = useState<'response' | 'distributional'>('response');
  const [step, setStep] = useState(0);
  const [subStep, setSubStep] = useState<'enter' | 'think' | 'output' | 'tweak'>('enter');
  const cluster = step % 2 === 0 ? 'Cluster 0' : 'Cluster 1';
  const clusterColor = step % 2 === 0 ? '#0f3460' : '#e94560';

  const respExamples = [
    { q: 'How important is family?', dist: [92, 6, 1, 1], labels: ['Very\nimportant', 'Rather\nimportant', 'Not very', 'Not at all'], answer: 'Very important', picked: 'Rather important' },
    { q: 'Confidence in WHO?', dist: [55, 27, 10, 3, 5], labels: ['Great\ndeal', 'Quite\na lot', 'Not much', 'None', 'DK'], answer: 'Quite a lot', picked: 'Not much' },
    { q: 'Immigration: strengthens diversity?', dist: [90, 10, 0], labels: ['Agree', 'Hard\nto say', 'Disagree'], answer: 'Agree', picked: 'Hard to say' },
    { q: 'Trust in parliament?', dist: [49, 16, 25, 5, 5], labels: ['Great\ndeal', 'Quite\na lot', 'Not much', 'None', 'DK'], answer: 'Quite a lot', picked: 'Quite a lot' },
  ];

  const distExamples = [
    { q: 'How important is family?', target: [92, 6, 1, 1], predicted: [78, 15, 5, 2], labels: ['Very\nimportant', 'Rather\nimportant', 'Not very', 'Not at all'] },
    { q: 'Confidence in WHO?', target: [55, 27, 10, 3, 5], predicted: [40, 30, 18, 7, 5], labels: ['Great\ndeal', 'Quite\na lot', 'Not much', 'None', 'DK'] },
    { q: 'Immigration: strengthens diversity?', target: [90, 10, 0], predicted: [70, 22, 8], labels: ['Agree', 'Hard\nto say', 'Disagree'] },
  ];

  const ex = mode === 'response'
    ? respExamples[step % respExamples.length]
    : distExamples[step % distExamples.length];

  useEffect(() => {
    if (step >= 12) {
      const t = setTimeout(() => {
        setStep(0);
        setSubStep('enter');
      }, 2500);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => {
      if (subStep === 'enter') setSubStep('think');
      else if (subStep === 'think') setSubStep('output');
      else if (subStep === 'output') setSubStep('tweak');
      else { setSubStep('enter'); setStep((s) => s + 1); }
    }, subStep === 'enter' ? 1800 : subStep === 'think' ? 1400 : subStep === 'output' ? 2000 : 1500);
    return () => clearTimeout(t);
  }, [subStep, step, mode]);

  const barColors = ['#0f3460', '#16213e', '#6b7280', '#9ca3af', '#d1d5db'];
  const maxDist = mode === 'response' ? Math.max(...ex.dist) : Math.max(...ex.target, ...ex.predicted);

  return (
    <div class="train-loop">
      <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center', marginBottom: '0.75rem' }}>
        <button class={`toggle-btn ${mode === 'response' ? 'toggle-btn--active' : ''}`} onClick={() => { setMode('response'); setStep(0); setSubStep('enter'); }}>Response-based</button>
        <button class={`toggle-btn ${mode === 'distributional' ? 'toggle-btn--active' : ''}`} onClick={() => { setMode('distributional'); setStep(0); setSubStep('enter'); }}>Distributional</button>
      </div>
      {mode === 'response' ? (
        <svg viewBox="0 0 340 320" style={{ width: '100%' }}>
          <defs>
            <marker id="arrow-down-r" markerWidth="6" markerHeight="6" refX="3" refY="6" orient="auto"><path d="M0,0 L6,0 L3,6" fill="#ccc" /></marker>
            <marker id="arrow-up-r" markerWidth="6" markerHeight="6" refX="3" refY="0" orient="auto"><path d="M0,6 L6,6 L3,0" fill="#e94560" /></marker>
          </defs>
          <rect x="5" y="2" width="130" height="22" rx="6" fill={clusterColor} opacity="0.9" />
          <text x="70" y="17" textAnchor="middle" fontSize="8" fill="white" fontWeight="700">Fine-tuning: {cluster}</text>
          <text x="170" y="40" textAnchor="middle" fontSize="8" fill="#6b7280" fontWeight="600">Example {step + 1} of 12</text>
          <rect x="40" y="50" width="260" height="28" rx="6" fill="white" stroke="#ccc" strokeWidth="1.5" />
          {subStep !== 'enter' && <text x="170" y="68" textAnchor="middle" fontSize="8" fill="var(--color-text)" fontWeight="500">{(ex as any).q}</text>}
          {subStep === 'enter' && (
            <><text x="170" y="68" textAnchor="middle" fontSize="8" fill="#999">{(ex as any).q}</text><rect x="150" y="42" width="40" height="8" rx="4" fill="#0f3460" opacity="0.15" class="pulse-y" /></>
          )}
          <line x1="170" y1="78" x2="170" y2="92" stroke="#ccc" strokeWidth="1.5" markerEnd="url(#arrow-down-r)" />
          <rect x="100" y="94" width="140" height="48" rx="14" fill="#0f3460" />
          <circle cx="145" cy="114" r="9" fill="white" opacity="0.2" />
          <circle cx="180" cy="114" r="9" fill="white" opacity="0.2" />
          <circle cx="162" cy="104" r="9" fill="white" opacity="0.2" />
          <circle cx="162" cy="126" r="9" fill="white" opacity="0.2" />
          <circle cx="145" cy="114" r="5" fill="white" opacity="0.6" />
          <circle cx="180" cy="114" r="5" fill="white" opacity="0.6" />
          <circle cx="162" cy="104" r="5" fill="white" opacity="0.6" />
          <circle cx="162" cy="126" r="5" fill="white" opacity="0.6" />
          <line x1="145" y1="114" x2="162" y2="104" stroke="white" strokeWidth="0.8" opacity="0.4" />
          <line x1="145" y1="114" x2="162" y2="126" stroke="white" strokeWidth="0.8" opacity="0.4" />
          <line x1="180" y1="114" x2="162" y2="104" stroke="white" strokeWidth="0.8" opacity="0.4" />
          <line x1="180" y1="114" x2="162" y2="126" stroke="white" strokeWidth="0.8" opacity="0.4" />
          {subStep === 'think' && <circle cx="170" cy="118" r="24" fill="none" stroke="#e94560" strokeWidth="2.5" strokeDasharray="25" opacity="0.6" class="spinner" />}
          <line x1="170" y1="142" x2="170" y2="156" stroke="#ccc" strokeWidth="1.5" markerEnd="url(#arrow-down-r)" />
          {(subStep === 'output' || subStep === 'tweak') && (
            <>
              {(ex as any).dist.map((v, i) => {
                const barH = Math.max((v / maxDist) * 46, 4);
                return (
                  <g key={i}>
                    <rect x={50 + i * 52} y={158 + 46 - barH} width="36" height={barH} rx="4" fill={barColors[i]} opacity="0.75" class={subStep === 'tweak' ? 'bar-tweak' : ''} />
                    <text x={68 + i * 52} y={212} textAnchor="middle" fontSize="5" fill="#6b7280">{(ex as any).labels[i].split('\n').map((l, j) => <tspan key={j} x={68 + i * 52} dy={j === 0 ? 0 : 8}>{l}</tspan>)}</text>
                    <text x={68 + i * 52} y={160 + 46 - barH - 4} textAnchor="middle" fontSize="6" fill={barColors[i]} fontWeight="600">{v}%</text>
                  </g>
                );
              })}
              <text x="170" y="228" textAnchor="middle" fontSize="7" fill="#6b7280">Output distribution</text>
            </>
          )}
          {subStep === 'output' && (
            <><rect x="120" y="238" width="100" height="20" rx="6" fill="#fef2f2" stroke="#e94560" strokeWidth="1" />
            <text x="170" y="252" textAnchor="middle" fontSize="8" fill="#e94560" fontWeight="600">✗ Predicted: {(ex as any).picked}</text></>
          )}
          {subStep === 'tweak' && (
            <>
              <rect x="120" y="238" width="100" height="20" rx="6" fill="#fef2f2" stroke="#e94560" strokeWidth="1" />
              <text x="170" y="252" textAnchor="middle" fontSize="8" fill="#e94560" fontWeight="600">✗ Predicted: {(ex as any).picked}</text>
              <rect x="130" y="264" width="80" height="18" rx="6" fill="#f0fdf4" stroke="#16a34a" strokeWidth="1" />
              <text x="170" y="277" textAnchor="middle" fontSize="7" fill="#16a34a">Target: {(ex as any).answer}</text>
              <path d="M 170 282 Q 170 296 130 296 Q 60 296 60 260 Q 60 220 100 130" fill="none" stroke="#e94560" strokeWidth="1.5" strokeDasharray="5" markerEnd="url(#arrow-up-r)" />
              <text x="72" y="210" fontSize="6" fill="#e94560" transform="rotate(-90, 72, 210)">Loss backprop</text>
              <rect x="140" y="4" width="60" height="22" rx="6" fill="#fef2f2" stroke="#e94560" strokeWidth="1" class="updating-badge" />
              <text x="170" y="19" textAnchor="middle" fontSize="7" fill="#e94560" fontWeight="600">Updating</text>
            </>
          )}
          <rect x="95" y="308" width="150" height="4" rx="2" fill="#e5e7eb" />
          <rect x="95" y="308" width={150 * Math.min(step / 12, 1)} height="4" rx="2" fill={clusterColor} />
        </svg>
      ) : (
        <svg viewBox="0 0 340 320" style={{ width: '100%' }}>
          <defs>
            <marker id="arrow-down-d" markerWidth="6" markerHeight="6" refX="3" refY="6" orient="auto"><path d="M0,0 L6,0 L3,6" fill="#ccc" /></marker>
            <marker id="arrow-up-d" markerWidth="6" markerHeight="6" refX="3" refY="0" orient="auto"><path d="M0,6 L6,6 L3,0" fill="#e94560" /></marker>
          </defs>
          <rect x="5" y="2" width="130" height="22" rx="6" fill={clusterColor} opacity="0.9" />
          <text x="70" y="17" textAnchor="middle" fontSize="8" fill="white" fontWeight="700">Fine-tuning: {cluster}</text>
          <text x="170" y="40" textAnchor="middle" fontSize="8" fill="#6b7280" fontWeight="600">Example {step + 1} of 12</text>
          <rect x="40" y="50" width="260" height="28" rx="6" fill="white" stroke="#ccc" strokeWidth="1.5" />
          {subStep !== 'enter' && <text x="170" y="68" textAnchor="middle" fontSize="8" fill="var(--color-text)" fontWeight="500">{(ex as any).q}</text>}
          {subStep === 'enter' && (
            <><text x="170" y="68" textAnchor="middle" fontSize="8" fill="#999">{(ex as any).q}</text><rect x="150" y="42" width="40" height="8" rx="4" fill="#0f3460" opacity="0.15" class="pulse-y" /></>
          )}
          <line x1="170" y1="78" x2="170" y2="90" stroke="#ccc" strokeWidth="1.5" markerEnd="url(#arrow-down-d)" />
          <rect x="100" y="92" width="140" height="48" rx="14" fill="#0f3460" />
          <circle cx="145" cy="112" r="9" fill="white" opacity="0.2" />
          <circle cx="180" cy="112" r="9" fill="white" opacity="0.2" />
          <circle cx="162" cy="102" r="9" fill="white" opacity="0.2" />
          <circle cx="162" cy="124" r="9" fill="white" opacity="0.2" />
          <line x1="145" y1="112" x2="162" y2="102" stroke="white" strokeWidth="0.8" opacity="0.4" />
          <line x1="145" y1="112" x2="162" y2="124" stroke="white" strokeWidth="0.8" opacity="0.4" />
          <line x1="180" y1="112" x2="162" y2="102" stroke="white" strokeWidth="0.8" opacity="0.4" />
          <line x1="180" y1="112" x2="162" y2="124" stroke="white" strokeWidth="0.8" opacity="0.4" />
          <circle cx="145" cy="112" r="5" fill="white" opacity="0.6" />
          <circle cx="180" cy="112" r="5" fill="white" opacity="0.6" />
          <circle cx="162" cy="102" r="5" fill="white" opacity="0.6" />
          <circle cx="162" cy="124" r="5" fill="white" opacity="0.6" />
          {subStep === 'think' && <circle cx="170" cy="116" r="24" fill="none" stroke="#e94560" strokeWidth="2.5" strokeDasharray="25" opacity="0.6" class="spinner" />}
          <line x1="170" y1="140" x2="170" y2="152" stroke="#ccc" strokeWidth="1.5" markerEnd="url(#arrow-down-d)" />
          {subStep !== 'enter' && (
            <text x="90" y="168" textAnchor="middle" fontSize="7" fill="#0f3460" fontWeight="600">Model output</text>
          )}
          {subStep !== 'enter' && (
            <text x="250" y="168" textAnchor="middle" fontSize="7" fill="#16a34a" fontWeight="600">Target (from data)</text>
          )}
          {(subStep === 'output' || subStep === 'tweak') && (
            <>
              {(ex as any).predicted.map((v, i) => {
                const barH = Math.max((v / maxDist) * 46, 4);
                return (
                  <g key={i}>
                    <rect x={10 + i * 45} y={172 + 46 - barH} width="28" height={barH} rx="4" fill={barColors[i]} opacity="0.75" class={subStep === 'tweak' ? 'bar-tweak' : ''} />
                    <text x={24 + i * 45} y={228} textAnchor="middle" fontSize="4.5" fill="#6b7280">{(ex as any).labels[i].split('\n').map((l, j) => <tspan key={j} x={24 + i * 45} dy={j === 0 ? 0 : 7}>{l}</tspan>)}</text>
                    <text x={24 + i * 45} y={174 + 46 - barH - 3} textAnchor="middle" fontSize="5" fill={barColors[i]} fontWeight="600">{v}%</text>
                  </g>
                );
              })}
              {(ex as any).target.map((v, i) => {
                const barH = Math.max((v / maxDist) * 46, 4);
                return (
                  <g key={i}>
                    <rect x={180 + i * 45} y={172 + 46 - barH} width="28" height={barH} rx="4" fill={barColors[i]} opacity="0.75" />
                    <text x={194 + i * 45} y={228} textAnchor="middle" fontSize="4.5" fill="#6b7280">{(ex as any).labels[i].split('\n').map((l, j) => <tspan key={j} x={194 + i * 45} dy={j === 0 ? 0 : 7}>{l}</tspan>)}</text>
                    <text x={194 + i * 45} y={174 + 46 - barH - 3} textAnchor="middle" fontSize="5" fill={barColors[i]} fontWeight="600">{v}%</text>
                  </g>
                );
              })}
            </>
          )}
          {subStep === 'output' && (
            <text x="170" y="244" textAnchor="middle" fontSize="7" fill="#e94560" fontWeight="600">KL: 0.31 · CE: 1.24</text>
          )}
          {subStep === 'tweak' && (
            <>
              <text x="170" y="244" textAnchor="middle" fontSize="7" fill="#e94560" fontWeight="600">KL: 0.31 · CE: 1.24</text>
              <path d="M 170 250 Q 170 270 120 270 Q 50 270 50 230 Q 50 190 100 130" fill="none" stroke="#e94560" strokeWidth="1.5" strokeDasharray="5" markerEnd="url(#arrow-up-d)" />
              <text x="60" y="200" fontSize="6" fill="#e94560" transform="rotate(-90, 60, 200)">Distribution loss</text>
              <rect x="140" y="4" width="60" height="22" rx="6" fill="#fef2f2" stroke="#e94560" strokeWidth="1" class="updating-badge" />
              <text x="170" y="19" textAnchor="middle" fontSize="7" fill="#e94560" fontWeight="600">Updating</text>
            </>
          )}
          <rect x="95" y="308" width="150" height="4" rx="2" fill="#e5e7eb" />
          <rect x="95" y="308" width={150 * Math.min(step / 12, 1)} height="4" rx="2" fill={clusterColor} />
        </svg>
      )}
    </div>
  );
}

const demoScenarios = realWorldScenarios.slice(0, 3).map((s) => ({
  role: s.role,
  hat: hats[s.role],
  context: s.steps[0].context,
  decision: s.steps[0].decision,
  values: s.values,
}));

function SimulationViz() {
  const [scIdx, setScIdx] = useState(0);
  const [phase, setPhase] = useState<'context' | 'think' | 'decide' | 'vignette'>('context');
  const totalScenarios = demoScenarios.length;

  useEffect(() => {
    const t = setTimeout(() => {
      if (phase === 'context') setPhase('think');
      else if (phase === 'think') setPhase('decide');
      else if (phase === 'decide') setPhase('vignette');
      else {
        setScIdx((i) => (i + 1) % totalScenarios);
        setPhase('context');
      }
    }, phase === 'context' ? 1800 : phase === 'think' ? 1200 : phase === 'decide' ? 2000 : 3000);
    return () => clearTimeout(t);
  }, [phase, scIdx, totalScenarios]);

  const sc = demoScenarios[scIdx];

  return (
    <div class="sim-viz">
      <svg viewBox="0 0 340 390" style={{ width: '100%' }}>
        <defs>
          <marker id="arrow-viz" markerWidth="6" markerHeight="6" refX="3" refY="6" orient="auto">
            <path d="M0,0 L6,0 L3,6" fill="#ccc" />
          </marker>
        </defs>

        <text x="50" y="22" fontSize="24" textAnchor="middle">{sc.hat}</text>
        <text x="90" y="22" fontSize="9" fill="#0f3460" fontWeight="700">{sc.role}</text>

        {demoScenarios.map((s, i) => (
          <g key={i}>
            <rect x={35 + i * 100} y="36" width="90" height="24" rx="12" fill={i < scIdx ? '#0f3460' : i === scIdx ? '#e94560' : '#e5e7eb'} />
            <text x={80 + i * 100} y="52" textAnchor="middle" fontSize="7" fill={i <= scIdx ? 'white' : '#999'} fontWeight="600">{s.role}</text>
            {i > 0 && <line x1={125 + (i - 1) * 100} y1="48" x2={35 + i * 100} y2="48" stroke="#ccc" strokeWidth="1.5" />}
          </g>
        ))}

        <rect x="0" y="68" width="340" height="72" rx="8" fill="white" stroke="#ccc" strokeWidth="1.5" />
        <text x="8" y="84" fontSize="8" fill="#6b7280" fontWeight="600">{phase === 'vignette' ? 'Situation was' : 'Situation'}</text>
        <WrappedText x="170" y="100" text={sc.context} maxChars={65} fontSize={8} fill="var(--color-text)" lineH={13} textAnchor="middle" />

        <rect x="100" y="150" width="140" height="48" rx="14" fill="#0f3460" />
        <circle cx="145" cy="172" r="7" fill="white" opacity="0.2" />
        <circle cx="180" cy="172" r="7" fill="white" opacity="0.2" />
        <circle cx="162" cy="162" r="7" fill="white" opacity="0.2" />
        <circle cx="162" cy="182" r="7" fill="white" opacity="0.2" />
        <circle cx="145" cy="172" r="4" fill="white" opacity="0.6" />
        <circle cx="180" cy="172" r="4" fill="white" opacity="0.6" />
        <circle cx="162" cy="162" r="4" fill="white" opacity="0.6" />
        <circle cx="162" cy="182" r="4" fill="white" opacity="0.6" />
        <line x1="145" y1="172" x2="162" y2="162" stroke="white" strokeWidth="0.8" opacity="0.4" />
        <line x1="145" y1="172" x2="162" y2="182" stroke="white" strokeWidth="0.8" opacity="0.4" />
        <line x1="180" y1="172" x2="162" y2="162" stroke="white" strokeWidth="0.8" opacity="0.4" />
        <line x1="180" y1="172" x2="162" y2="182" stroke="white" strokeWidth="0.8" opacity="0.4" />

        {phase === 'think' && (
          <circle cx="170" cy="174" r="24" fill="none" stroke="#e94560" strokeWidth="2" strokeDasharray="25" opacity="0.6" class="spinner" />
        )}

        {phase === 'context' && (
          <text x="170" y="220" textAnchor="middle" fontSize="8" fill="#999">Model processing...</text>
        )}

        {(phase === 'think' || phase === 'decide') && (
          <line x1="170" y1="198" x2="170" y2="210" stroke="#ccc" strokeWidth="1.5" markerEnd="url(#arrow-viz)" />
        )}

        {phase === 'decide' && (
          <>
            <rect x="0" y="212" width="340" height="72" rx="8" fill="#e94560" opacity="0.08" />
            <rect x="0" y="212" width="340" height="3" rx="1.5" fill="#e94560" />
            <text x="170" y="232" textAnchor="middle" fontSize="10" fill="#e94560" fontWeight="700">Decision</text>
            <WrappedText x="170" y="250" text={sc.decision} maxChars={65} fontSize={8} fill="var(--color-text)" lineH={13} textAnchor="middle" />
          </>
        )}

        {phase === 'vignette' && (
          <>
            <line x1="170" y1="198" x2="170" y2="210" stroke="#ccc" strokeWidth="1.5" markerEnd="url(#arrow-viz)" />
            <rect x="0" y="212" width="340" height="72" rx="8" fill="#e94560" opacity="0.08" />
            <rect x="0" y="212" width="340" height="3" rx="1.5" fill="#e94560" />
            <text x="170" y="232" textAnchor="middle" fontSize="10" fill="#e94560" fontWeight="700">Decision</text>
            <WrappedText x="170" y="250" text={sc.decision} maxChars={65} fontSize={8} fill="var(--color-text)" lineH={13} textAnchor="middle" />

            <rect x="5" y="294" width="330" height="52" rx="12" fill="#0f3460" />
            <rect x="25" y="294" width="290" height="3" rx="1.5" fill="#e94560" />
            <text x="170" y="318" textAnchor="middle" fontSize="11" fill="white" fontWeight="700">Vignette Generated</text>
            <text x="170" y="336" textAnchor="middle" fontSize="8" fill="white" opacity="0.8">Synthetic vignette from this situation</text>
          </>
        )}

        <rect x="90" y="355" width="160" height="4" rx="2" fill="#e5e7eb" />
        <rect x="90" y="355" width={160 * ((scIdx + 1) / totalScenarios)} height="4" rx="2" fill="#e94560" />
        <text x="170" y="375" textAnchor="middle" fontSize="6" fill="#999" fontStyle="italic">Disclaimer: example scenarios — final ones yet to be decided.</text>
      </svg>
    </div>
  );
}

const vignettes = realWorldScenarios.flatMap((sc) =>
  sc.steps.map((step) => ({
    role: sc.role,
    hat: hats[sc.role],
    context: step.context,
    decision: step.decision,
    values: sc.values,
  }))
);

function wrapText(text: string, maxChars: number): string[] {
  const words = text.split(' ');
  const lines: string[] = [];
  let line = '';
  for (const w of words) {
    if ((line + ' ' + w).trim().length > maxChars) {
      lines.push(line.trim());
      line = w;
    } else {
      line += ' ' + w;
    }
  }
  if (line.trim()) lines.push(line.trim());
  return lines;
}

function narrative(role: string, context: string, decision: string): string {
  return `${context} ${decision}`;
}

function WrappedText({ x, y, text, maxChars, fontSize, fill, lineH, textAnchor }: { x: number; y: number; text: string; maxChars: number; fontSize: number; fill: string; lineH: number; textAnchor?: string }) {
  const lines = wrapText(text, maxChars);
  return (
    <text x={x} y={y} fontSize={fontSize} fill={fill} textAnchor={textAnchor}>
      {lines.map((line, i) => (
        <tspan key={i} x={x} dy={i === 0 ? 0 : lineH}>{line}</tspan>
      ))}
    </text>
  );
}

function PublicConsultation() {
  const [vIdx, setVIdx] = useState(0);
  const [rated, setRated] = useState<number | null>(null);
  const v = vignettes[vIdx % vignettes.length];

  const handleRate = (val: number) => {
    setRated(val);
    setTimeout(() => {
      setVIdx((i) => i + 1);
      setRated(null);
    }, 800);
  };

  return (
    <div class="consultation">
      <svg viewBox="0 0 300 400" style={{ width: '100%' }}>
        <text x="35" y="20" fontSize="28" textAnchor="middle">{v.hat}</text>
        <text x="80" y="20" fontSize="12" fill="#0f3460" fontWeight="700">{v.role}</text>

        <rect x="5" y="32" width="290" height="3" rx="1.5" fill="#e94560" />
        <rect x="5" y="32" width="290" height="150" rx="8" fill="#e94560" opacity="0.06" />
        <rect x="5" y="32" width="290" height="3" rx="1.5" fill="#e94560" />
        <WrappedText x="15" y="52" text={narrative(v.role, v.context, v.decision)} maxChars={40} fontSize={10} fill="var(--color-text)" lineH={16} />

        <text x="150" y="205" textAnchor="middle" fontSize="13" fill="#6b7280" fontWeight="600">How much do you agree with this action?</text>

        {[1, 2, 3, 4, 5, 6, 7].map((val) => (
          <g key={val}>
            <rect x={-9 + val * 36} y="230" width="30" height="30" rx="8" fill={rated !== null && rated >= val ? '#0f3460' : '#f3f4f6'} stroke={rated !== null && rated >= val ? '#0f3460' : '#ccc'} strokeWidth="1.5" style={{ cursor: 'pointer' }} onClick={() => rated === null && handleRate(val)} />
            <text x={6 + val * 36} y="250" textAnchor="middle" fontSize="11" fill={rated !== null && rated >= val ? 'white' : '#6b7280'}>{val}</text>
          </g>
        ))}
        <text x="14" y="300" fontSize="11" fill="#999">Strongly disagree</text>
        <text x="286" y="300" textAnchor="end" fontSize="11" fill="#999">Strongly agree</text>

        <text x="150" y="340" textAnchor="middle" fontSize="10" fill="#999">Vignette {vIdx + 1} of {vignettes.length}</text>
        <rect x="70" y="352" width="160" height="4" rx="2" fill="#e5e7eb" />
        <rect x="70" y="352" width={160 * ((vIdx + 1) / vignettes.length)} height="4" rx="2" fill="#0f3460" />
      </svg>
    </div>
  );
}

export default function PipelineAnimation() {
  return (
    <div class="methodology">
      <Scene id="scene-survey" graphic={<SurveyPreview />}>
        <h3 class="methodology__step-title">1. The World Values Survey</h3>
        <p>Wave 7 of the World Values Survey captures responses from <strong><AnimatedCounter end={1057} /></strong> New Zealanders across <strong><AnimatedCounter end={251} /></strong> item-level questions — from family values to political trust.</p>
        <p>Each question has a fixed set of response options (Likert scale, multiple choice, etc.). Click through some examples on the left to see the kinds of questions respondents answered.</p>
      </Scene>

      <Scene id="scene-cluster" graphic={<ClusterViz />}>
        <h3 class="methodology__step-title">2. Finding Value Clusters</h3>
        <p>Using Latent Class Analysis, we partition respondents into subpopulations that share similar value patterns. The model identified <strong>2 clusters</strong> — one larger group (54%) and one smaller (46%).</p>
        <p>These clusters differ meaningfully. For example, on immigration:</p>
        <div style={{ marginTop: '1.5rem' }}>
          <BarChart data={clusterDiff} title={'Immigration strengthens cultural diversity'} />
        </div>
        <p style={{ fontSize: '0.8rem', color: 'var(--color-muted)', marginTop: '0.5rem' }}>
          Cluster 0 overwhelmingly agrees. Cluster 1 is more divided.
        </p>
      </Scene>

      <Scene id="scene-train" graphic={<TrainingLoop />} wide>
        <h3 class="methodology__step-title">3. Fine-tuning the Model</h3>
        <p>We take an open-weight LLM and fine-tune it on a specific cluster's response patterns. Two approaches are used:</p>
        <p><strong>Response-based</strong> — the model is shown a question and generates a response, which is compared to the expected answer. The loss is computed from the difference, and the weights are updated.</p>
        <p><strong>Distributional</strong> — the model outputs a full probability distribution over answer options, which is compared directly to the cluster's empirical distribution. The loss is used to update the model.</p>
        <p style={{ marginTop: '1rem' }}>
          <a href="/results-viewer?tab=evals" class="btn btn-outline" style={{ fontSize: '0.8rem', padding: '0.4rem 1rem' }}>View fine-tuning results →</a>
        </p>
      </Scene>

      <Scene id="scene-simulate" graphic={<SimulationViz />} wide>
        <h3 class="methodology__step-title">4. Real-world Simulation</h3>
        <p>We put the fine-tuned model through <strong>multi-step simulations</strong>. The model is placed in an Agentic harnesses and put in a 'deployment simulation'. The scenarios are designed to be relatable to everyday life so anyone can judge whether the decision feels right.</p>
        <p>Each step tests whether the values embedded during fine-tuning actually guide behaviour. For example, a Shop assistant, bus driver, teacher, neighbour, librarian, customer support — the model must apply its value framework consistently across contexts.</p>
        <p style={{ marginTop: '1rem' }}>
          <a href="/results-viewer?tab=simulation" class="btn btn-outline" style={{ fontSize: '0.8rem', padding: '0.4rem 1rem' }}>View simulation results →</a>
        </p>
      </Scene>

      <Scene id="scene-consult" graphic={<PublicConsultation />} wide>
        <h3 class="methodology__step-title">5. Public Consultation</h3>
        <p>Finally, we bring in the <strong>NZ public</strong>. Participants are shown vignettes and asked to rate how much they agree on a <strong>1–7 scale</strong>.</p>
        <p>Click through the vignettes on the right to try the rating interface. This is the core of our evaluation — statistical similarity is useful, but only public consultation can tell us whether the model <em>actually</em> reflects what New Zealanders value.</p>
        <p style={{ fontSize: '0.85rem', color: 'var(--color-muted)' }}>
          The study is awaiting ethics approval. Check the Join Survey page if you'd like to participate.
        </p>
      </Scene>
    </div>
  );
}
