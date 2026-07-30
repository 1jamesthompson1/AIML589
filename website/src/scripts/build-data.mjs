import { readFileSync, writeFileSync, existsSync, readdirSync, statSync } from 'fs';
import { join } from 'path';
import papa from 'papaparse';

const EVALS_DIR = new URL('../../../code/fine-tuning/output/evals', import.meta.url).pathname;
const OUT_DIR = new URL('../data/', import.meta.url).pathname;
const OUT_FILE = join(OUT_DIR, 'evals.json');

function readJSON(p) {
  return JSON.parse(readFileSync(p, 'utf-8'));
}

function readCSV(p) {
  const text = readFileSync(p, 'utf-8');
  const parsed = papa.parse(text, { header: true, dynamicTyping: false });
  return parsed.data;
}

function pyListToJSON(s) {
  if (typeof s !== 'string' || !s.startsWith('[')) return s;
  try { JSON.parse(s); return s; } catch {}
  const items = s.slice(1, -1).match(/'[^']*'|"[^"]*"/g);
  if (!items) return s;
  const parsed = items.map((item) => {
    const content = item.slice(1, -1).replace(/"/g, '\\"');
    return `"${content}"`;
  });
  return '[' + parsed.join(', ') + ']';
}

function trimResult(row, runName) {
  let subpop = row.subpopulation;
  if (!subpop) {
    const parts = runName.split('-');
    if (parts.length >= 3 && ['overall', 'cluster_0', 'cluster_1'].includes(parts[1])) {
      subpop = parts[1];
    } else {
      subpop = 'unknown';
    }
  }
  return {
    question_id: row.question_id,
    question: row.question,
    sub_question: row.sub_question || '',
    column_name: row.column_name,
    question_format: row.question_format,
    system_prompt_id: row.system_prompt_id,
    subpopulation: subpop,
    model_answer: row.model_answer?.slice(0, 200) || '',
    model_reasoning: '', // trimmed to save space
    categories: pyListToJSON(row.categories || ''),
    model_distribution: pyListToJSON(row.model_distribution || ''),
    true_distribution: pyListToJSON(row.true_distribution || ''),
    kl_divergence: row.kl_divergence || '',
    cross_entropy: row.cross_entropy || '',
    expected_text: row.expected_text?.slice(0, 200) || '',
  };
}

function buildEvals() {
  const result = { models: {} };

  if (!existsSync(EVALS_DIR)) {
    console.warn('Warning: evals directory not found at', EVALS_DIR);
    return result;
  }

  const modelDirs = readdirSync(EVALS_DIR).filter((d) =>
    statSync(join(EVALS_DIR, d)).isDirectory()
  );

  for (const modelName of modelDirs) {
    const modelDir = join(EVALS_DIR, modelName);
    const runDirs = readdirSync(modelDir).filter((d) =>
      statSync(join(modelDir, d)).isDirectory()
    );

    const runs = [];

    for (const runName of runDirs) {
      const configPath = join(modelDir, runName, 'config.json');
      const csvPath = join(modelDir, runName, 'per_question_results.csv');

      if (!existsSync(configPath) || !existsSync(csvPath)) continue;

      const config = readJSON(configPath);
      const csvRows = readCSV(csvPath);

      const results = csvRows
        .filter((r) => r.question_id)
        .map((r) => trimResult(r, runName));

      runs.push({
        config: {
          target: config.target,
          dataset: config.dataset,
          run_name: config.run_name,
          timestamp: config.timestamp,
          model_sha: config.model_sha,
        },
        results,
      });

      console.log(`  ${modelName}/${runName}: ${results.length} questions`);
    }

    if (runs.length > 0) {
      result.models[modelName] = runs;
    }
  }

  return result;
}

function main() {
  console.log('Building eval data...');
  const data = buildEvals();

  writeFileSync(OUT_FILE, JSON.stringify(data, null, 2));
  console.log(`Wrote ${OUT_FILE} (${Object.keys(data.models).length} models)`);
}

main();
