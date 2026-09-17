import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

// Agent reference docs, generated from the profile modules' docstrings and
// situations.json by workbench/agent-docs/generate_docs_site.py.
const agents = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/agents' }),
  schema: z.object({
    title: z.string(),
    id: z.string(),
    organisation: z.string(),
    summary: z.string(),
  }),
});

export const collections = { agents };
