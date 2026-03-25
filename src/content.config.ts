import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const llmInference = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/llm-inference' }),
  schema: z.object({
    title: z.string(),
    weight: z.number().optional(),
    description: z.string().optional(),
  }),
});

export const collections = {
  'llm-inference': llmInference,
};
