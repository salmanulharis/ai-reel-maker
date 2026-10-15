// Checks the in-app help (src/docs/content.ts): structure, links, formatting, and search.
// Run with `npm run check:docs` (Node 22.6+ runs the TypeScript content file directly).
// Pass --labels to also list **bold** phrases that do not appear in the UI wording file, which
// usually means a help article names a button or setting that was renamed.
import { readFileSync } from "node:fs";
import { DOC_ARTICLES, DOC_CATEGORIES, searchDocs } from "../src/docs/content.ts";

const SETTINGS_SECTIONS = ["optimize", "ai", "speech", "video", "storage", "system", "performance", "connection"];
const problems = [];
const ids = Object.keys(DOC_ARTICLES);

const placed = DOC_CATEGORIES.flatMap((category) => category.articles);
for (const id of ids) {
  const count = placed.filter((item) => item === id).length;
  if (count !== 1) problems.push(`Article "${id}" is in ${count} categories (expected 1).`);
}

const texts = [];
for (const [id, article] of Object.entries(DOC_ARTICLES)) {
  if (!article.title || !article.summary || !article.blocks.length) problems.push(`Article "${id}" is incomplete.`);
  for (const block of article.blocks) {
    if (block.kind === "see" && !ids.includes(block.article)) problems.push(`"${id}" links to missing article "${block.article}".`);
    if (block.kind === "settings" && !SETTINGS_SECTIONS.includes(block.section)) problems.push(`"${id}" links to unknown settings section "${block.section}".`);
    const strings = [block.text, ...(block.items || []), ...(block.rows || []).flat()].filter((value) => typeof value === "string");
    for (const value of strings) {
      texts.push(value);
      if ((value.match(/\*\*/g) || []).length % 2) problems.push(`"${id}" has an unclosed **bold**: ${value.slice(0, 60)}`);
      if ((value.match(/`/g) || []).length % 2) problems.push(`"${id}" has an unclosed \`code\`: ${value.slice(0, 60)}`);
    }
  }
}

if (searchDocs("").length !== 0) problems.push("An empty search should return nothing.");
for (const [query, expected] of [["slow", "cpu-speed"], ["siglip", "setup-image-matching"], ["out of memory", "trouble-memory"]]) {
  if (!searchDocs(query).includes(expected)) problems.push(`Searching "${query}" does not find "${expected}".`);
}

if (process.argv.includes("--labels")) {
  const wording = readFileSync(new URL("../src/i18n/en.json", import.meta.url), "utf8");
  const bold = new Set(texts.flatMap((value) => [...value.matchAll(/\*\*([^*]+)\*\*/g)].map((m) => m[1])));
  const missing = [...bold].filter((phrase) => !wording.includes(phrase.split(" → ").pop()));
  console.log(`Bold phrases not found in en.json (${missing.length}):`);
  for (const phrase of missing) console.log(`  - ${phrase}`);
}

if (problems.length) {
  console.error(problems.map((line) => `✗ ${line}`).join("\n"));
  process.exit(1);
}
console.log(`✓ Help content OK: ${ids.length} articles in ${DOC_CATEGORIES.length} categories.`);
