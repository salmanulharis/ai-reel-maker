// Contract test: specs written by the Python side (tests/fixtures/render_spec_*.json, made by
// tests/test_compose.py) are accepted by the composition's schema. Run: npm test (Node 22.6+).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { durationInFrames, parseSpec } from "../src/spec.ts";

const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "tests", "fixtures");

test("Python render specs pass the composition's schema", () => {
  const files = readdirSync(fixtures).filter((name) => name.startsWith("render_spec_") && name.endsWith(".json"));
  assert.ok(files.length > 0, "no render spec fixtures; run python -m pytest tests/test_compose.py first");
  for (const name of files) {
    const spec = parseSpec(JSON.parse(readFileSync(join(fixtures, name), "utf8")));
    assert.ok(durationInFrames(spec) >= 1, name);
  }
});

test("a broken spec is refused with the field named", () => {
  assert.throws(() => parseSpec({ width: 10 }), /Invalid render spec/);
});
