import { describe, expect, it } from "vitest";
import { buildTimeline, clipEndMs, DEFAULT_CONFIG, generatePlan, planCounts, SEALS } from "../src/protocol";

describe("generatePlan", () => {
  it("is deterministic for a seed and differs across seeds", () => {
    expect(generatePlan("abc")).toEqual(generatePlan("abc"));
    const a = generatePlan("abc").filter((p) => p.block === "seal").map((p) => p.label);
    const b = generatePlan("abd").filter((p) => p.block === "seal").map((p) => p.label);
    expect(a).not.toEqual(b);
  });

  it("has the expected counts per block", () => {
    const plan = generatePlan("x");
    expect(planCounts(plan)).toEqual({ seal: 13 * 3, combo: 3, free: 1 });
    expect(plan).toHaveLength(43);
    expect(plan.map((p) => p.index)).toEqual(plan.map((_, i) => i));
  });

  it("orders blocks seal -> combo -> free", () => {
    const blocks = generatePlan("x").map((p) => p.block);
    expect(blocks.slice(0, 39).every((b) => b === "seal")).toBe(true);
    expect(blocks.slice(39)).toEqual(["combo", "combo", "combo", "free"]);
  });

  it("each take contains every seal exactly once, and takes are shuffled differently", () => {
    const plan = generatePlan("seed-1").filter((p) => p.block === "seal");
    const takes = [1, 2, 3].map((t) => plan.filter((p) => p.take === t).map((p) => p.label!));
    for (const t of takes) expect([...t].sort()).toEqual([...SEALS].sort());
    expect(new Set(takes.map((t) => t.join())).size).toBe(3);
  });

  it("has unique filesystem-safe clip ids", () => {
    const ids = generatePlan("x").map((p) => p.clip_id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const id of ids) expect(id).toMatch(/^[A-Za-z0-9_-]{4,64}$/);
  });

  it("respects a custom config", () => {
    const cfg = { ...DEFAULT_CONFIG, sealTakes: 1, combos: [["tiger", "horse"] as ["tiger", "horse"]] };
    expect(planCounts(generatePlan("x", cfg))).toEqual({ seal: 13, combo: 1, free: 1 });
  });
});

describe("buildTimeline", () => {
  const plan = generatePlan("x");
  const marksOf = (steps: ReturnType<typeof buildTimeline>) =>
    steps.flatMap((s) => s.marks.map((m) => `${m.segment}:${m.name}@${s.at}`));

  it("seal clip: 3-2-1 then form 2 s, hold 3 s, release 1.5 s", () => {
    const steps = buildTimeline(plan[0]);
    expect(steps.filter((s) => s.phase === "countdown").map((s) => [s.n, s.at])).toEqual([[3, 0], [2, 1000], [1, 2000]]);
    expect(marksOf(steps)).toEqual(["0:form_start@3000", "0:hold_start@5000", "0:hold_end@8000", "0:release_end@9500"]);
    expect(clipEndMs(steps)).toBe(9500 + DEFAULT_CONFIG.tailMs);
  });

  it("combo clip: per-seal sub-prompts, release_end of seal i coincides with form_start of i+1", () => {
    const combo = plan.find((p) => p.block === "combo")!;
    const m = marksOf(buildTimeline(combo));
    expect(m).toEqual([
      "0:form_start@3000", "0:hold_start@4500", "0:hold_end@6000",
      "0:release_end@6500", "1:form_start@6500", "1:hold_start@8000", "1:hold_end@9500", "1:release_end@10000",
    ]);
  });

  it("free clip lasts 2 minutes and cycles suggestions", () => {
    const free = plan[plan.length - 1];
    const steps = buildTimeline(free);
    const end = steps[steps.length - 1];
    expect(end.phase).toBe("end");
    expect(end.at - 3000).toBe(120_000);
    const texts = steps.filter((s) => s.phase === "free").map((s) => s.text);
    expect(texts[0]).toBe(DEFAULT_CONFIG.freeSuggestions[0]);
    expect(new Set(texts).size).toBe(DEFAULT_CONFIG.freeSuggestions.length);
  });
});
