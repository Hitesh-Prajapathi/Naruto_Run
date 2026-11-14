import { describe, expect, it } from "vitest";
import { CONSENT_TEXT, renderMarkdown, SEAL_INFO } from "../src/content";
import { SEALS } from "../src/protocol";

describe("content", () => {
  it("seals.json covers every protocol seal; dragon is TBD; confusable pairs present", () => {
    expect(Object.keys(SEAL_INFO).sort()).toEqual([...SEALS].sort());
    expect(SEAL_INFO.dragon.description).toMatch(/TBD by project owner/);
    expect(SEAL_INFO.ram.confusable_with).toContain("rat");
    expect(SEAL_INFO.dog.confusable_with).toContain("hare");
    expect(SEAL_INFO.tiger.confusable_with).toContain("rat");
  });
  it("consent covers purpose, recording, no audio, retention, withdrawal", () => {
    for (const w of ["Purpose", "No audio", "Retention", "withdraw", "subject ID"]) expect(CONSENT_TEXT.toLowerCase()).toContain(w.toLowerCase());
  });
  it("markdown renderer escapes HTML", () => {
    expect(renderMarkdown("# a<b>\n- **x**")).toBe("<h1>a&lt;b&gt;</h1>\n<ul>\n<li><strong>x</strong></li>\n</ul>");
  });
});
