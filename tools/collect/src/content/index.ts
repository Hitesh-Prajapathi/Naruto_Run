import consentText from "./consent.md?raw";
import sealsJson from "./seals.json";
import { fnv1a } from "../rng";

export const CONSENT_TEXT: string = consentText;
export const CONSENT_HASH = fnv1a(consentText);

export interface SealInfo {
  description: string;
  confusable_with: string[];
  status: string;
}
export const SEAL_INFO = sealsJson.seals as Record<string, SealInfo>;

/** Minimal markdown -> HTML (headings, bullets, **bold**, paragraphs). Input is our own file; still escaped. */
export function renderMarkdown(md: string): string {
  const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = (s: string) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  const out: string[] = [];
  let list = false;
  for (const line of md.split("\n")) {
    const li = /^- (.*)/.exec(line);
    if (li) {
      if (!list) out.push("<ul>"), (list = true);
      out.push(`<li>${inline(li[1])}</li>`);
      continue;
    }
    if (list) out.push("</ul>"), (list = false);
    const h = /^(#{1,3}) (.*)/.exec(line);
    if (h) out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`);
    else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push("</ul>");
  return out.join("\n");
}
