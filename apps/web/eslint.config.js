// Minimal flat config. Pre-existing violations are downgraded to warnings
// rather than mass-edited; tighten rules to "error" as they are cleaned up.
import js from "@eslint/js";
import globals from "globals";
import prettier from "eslint-config-prettier";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/**", "node_modules/**", "coverage/**", "playwright-report/**", "test-results/**", "public/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["**/*.ts"],
    languageOptions: { globals: { ...globals.browser } },
    rules: {
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-non-null-assertion": "off",
      "prefer-const": "warn", // 1 pre-existing hit (src/transport/mockEventSource.ts)
    },
  },
  {
    files: ["*.config.ts", "*.config.js"],
    languageOptions: { globals: { ...globals.node } },
  },
  prettier,
);
