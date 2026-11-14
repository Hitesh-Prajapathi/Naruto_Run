/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_TRANSPORT_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
