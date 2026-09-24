/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** OTLP/HTTP traces endpoint, e.g. http://localhost:4318/v1/traces. Unset = no browser span export. */
  readonly VITE_OTLP_TRACES_URL?: string;
  /** Show the dev footer (trace id, request id) in production builds too. */
  readonly VITE_SHOW_DEV_FOOTER?: string;
  /** Jaeger UI base for trace links in the dev footer. */
  readonly VITE_JAEGER_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
