# syntax=docker/dockerfile:1
# Build the static bundle, then serve it with nginx, which also proxies /api to the broker.
#   docker build -t hybridsearch-web web/
#   docker run -p 8081:80 -e BROKER_UPSTREAM=http://broker:8080 hybridsearch-web

FROM node:24-alpine AS build
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@9.0.0 --activate
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
# Optional build-time flags (see src/vite-env.d.ts)
ARG VITE_OTLP_TRACES_URL=""
ARG VITE_SHOW_DEV_FOOTER=""
ENV VITE_OTLP_TRACES_URL=$VITE_OTLP_TRACES_URL VITE_SHOW_DEV_FOOTER=$VITE_SHOW_DEV_FOOTER
RUN pnpm build

FROM nginx:1.27-alpine
ENV BROKER_UPSTREAM=http://broker:8080
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
