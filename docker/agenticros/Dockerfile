# AgenticROS MCP sidecar for the G1 Isaac ROS graph.
# The ROS bridge itself runs in the sibling rosbridge container; this image
# provides AgenticROS' stdio MCP server for Codex/Claude/Hermes clients.
FROM node:20-bookworm-slim

ARG AGENTICROS_REF=e44910f6a71968d3214ca8b1107b20b8dba21f42
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && corepack enable \
    && corepack prepare pnpm@9.15.4 --activate

RUN mkdir -p /src && cd /src && \
    git init -q && \
    git remote add origin https://github.com/agenticros/agenticros.git && \
    git fetch --depth 1 origin "${AGENTICROS_REF}" && \
    git checkout -q --detach FETCH_HEAD
WORKDIR /src
RUN pnpm install --frozen-lockfile --ignore-scripts \
    && pnpm build \
    && pnpm --filter @agenticros/claude-code deploy --prod /opt/agenticros-mcp

COPY agenticros-config.json /root/.agenticros/config.json
RUN mkdir -p /root/.agenticros

ENTRYPOINT ["node", "/opt/agenticros-mcp/dist/index.js"]
