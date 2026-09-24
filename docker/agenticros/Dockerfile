# AgenticROS MCP sidecar for the G1 Isaac ROS graph.
# The ROS bridge itself runs in the sibling rosbridge container; this image
# provides AgenticROS' stdio MCP server for Codex/Claude/Hermes clients.
FROM node:20-bookworm-slim

ARG AGENTICROS_REF=main
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && corepack enable \
    && corepack prepare pnpm@9.15.4 --activate

RUN git clone --depth 1 --branch "${AGENTICROS_REF}" \
    https://github.com/agenticros/agenticros.git /src
WORKDIR /src
RUN pnpm install --frozen-lockfile \
    && pnpm build \
    && pnpm --filter @agenticros/agenticros-claude-code deploy --prod /opt/agenticros-mcp

COPY agenticros-config.json /root/.agenticros/config.json
RUN mkdir -p /root/.agenticros

ENTRYPOINT ["node", "/opt/agenticros-mcp/dist/index.js"]
