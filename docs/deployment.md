# Deployment (Docker Compose)

## Included Services
- MySQL
- Neo4j
- Qdrant
- Redis
- FastAPI

## Steps
1. Copy env template:
   cp .env.example .env
2. Fill API keys in .env:
   OPENAI_API_KEY=...
3. Build and start:
   docker compose up -d --build
4. Verify:
   - API: http://localhost:8000/health
   - Neo4j Browser: http://localhost:7474
   - Qdrant: http://localhost:6333

## Production Recommendations
- Use managed MySQL with read replicas.
- Enable Neo4j clustering for HA.
- Scale FastAPI replicas behind a load balancer.
- Put Redis in persistent mode (AOF/RDB).
- Externalize secrets into vault/KMS.
