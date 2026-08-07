The following event assertions are provided in scrambled, non-chronological order. Reconstruct the true causal DAG (Directed Acyclic Graph) based solely on semantic dependencies, ignoring the sequence in which these statements appear below.

Assertions:
[A] System B emits a `503 Service Unavailable` due to thread pool exhaustion.
[B] Database connection pool size is decreased from 50 to 5 in configuration commit `c8f1a`.
[C] Batch analytics job `cron_midnight` triggers 100 concurrent worker threads.
[D] Deployment pipeline deploys commit `c8f1a` to production cluster west-2.
[E] Incoming HTTP requests queue up on API Gateway instance `gw-01`.
[F] API Gateway `gw-01` drops inbound client connections with TCP resets.
[G] Developer merges PR #402 containing connection pool changes.

Tasks:
1. Re-order the events into a strict linear sequence from initial root cause to final failure state.
2. Output a ASCII dependency graph (`Node A -> Node B`).
3. Write a root-cause incident response report identifying the exact commit and action that initiated the cascading failure, accompanied by recommendations.

Keep the output concise and strictly under 1,200 tokens.
