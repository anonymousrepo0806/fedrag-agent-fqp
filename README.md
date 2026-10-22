Federated query processing over per client knowledge graphs for federated clinical learning.

Files
fedrag_sim.py: cohort, knowledge graph, retrieval placements, models, federated algorithms, metrics (in process simulation).
fqp.py: federated query processing engine (Bloom filter source summaries, batched sub queries, bound join, decoys, replication).
collude.py: collusion simulation and probe baseline measurement.
fl_net.py: the same code split into services (one Docker container each).
hypo_learn.py, measurements.py, textgen.py, textnet.py, sparql_ext.py, agent_fqp.py, edge_controller.py, connection_module.py, cache_store.py: supporting modules.
Dockerfile, docker-compose.yml, scripts/*.sh: CPU only container setup and sweeps.

Data
MIMIC III is not included. Obtain it under its data use agreement.

