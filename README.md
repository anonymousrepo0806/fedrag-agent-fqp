Federated query processing over per client knowledge graphs for federated clinical learning.

Files
fedrag_sim.py: cohort, knowledge graph, retrieval placements, models, federated algorithms, metrics (in process simulation).
fqp.py: federated query processing engine (Bloom filter source summaries, batched sub queries, bound join, decoys, replication).
collude.py: collusion simulation and probe baseline measurement.
fl_net.py: the same code split into services (one Docker container each).
hypo_learn.py, measurements.py, textgen.py, textnet.py, sparql_ext.py, agent_fqp.py, edge_controller.py, connection_module.py, cache_store.py: supporting modules.
Dockerfile, docker-compose.yml, scripts/*.sh: CPU only container setup and sweeps.

Data
MIMIC III is not included. Obtain it under its data use agreement and point the commands to your copy.

Example
python fedrag_sim.py retrieval --cohort work/mortality_next/cohort.pkl --kg work/mortality_next/kg.pkl --clients 10 --seeds 1 2 3 --arms server client client_local fqp --fqp_repl 2 --fqp_hops 2 --fqp_join_top 20 --fqp_join_w 0.2 --depth 20 --k 10 20 --out results.json
