import time
import typing

import bittensor as bt
import networkx as nx
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from common.base.miner import BaseMinerNeuron


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.axon.attach(
            forward_fn=self.forward_graph,
            blacklist_fn=self.backlist_graph,
            priority_fn=self.priority_graph,
        )

    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        number_of_nodes = synapse.number_of_nodes
        adjacency_list = synapse.adjacency_list
        dict_of_lists = {i: adjacency_list[i] for i in range(number_of_nodes)}
        graph = nx.from_dict_of_lists(dict_of_lists)
        # Convert NetworkX graph to our solver format
        n = number_of_nodes
        edges = []
        for u in range(n):
            for v in adjacency_list[u]:
                if u < v:  # Avoid duplicate edges for undirected graph
                    edges.append((u, v))
        
        # Import and use our exact solver
        from CliqueAI.solver.max_clique_solver import solve_max_clique_all
        
        # Run our exact maximum clique solver with a time budget
        start_time = time.time()
        result = solve_max_clique_all(n, edges, time_budget_sec=30)
        solve_time = time.time() - start_time
        
        bt.logging.info(f"Exact solver completed in {solve_time:.3f}s")
        bt.logging.info(f"Found maximum clique of size {result['omega']}")
        bt.logging.info(f"Solver was {'complete' if result['complete'] else 'incomplete (time limit)'}")
        
        # Use the exact solution if found, otherwise fall back to NetworkX approximation
        if result['witness']:
            maximum_clique = result['witness']
            bt.logging.info(f"Using exact solution: {len(maximum_clique)} {maximum_clique}")
        elif len(result['max_cliques']) > 0:
            maximum_clique = result['max_cliques'][0]
            bt.logging.info(f"Using extra solution: {len(maximum_clique)} {maximum_clique}")
        else:
            bt.logging.warning("Exact solver found no solution, falling back to NetworkX approximation")
            maximum_clique = nx.approximation.max_clique(graph)
            bt.logging.info(f"Using networkx solution: {len(maximum_clique)} {maximum_clique}")
        bt.logging.info(
            f"Maximum clique found: {maximum_clique} with size {len(maximum_clique)}"
        )
        synapse.adjacency_list = [[]]  # Clear up the adjacency list to reduce response size.
        synapse.maximum_clique = maximum_clique
        return synapse

    async def backlist_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_graph(self, synapse: MaximumCliqueOfLambdaGraph) -> float:
        return await self.priority(synapse)


if __name__ == "__main__":
    # # Synapse
    # def generate_sample_graph(n: int = 100, edge_count: int = 800):
    #     """Generate a random undirected graph with n vertices and edge_count edges."""
    #     max_edges = n * (n - 1) // 2
    #     if edge_count > max_edges:
    #         raise ValueError(f"Too many edges requested. Max possible for n={n} is {max_edges}.")
    #     all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    #     import random
    #     chosen = random.sample(all_edges, edge_count)
    #     return n, chosen
    # n, edges = generate_sample_graph(500, 100000)

    # import os
    # import json
    # try:
    #     sample_file_path = os.path.join(os.path.dirname(__file__), "miner/sample_v4_01.json")
    #     with open(sample_file_path, 'r') as f:
    #         sample_data = json.load(f)
        
    #     if "adjacency_list" in sample_data:
    #         adjacency_list = sample_data["adjacency_list"]
    #         n = len(adjacency_list)
            
    #         # Convert adjacency list to edge list
    #         edges = []
            
    #         print(f"Loaded sample graph from ../sample.js: {n} vertices, {len(edges)} edges")
    #         # return n, edges
    # except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
    #     print(f"Could not load sample data: {e}, falling back to random generation")
    
    # for u in range(n):
    #     for v in adjacency_list[u]:
    #         if u < v:  # Avoid duplicate edges for undirected graph
    #             edges.append((u, v))


    # synapse = MaximumCliqueOfLambdaGraph(
    #     uuid="",
    #     label="",
    #     number_of_nodes=n,
    #     adjacency_list=edges,
    #     timeout=30,
    # )
    

    with Miner() as miner:
        # import asyncio
        # asyncio.run(miner.forward_graph(synapse))
        # print("done")
        bt.logging.info("Miner has started running.")
        while True:
            if miner.should_exit:
                bt.logging.info("Miner is exiting.")
                break
            time.sleep(1)
