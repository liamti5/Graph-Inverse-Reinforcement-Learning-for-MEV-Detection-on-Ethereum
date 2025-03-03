import networkx as nx
import typer

app = typer.Typer()


class Graph:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def compute_graph_metrics(self):
        simple_G = nx.DiGraph(self.graph)
        density = nx.density(simple_G)

        avg_degree = (
            sum(dict(simple_G.degree()).values()) / simple_G.number_of_nodes()
            if simple_G.number_of_nodes() > 0
            else 0
        )

        clustering_coeff = nx.average_clustering(simple_G.to_undirected())

        components = list(nx.weakly_connected_components(simple_G))
        largest_component_size = max((len(c) for c in components), default=0)
        num_isolated_nodes = len(list(nx.isolates(simple_G)))

        degree_distribution = {}
        for node, degree in simple_G.degree():
            degree_distribution[degree] = degree_distribution.get(degree, 0) + 1

        return {
            "size": simple_G.number_of_nodes(),
            "density": density,
            "average_degree": avg_degree,
            "clustering_coefficient": clustering_coeff,
            "largest_component_size": largest_component_size,
            "num_isolated_nodes": num_isolated_nodes,
            "degree_distribution": degree_distribution,
        }


@app.command()
def main():
    pass


if __name__ == "__main__":
    app()
