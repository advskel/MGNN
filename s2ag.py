import random

from semanticscholar import SemanticScholar
from tqdm import tqdm
from sortedcontainers import SortedSet
from incidence_graph import IncidenceGraph

def generate(MAX_SIZE=10000, MIN_CITATIONS=5, MAX_COLLABS=10,
             P=1.0, API=None):
    sch = SemanticScholar(api_key=API)
    G = IncidenceGraph()
    mapping = dict()  # author ID to vertex ID
    visited = set()  # visited papers and authors
    queue = SortedSet(key=lambda x: x[0])

    # good starts:
    # 1717349 Donald Knuth

    queue.add(('1717349', 'author', -1))
    progress = tqdm(total=MAX_SIZE, desc='Populating graph (papers: 0, authors: 0, queue: 0)')
    #queue.add(('c16a634ff3ea273ba4b8a606007520c47be718a0', 'paper', 119))
    papers = 0
    while queue and len(G) < MAX_SIZE:
        id, type, count = queue.pop()
        progress.set_description(f'Populating graph (papers: {papers}, authors: {G.size(0)}, queue: {len(queue)})')
        if id in visited:
            continue
        visited.add(id)

        if type == 'author':
            obj = sch.get_author_papers(id, fields=['paperId', 'citationCount'], limit=1000)
            for paper in obj.items:
                if paper.paperId is None or paper.citationCount is None:
                    continue
                if paper.citationCount > MIN_CITATIONS and paper.paperId not in visited:
                    queue.add((paper.paperId, 'paper', paper.citationCount))
        elif type == 'paper':
            obj = sch.get_paper_authors(id, fields=['authorId'], limit=1000)

            if len(obj.items) > MAX_COLLABS:
                for authors in obj.items:
                    if authors.authorId is None:
                        continue
                    if authors.authorId not in visited:
                        queue.add((authors.authorId, 'author', -1))
                continue

            vertices = []
            not_mapped = []
            for authors in obj.items:
                if authors.authorId is None:
                    continue
                if authors.authorId not in mapping:
                    not_mapped.append(authors.authorId)
                else:
                    vertices.append(mapping[authors.authorId])
                if authors.authorId not in visited:
                    queue.add((authors.authorId, 'author', -1))
            if len(not_mapped) > 0 and random.random() > P:
                continue

            for author in not_mapped:
                mapping[author] = len(mapping)
                vertices.append(mapping[author])
            papers += 1

            if vertices not in G:
                G.put_simplex(vertices)
            progress.update(len(G) - progress.last_print_n)
            G.propagate_data(vertices=vertices, data=count, neighbor_dists=[], rel_dims=range(-len(vertices)+1, 1), update=lambda x, y: x + y)
    progress.close()

    # for author, vertex in tqdm(mapping.items(), desc='Sanity check'):
    #     obj = sch.get_author(author, fields=['citationCount'])
    #     if obj.citationCount < G.get(vertex):
    #         print(f'Author {author} has {obj.citationCount} citations, but graph has {G.get(vertex)} citations')
    #     #assert obj.citationCount >= G.get(vertex)

    return G