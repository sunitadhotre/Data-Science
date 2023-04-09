import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edge('A', 'B') ;
G.add_edge('A', 'C') ;
G.add_edge('B', 'D' );
G.add_edge('B', 'E') ;
G.add_edge('D', 'E') ;
nx.draw_networkx(G)


fb = nx.read_edgelist("D:\\SSD_Python01012023\\facebook_combined.txt")
fb_n , fb_k = fb.order(), fb.size()
fb_avg_deg = fb_k / fb_n
print ('Nodes: ', fb_n)
print ('Edges: ', fb_k)
print ('Average degree: ', fb_avg_deg
)

print ('# connected components of Facebook network: ', nx.number_connected_components(fb))

fb_prunned = nx.read_edgelist("D:\\SSD_Python01012023\\facebook_combined.txt")
fb_prunned.remove_node('0')
print ('Remaining nodes:', fb_prunned.number_of_nodes())
print ('New # connected components:', nx.number_connected_components(fb_prunned))

fb_components = nx.connected_components(fb_prunned)
print ('Sizes of the connected components', [len(c) for c in fb_components])

# Centrality measures for the star graph:
degree = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
print ('Degree centrality: ', sorted(degree.items(), key = lambda x: x[1], reverse = True))
print ('Betweenness centrality: ', sorted(betweenness.items(), key = lambda x: x[1], reverse = True))


degree_cent_fb = nx.degree_centrality(fb)
print ('Facebook degree centrality: ', sorted (degree_cent_fb. items(), key = lambda x: x[1], reverse = True)[:10])
degree_hist = plt. hist( list (degree_cent_fb. values()), 100)
plt.loglog (degree_hist [1][1:], degree_hist[0], 'b', marker = 'o')


betweenness_fb = nx. betweenness_centrality(fb)
closeness_fb = nx. closeness_centrality(fb)
eigencentrality_fb = nx.eigenvector_centrality(fb)
print ('Facebook betweenness centrality:', sorted (betweenness_fb. items(), key = lambda x: x[1], reverse = True)[:10])
print ('Facebook closeness centrality:', sorted (closeness_fb. items(), key = lambda x: x[1],reverse = True)[:10])
print ('Facebook eigenvector centrality:', sorted (eigencentrality_fb. items(), key = lambda x: x[1], reverse = True)[:10]
)

fig = plt.figure(figsize = (6,5))

degree_hist = plt.hist(list(degree_cent_fb.values()), 100)
plt.xlabel('Degree centrality')
plt.ylabel('Number of nodes')
plt.title('Degree centrality histogram')
plt.savefig("D:\\SSD_Python01012023\\degree_centrality_hist.png", dpi = 300, bbox_inches = 'tight')

plt.show()

fig = plt.figure(figsize = (6,5))
degree_hist = plt.hist(list(degree_cent_fb.values()), 100)
plt.loglog(degree_hist[1][1:], degree_hist[0], 'b', marker = 'o')
plt.ylabel('Number of nodes (log)')
plt.xlabel('Degree centrality (log)')
plt.title('Sorted nodes degree (loglog)')
plt.savefig("D:\\SSD_Python01012023\\degree_centrality_hist_log.png", dpi = 300, bbox_inches = 'tight')
plt.show()

betweenness_fb = nx.betweenness_centrality(fb)
closeness_fb = nx.closeness_centrality(fb)
eigencentrality_fb = nx.eigenvector_centrality(fb)
print ('Facebook betweenness centrality:', sorted(betweenness_fb.items(), key = lambda x: x[1], reverse = True)[:10])
print ('Facebook closeness centrality:', sorted(closeness_fb.items(), key = lambda x: x[1], reverse = True)[:10])
print ('Facebook eigenvector centrality:', sorted(eigencentrality_fb.items(), key = lambda x: x[1], reverse = True)[:10])
plt.show()

def trim_degree_centrality(graph, degree = 0.01):
    g = graph.copy()
    d = nx.degree_centrality(g)
    for n in g.nodes():
        if d[n] <= degree:
            g.remove_node(n)
    return g

thr = 21.0/(fb.order() - 1.0)
print ('Degree centrality threshold:', thr)

fb_trimed = trim_degree_centrality (fb , degree = thr)
print ('Remaining # nodes:', len (fb_trimed))
