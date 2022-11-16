### LINMA2472 ALGORITHMS IN DATASCIENCE
### PROJECT 1

import re
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sklearn.cluster
import time
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import operator
from sklearn.linear_model import LinearRegression

import community  # pip install python-louvain (https://github.com/taynaud/python-louvain)

## STEP 1 PARSE .txt FILE TO EXCTRACT CO-OCCURENCE MATRIX OF CHARACTER IN EACH SCENE

file1_loc = "HarryPotterChamberOfSecrets.txt"

text1_brut = []  # Text au format brut (toutes les lignes non vides)
with open(file1_loc, "r") as f:
    for line in f:
        line = line.replace("/", " \ne ")
        line = line.strip()
        if line != '' and line != '\n':
            text1_brut.append(line)

CharScene = []  # Text avec scene et characters
for line in text1_brut:
    line = re.sub("([0-9])+([A-Z])?", "", line)
    line = line.strip()
    if "INT." in line or "EXT." in line:
        line = "SCENE"
    line = line.replace("CONTINUED", "")
    line = line.replace("OMITTED", "")
    line = line.replace("(CONT'D)", "")
    line = line.replace(" (O.S.)", "  ")
    line = line.replace("JUSTIN-FINCH-FLETCHLEY", "JUSTIN FINCH-FLETCHLEY")
    line = line.replace("DUMBLEDORE", "ALBUS DUMBLEDORE")
    line = line.replace("LUCIOUS MALFOY", "LUCIUS MALFOY")
    line = line.replace("TOM", "TOM RIDDLE")
    line = line.replace("RIDDLE", "TOM RIDDLE")
    line = line.replace("!", "")
    if (":") in line or (".") in line or ('(') in line:
        line = ""
    elif line.isupper() and len(line) > 2:
        CharScene.append(line.strip())

errors = ["THUD", "UP", "LEAVE HIM", "THE END", "WATCH OUT", "STOP", "HOME", "FLASH", "TURN", "THE WINDOWS"]

CharScene_clean = []
for i in CharScene:
    if i not in errors:
        CharScene_clean.append(i)

characters1 = []
vector = []
for i in CharScene_clean:
    if len(i.split()) <= 2 and i != "SCENE":
        characters1.append(i)
        vector.append(i)
    if i == "SCENE":
        vector.append(i)

# print("# Characters1: " + str(len(set(characters1))))  # error (4/40) : { "LAND HO", "UNDERWATER," , "THE END" , "FADE IN:" }
# print("# Scene1: " + str(CharScene.count("SCENE")))  # error in scene: {LORD CUTLER BECKETT}

all_char = []
scene_char = []
for i in range(len(vector)):
    if vector[i] == "SCENE":
        scene_char = []
        all_char.append(scene_char)

    else:
        scene_char.append(vector[i])
all_char.append(scene_char)

# print(all_char)


### CO-OCCURENCE MATRIX  ###
all_char_set = []
for i in range(len(all_char)):
    if all_char[i] == []:
        pass
    else:
        all_char_set.append(list(set(all_char[i])))

# print(all_char_set)
sorted_char = sorted(list(set(characters1)))

df = pd.DataFrame(np.zeros((len(sorted_char), len(sorted_char))), columns=sorted_char, index=sorted_char)

for sc in all_char:  # On peut remplacer par all_char
    if len(sc) < 2:
        pass
    for i in range(len(sc) - 1):
        for j in range(1, len(sc)):
            a = sorted_char.index(sc[i])
            b = sorted_char.index(sc[j])
            df.iloc[a, b] += 1
            df.iloc[b, a] += 1
node_size = []
for i in range(len(df)):
    # df.iloc[i,i]= vector.count(sorted_char[i])
    df.iloc[i, i] = 0
    node_size.append(vector.count(sorted_char[i]))

##### PART 2 #####

G = nx.from_pandas_adjacency(df)

"""###--- Visualisation ---### OK
fgz = (20, 20)
pos = nx.spring_layout(G, seed=2472, k=3)

plt.figure(figsize=fgz)

weight_of_edges = list(nx.get_edge_attributes(G, "weight").values())

nx.draw_networkx_nodes(G, pos, node_size=node_size)
nx.draw_networkx_edges(G, pos, alpha=0.1,)
nx.draw_networkx_labels(G, pos)
plt.title("Graph visualisation")
plt.tight_layout()
plt.show()
print(G)

###--- Degree associativity ---### OK
degree = nx.degree_assortativity_coefficient(G)
print("Degree of assortativity: ", round(degree, 3))

###--- Louvain algorithm ---### OK
# Make partition
partition = community.best_partition(G, randomize=False)  # Dictionnaire
n_indiv = len(partition)  # Nbr d'individus
communities_colors = ["#F78181", "#F7D358", "#81F781", "#819FF7", "#BE81F7", "#9D9D9D", "#B676B1"  ]
communities_shape = "opdhsv*>"
# Draw graph with partition
plt.figure(figsize=fgz)
for i in range(max(partition.values()) + 1):
    individus = 0 # number of individuals in the community
    node_list = [] # Character we will plot from that community
    for char, commu in partition.items():
        if commu == i:
            individus += 1
            node_list.append(char)
    nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                           cmap=plt.get_cmap('rainbow'),
                           label=individus,
                           node_color=communities_colors[i],
                           node_shape=communities_shape[i])

nx.draw_networkx_edges(G, pos, alpha=0.1)
nx.draw_networkx_labels(G, pos)
plt.legend(title="# Nodes", prop={'size': 16})
plt.title("Louvain Algorithm")
plt.show()

###--- Spectral clustering on the Laplacian ---###  BOF
plt.figure(figsize=fgz)
nx.draw(G, pos=nx.spectral_layout(G, dim = 3, weight = None, center = [10,10,10]), with_labels=False, edgelist=[], node_size=10)
plt.title("Spectral decomposition")
plt.show()

# in the 3-dimensional embedding space cluster the nodes in 4 communities with K-Means, then deduce communities
plt.figure(figsize=fgz)
spectral_coords = nx.spectral_layout(G, dim=3)
partitions = sklearn.cluster.KMeans(n_clusters=5).fit_predict([spectral_coords[node] for node in G.nodes()])
#nx.draw(G, with_labels=True, node_color = partitions, alpha=0.3)

nx.draw_networkx_nodes(G, pos, node_color = partitions, alpha=0.5)
nx.draw_networkx_edges(G, pos, alpha=0.1)
nx.draw_networkx_labels(G, pos)
plt.title("K-means community on spectral decomposition")
plt.show()"""


# Independent Cascade Model
def ICM(Gi, A0, p=0.25, n_iter=50, plot=False):
    res = []
    x = []  # time
    y = []  # n_infected

    for i in range(n_iter):
        n_infected = [len(A0)]
        G = Gi.copy()
        cascade = [A0]
        observation = A0.copy()

        t0 = time.time()
        x.append(t0 - t0);
        y.append(len(A0))  # Point de départ
        while len(observation) > 0:
            for infected in observation:
                period = []
                lost_edges = []
                for attempt in G.neighbors(infected):
                    if any(attempt in sublist for sublist in cascade):
                        pass
                    else:
                        if np.random.random(1)[0] < p:
                            period.append(attempt)
                            observation.append(attempt)
                        else:
                            lost_edges.append(attempt)
                for i in lost_edges:
                    G.remove_edge(i, infected)
                if period != []: cascade.append(period)
                n_infected.append(len(period))
                x.append(time.time() - t0)
                y.append(sum(n_infected))
                observation.remove(infected)
        res.append(sum(n_infected))
    if plot:
        p2 = np.poly1d(np.polyfit(x, y, 3))
        xp = np.linspace(min(x), max(x), 100)
        plt.plot(xp, p2(xp), c='r')
        plt.scatter(x, y, alpha=0.1, s=5)
        plt.show()
        return (np.mean(res), p2)

    return np.mean(res)


# Simulate the spread process using Independent Cascade Model over the graph
# print(ICM(G, ["HERMIONE"]))

def greedy_algorithm(Gi, k=0.05, p=0.25, n_iter=500):
    A0 = []
    nA0 = int(len(Gi) * k)
    max_score_log = []  # max score real time
    max_time_log = []  # max time real time
    score_log = []  # Log of the score
    time_log = []  # log of the time
    t0 = time.time()
    while len(A0) < nA0:
        max_score = 0
        max_node = ""
        for node in (set(Gi.nodes()) - set(A0)):
            node_set = A0 + [node]
            node_score = ICM(Gi, node_set, p, n_iter)
            if node_score > max_score:
                max_node = node
                max_score = node_score
            max_score_log.append(max_score)
            max_time_log.append(time.time() - t0)
            time_log.append(time.time() - t0)
            score_log.append(node_score)
        A0.append(max_node)
    max_score_log.append(max_score)
    max_time_log.append(time.time() - t0)

    plt.figure(figsize=(8, 8))
    plt.scatter(time_log, score_log, alpha=0.3)
    plt.plot(max_time_log, max_score_log, c='red', alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('Influenced nodes (total nodes = {0})'.format(len(Gi)))
    plt.title('Spreadind curve [ p = {0} ; k = {1} ; n_iter = {2}]'.format(p, k, n_iter))
    plt.suptitle("Maximal influence nodes: {0}".format(A0))
    plt.show()

    return (A0, max_score_log[-1])


# r = greedy_algorithm(G)

## Comparaison:

# A0 = résultats greedy
# ICM(G, greedy_algorithm(G), plot=True)

# A0 = les 2 + grands degree
# A0 = list(np.array(sorted(G.degree, key=lambda x: x[1], reverse=True)[:2])[:,0])
# ICM(G, A0, plot=True)


### Barabasi albert grap
n = len(G.nodes())
m = G.number_of_edges()/n
B = nx.barabasi_albert_graph(n, int(m))
print(greedy_algorithm(B))

## COMPARAISON DE 2 GRAPHES:
# Nodes
# Degree
# Degree_assortativity_coefficient(G)