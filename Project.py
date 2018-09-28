# <center><h2>Visualization Project</h2></center>

# Let's first see the official datasets

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances
from platypus import NSGAII, NSGAIII, Problem, Real
import sys
import datetime

n_dim = 0
font = {'family': 'serif', 
    'weight': 'normal', 
    'horizontalalignment': 'right',
    'size': 10, }
def graph(d1, d2, m, row, num, stress, inter):
    global n_dim
    global fig
    ax = fig.add_subplot(1, 4, num)
    x_val = d1["x1"].values
    y_val = d1["x2"].values
    link = d2.values.T
    plt.plot(x_val[link], y_val[link], 'g-', zorder=0) # Edges
    low1 = d1["x1"].min()
    low2 = d1["x2"].min()
    x_range = d1["x1"].max() - low1
    y_range = d1["x2"].max() - low2
    plt.axis([low1 - 0.2*x_range, d1["x1"].max() + 0.2*y_range, low2 - 0.2*x_range, d1["x2"].max() + y_range])

    t = np.arange(len(d1.index))
    text="Stress=%0.3f"%stress+", Inter="+str(inter)
    plt.scatter(d1["x1"], d1["x2"], edgecolors='none', cmap=plt.matplotlib.cm.jet, zorder=2, s=20)  # c=np.reshape(t,(-1, len(t))), 
    y_min,y_max = ax.get_ylim()
    x_min,x_max = ax.get_xlim()
    plt.xticks(np.arange(math.floor(x_min), math.ceil(x_max), math.ceil((math.ceil(x_max)-math.ceil(x_min))/5)))
    plt.yticks(np.arange(math.floor(y_min), math.ceil(y_max), math.ceil((math.ceil(y_max)-math.ceil(y_min))/5)))
    plt.text(x_max*0.85, y_max*0.8, r""+text, fontdict=font)
    for i, txt in enumerate(range(len(d1.index))):
        ax.annotate(txt, xy=(d1["x1"][i]+x_range/50+(pow(-1,i)*((i%len(d1.index))/10)), d1["x2"][i]+y_range/50+(pow(-1,i)*((i%len(d1.index))/10))), size=6)
    #plt.show()

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

def ccw(A,B,C):
    val = (C.y-A.y)*(B.x-A.x) - (B.y-A.y)*(C.x-A.x)
    if val == 0:
        return 0
    if val > 0:
        return 1
    return 2

def intersect(A,B,C,D):
    if A == C or B == C or A == D or B == D:
        return False
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def number_of_intersection(x,y,d2):
    pts = []
    for i in range(0,len(x)):
        pts.append(Point(x[i], y[i]))

    x1 = d2["node1"].values
    y1 = d2["node2"].values

    count = 0
    for i in range(0, len(x1)):
        for j in range(i+1, len(y1)):
            if intersect(pts[x1[i]], pts[y1[i]], pts[x1[j]], pts[y1[j]]):
                count = count + 1
    return count

def my_graph(points, edges):
    global n_dim
    global fig
    number = 0
    d1 = pd.read_csv(points)
    d2 = pd.read_csv(edges)
    n_dim = len(d1.columns)
    m1 = 'MDS'
    m2 = 'MOPSO-NSGAII'
    m3 = 'MOPSO-NSGAIII'

    # ## MDS
    mds = MDS(n_components=2)
    dat = pd.DataFrame(mds.fit(d1).embedding_,columns=["x1","x2"])

    dij = manhattan_distances(d1.values, d1.values)
    dis2 = manhattan_distances(dat.values, dat.values)
    mds_stress = (np.sum((dij-dis2)**2))**0.5   
    mds_inter = number_of_intersection(dat["x1"],dat["x2"], d2)
    graph(dat, d2, m1, number, 1,mds_stress,mds_inter)
    print "MDS Stress : ", mds_stress
    print "Total Intersection MDS = ", mds_inter
    
    # ## Multi Objective Particle Swarm Optimization (MOPSO)
    def objective_mopso(x):
        twoD = np.reshape(x, (-1, 2))
        dis = manhattan_distances(twoD, twoD)
        stress = (np.sum((dij-dis)**2))**0.5
        mopso_inter = number_of_intersection(x[0:][::2],x[1:][::2], d2)
        return [stress, mopso_inter]
    
    problem = Problem(len(d1)*2, 2)
    problem.types[:] = Real(-50, 50)
    problem.function = objective_mopso
    
    ngsaii = NSGAII(problem, population_size=100)
    ngsaii.run(10000)
    result = ngsaii.result
   
    ngsaiii = NSGAIII(problem, population_size=100, divisions_outer=100)
    ngsaiii.run(10000)
    result2 = ngsaiii.result

    ax = fig.add_subplot(1, 4, 4)
   
    lngsaii = plt.scatter([s.objectives[0] for s in ngsaii.result],
                [s.objectives[1] for s in ngsaii.result], color='b')
    lngsaiii = plt.scatter([s.objectives[0] for s in ngsaiii.result],
                [s.objectives[1] for s in ngsaiii.result], color='g')
    lmds, = plt.plot(mds_stress, mds_inter, 'ro', markersize=8)
    y_min,y_max = ax.get_ylim()
    x_min,x_max = ax.get_xlim()
    plt.xticks(np.arange(math.floor(x_min), math.ceil(x_max), math.ceil((math.ceil(x_max)-math.ceil(x_min))/5)))

    if math.ceil(y_min) == math.floor(y_max):
        plt.yticks(np.arange(math.ceil(y_min-1.1),math.ceil(y_max+1.1),1))
    else:
        plt.yticks(np.arange(math.floor(y_min-.1), math.ceil(y_max+.1), math.ceil((math.ceil(y_max+.1)-math.ceil(y_min-.1))/5)))

    plt.savefig('./output/graph'+str(number)+'.png', bbox_inches='tight')

    count = 0
    out = {}
    ngsaii_inter = result[0].objectives[1]
    ngsaii_stress = result[0].objectives[0]
    val = 0
    for solution in ngsaii.result:
        out[solution.objectives] = solution.variables
        if solution.objectives[1] < ngsaii_inter:
            ngsaii_inter = solution.objectives[1]
            ngsaii_stress = solution.objectives[0]
            val = count
        count += 1
    sample = out[result[val].objectives]
    x = sample[0:][::2]
    y = sample[1:][::2]
    newD = pd.DataFrame({"x1":x,"x2":y})
    print "Stress for NGSAII = ", ngsaii_stress
    print "Total Intersection NGSAII = ", ngsaii_inter
    graph(newD,d2,m2,number,2, ngsaii_stress, ngsaii_inter)
    
    count = 0
    out = {}
    ngsaiii_inter = result2[0].objectives[1]
    ngsaiii_stress = result2[0].objectives[0]
    val = 0
    for solution in ngsaiii.result:
        out[solution.objectives] = solution.variables
        if solution.objectives[1] < ngsaiii_inter:
            ngsaiii_inter = solution.objectives[1]
            ngsaiii_stress = solution.objectives[0]
            val = count
        count += 1
    sample = out[result2[val].objectives]
    x = sample[0:][::2]
    y = sample[1:][::2]
    newD = pd.DataFrame({"x1":x,"x2":y})
    print "Stress for NGSAIII = ", ngsaiii_stress
    print "Total Intersection NGSAIII = ", ngsaiii_inter
    graph(newD,d2,m3,number,3, ngsaiii_stress, ngsaiii_inter)


fig = plt.figure(1, figsize=(14,2))
if len(sys.argv) < 3:
    print "Please input points and edges"
    print "Usage: python Project.py <points.csv> <edges.csv>"
    sys.exit()
my_graph(sys.argv[1], sys.argv[2])
plt.savefig('output/fig_'+datetime.datetime.now().strftime('%H%M%S')+'.png', bbox_inches='tight')
plt.show()
