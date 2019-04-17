import numpy as np
if __name__ == "__main__":
    f = open('E:\\360MoveData\\Users\\51342\\Desktop\\PageRank\\input.txt')
    edges = [line.strip('\n').split(' ')for line in f]
    print(edges)

    nodes = []
    for edge in edges:
        if edge[0]not in nodes:
            nodes.append(edge[0])
        if edge[1] not in nodes:
            nodes.append(edge[1])
    print(nodes)

    N = len(nodes)

    i = 0
    node_to_num = {}
    for node in nodes:
        node_to_num[node] = i
        i +=1
    for edge in edges:
        edge[0] = node_to_num[edge[0]]
        edge[1] = node_to_num[edge[1]]
    print(edges)

    S = np.zeros([N,N])
    for edge in edges:
        S[edge[1],edge[0]] = 1
    print(S)

    for j in range(N):
        sum_of_col = sum(S[:,j])
        for i in range(N):
            S[i,j]/=sum_of_col
    print(S)
    alpha = 0.85
    A = alpha*S + (1-alpha)/N*np.ones([N,N])
    print(A)
    P_n = np.ones(N)/N
    P_n1 = np.zeros(N)
    e = 100000
    k = 0
    print('loop...')
    while e>0.00000001:
        P_n1 = np.dot(A,P_n)
        e = P_n1-P_n
        e = max(map(abs,e))
        P_n = P_n1
        k+= 1
        print("iteration %s:" % str(k),P_n1)
    print('final result:',P_n)