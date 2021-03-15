# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 18:01:05 2021

@author: atapi
"""

# módulos de Python que vamos a utilizar
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as spi
from scipy import interpolate
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from matplotlib import pyplot

import networkx as nx
import networkx.drawing.nx_pylab as nxplot

# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import os

# Directorio de trabajo
path = 'D:\\Alejandro Tapia\\Drive\\100 Loyola\\120 Investigación\\121 Mis trabajos\\4 PYTHON\\Propuesta - Penstock optimization 3D\\python'
os.chdir(path)

# IMPORTO LA BASE DE DATOS DE INDIVIDUOS PRE GENERADOS
BaseDatos = np.loadtxt('PopDataBase.csv',delimiter=",")

# MODO MULTI O SINGLE
SO = True

# IMPORTO EL MAPA DE ALTURAS [X Y Z] Y EL PERFIL DEL RÍO [X Y]
VX    = np.loadtxt('HeightMap_X.csv',delimiter=",")
VY    = np.loadtxt('HeightMap_Y.csv',delimiter=",")
TZ    = np.loadtxt('HeightMap_Z.csv',delimiter=",")
RIVER = np.loadtxt('HeightMap_R.csv',delimiter=",")
RIVER = np.flipud(RIVER)
# CREO EL MESHGRID
TY,TX = np.meshgrid(VY,VX)
# LEO DIMENSIONES
N    = len(VY)
M    = len(VX)
S    = len(RIVER[:,1])
# GUARDO TODO EN TERRAIN
TERRAIN    = np.zeros((3,M,N))
TERRAIN[0] = TX
TERRAIN[1] = TY
TERRAIN[2] = TZ

# CTES DEL MODELO DE LA PLANTA
F    = 0.010
RHO  = 1000
G    = 9.8
DNOZ = 22e-3
SNOZ = (np.pi*DNOZ**2)/4
REND = 0.9
Pmin  = 8.0E3

# COSTE DE LA TUBERÍA, SOPORTES Y EXCAVACIONES
NewValues = True
if NewValues:
    Ksop = 9.0
    Kexc = 8.0
    Xsop = 0.2
    Bexc = np.pi/180*35
else:
    Cnod = 50
    Ktub = 300
    Ksop = 130
    Kexc = 100
    Xsop = 0.2
    Bexc = np.pi/180*10

# FUNCIÓN DE INTERPOLACION EN EL TERRENO
F_TERRAIN = interpolate.interp2d(VX,VY,np.transpose(TZ))

# PARA ENCONTRAR LOS INDICES I,J DEL TERRENO CORRESPONDIENTE AL GEN I
def ij_to_k(i,j):
    k = S + N*(i) + j
    return k

# PARA ENCONTRAR EL GEN I CORRESPONDIENTE A LOS INDICES I,J DEL TERRENO
def k_to_ij(k):
    k = k-S
    i = k // N
    j = k %  N
    return i,j

# CREO EL MAPA DE NODOS VECINOS
def creaMapa():
    mapa = []
    # RIO -> TERRENO
    for k in range(S):
        mapa.append([])
        x,y = RIVER[k,:]
        lim_x = np.where(x<VX)[0][0]-1
        lim_y = np.where(y<VY)[0][0]-1
        mapa[k].append(ij_to_k(lim_x,  lim_y  ))
        mapa[k].append(ij_to_k(lim_x+1,lim_y  ))
        mapa[k].append(ij_to_k(lim_x,  lim_y+1))
        mapa[k].append(ij_to_k(lim_x+1,lim_y+1))
    # TERRENO -> TERRENO
    for k in range(S,N*M+S):
        mapa.append([])
        i,j = k_to_ij(k)
        if i>0:   mapa[k].append(ij_to_k(i-1,j))
        if i<M-1: mapa[k].append(ij_to_k(i+1,j))
        if j>0:   mapa[k].append(ij_to_k(i,j-1))
        if j<N-1: mapa[k].append(ij_to_k(i,j+1))
    # TERRENO -> RIO
    for k in range(S):
        for i,j in enumerate(mapa[k]):
            mapa[j].append(k)
    # RIO -> RIO
    for k in range(S-1):
        mapa[k].append(k+1)
        mapa[k+1].append(k)
    return mapa

def crea_grafo(mapa):
    # CREO GRAFO VACÍO
    G = nx.DiGraph()
    # RECORRO EL MAPA
    for nodo in range(len(mapa)):
        # OBTENGO LA ALTURA Z DEL NODO
        if nodo < len(RIVER):
            # SI ES DEL RIO
            xnodo,ynodo = RIVER[nodo]
        else:
            # SI ES DEL TERRENO
            inodo,jnodo = k_to_ij(nodo)
            xnodo = VX[inodo]
            ynodo = VY[jnodo]
        znodo = F_TERRAIN(xnodo,ynodo)
        # RECORRO LOS VECINOS DE CADA ELEMENTO
        for numvecino,vecino in enumerate(mapa[nodo]):
            # CALCULO LA ALTURA Z DE CADA VECINO
            if vecino < len(RIVER):
                # SI ES DEL RIO
                xvecino,yvecino = RIVER[vecino]
            else:
                # SI ES DEL TERRENO
                ivecino,jvecino = k_to_ij(vecino)
                xvecino = VX[ivecino]
                yvecino = VY[jvecino]
            zvecino = F_TERRAIN(xvecino,yvecino)
            # SI EL VECINO ESTA MAS ABAJO, CONECTO AMBOS
            if znodo[0] >= zvecino[0]:
                G.add_edge(nodo, vecino, weight=random.random())
    for nodo in range(S-1):
        G.add_edge(nodo+1,nodo,weight=100)
    return G

def dibujaGrafo(G):
    posiciones = {}
    for K in range(S):
        posiciones[K] = RIVER[K]
    for k in range(S,S+M*N):
        i,j  = k_to_ij(k)
        posiciones[k] = np.array((VX[i],VY[j]))
    nx.draw(G,with_labels='true',pos=posiciones)

def crea_individuo(size=S+M*N+5):
    # INICIALIZO EL INDIVIDUO CON CEROS
    individuo = [0 for _ in range(size)]
    mapa = creaMapa()
    G = crea_grafo(mapa)
    nodoPresa   = random.randint(0,S-1)
    nodoTurbina = nodoPresa
    while nodoTurbina==nodoPresa:
        nodoTurbina = random.randint(0,S-1)
    xp,yp = RIVER[nodoPresa,:]
    xt,yt = RIVER[nodoTurbina,:]
    if F_TERRAIN(xt,yt) > F_TERRAIN(xp,yp):
        nodoAux     = nodoPresa
        nodoPresa   = nodoTurbina
        nodoTurbina = nodoAux
#    display('Solucion de Dijkstra:')
#    display(nx.dijkstra_path(G,nodoPresa,nodoTurbina))
#    display(nodoPresa)
#    display(nodoTurbina)
    for i,j in enumerate(nx.dijkstra_path(G,nodoPresa,nodoTurbina)):
#    for i,j in enumerate(nx.shortest_path(G, source=nodoPresa, target=nodoTurbina, weight=None, method='dijkstra')):
        individuo[j]=1
    # CREO UN DIÁMETRO
    individuo[-5:] = [random.randint(0,1) for _ in range(5)]
    return individuo

def crea_individuo_mejorado(size=S+M*N+5):
    Hmin = 80
    # INICIALIZO EL INDIVIDUO CON CEROS
    individuo = [0 for _ in range(size)]
    mapa = creaMapa()
    G = crea_grafo(mapa)
    nodoPresa   = random.randint(0,S-1)
    nodoTurbina = nodoPresa
    # nodoPresa = S-1
    # nodoTurbina = 0
    while nodoTurbina==nodoPresa:
        nodoTurbina = random.randint(0,S-1)
    if nodoTurbina>nodoPresa:
        nodoAux     = nodoPresa
        nodoPresa   = nodoTurbina
        nodoTurbina = nodoAux
    xp,yp = RIVER[nodoPresa,:]
    xt,yt = RIVER[nodoTurbina,:]
    # COMPRUEBO SI ES FACTIBLE EN ALTURA
    # Y SI NO ES FACTIBLE, LOS SEPARO HASTA QUE CUMPLAN     
    while F_TERRAIN(xp,yp)-F_TERRAIN(xt,yt)<Hmin:
        if nodoTurbina>1:
            nodoTurbina = nodoTurbina-1
        if nodoPresa<S-1:
            nodoPresa   = nodoPresa+1
        xp,yp = RIVER[nodoPresa,:]
        xt,yt = RIVER[nodoTurbina,:]
        # display(nodoPresa)
        # display(nodoTurbina)
        # display(F_TERRAIN(xp,yp)-F_TERRAIN(xt,yt))
    # APLICO EL ALGORITMO DE DIJKSTRA Y CREO EL INDIVIDUO CON LA SOLUCION
    for i,j in enumerate(nx.dijkstra_path(G,nodoPresa,nodoTurbina)):
        individuo[j]=1
    # POR ULTIMO CREO UN DIÁMETRO ALEATORIO
    individuo[-5:] = [1 for _ in range(5)]
    return individuo

def crea_individuo_pregenerado():
    individuo = BaseDatos[random.randint(0,len(BaseDatos)-1)].tolist()
    return individuo

# def crea_individuo_prefab(base=BaseDatos):
#     i = random.randint(0,5000-1)
#     individuo = base[i]
#     return individuo
    
 
def cruce(ind1, ind2, eta=0.5):
    for k in range(0,len(ind1)):
        if ind1[k]==1 and ind2[k]==0:
            if random.random() < eta:
                ind2[k] = 1
        elif ind1[k]==0 and ind2[k]==1:
            if random.random() < eta:
                ind1[k] = 1
    return ind1, ind2

def mutacion(individuo, indpb_move, indpb_01, indpb_10):
    # MUTACION DEL DIAMETRO
    for k, m in enumerate(individuo[-5:]):
        if m==1:
            if random.random() < indpb_10:
                individuo[M*N+S+k] = 0
        if m==0:
            if random.random() < indpb_01:
                individuo[M*N+S+k] = 1
    # MUTACION 1-0 DE LOS NODOS
    for k, m in enumerate(individuo[:-5]):
        if m==1:
            if random.random() < indpb_10:
                individuo[k]=0
    # MUTACION MOVE DE LOS NODOS
    for k, m in enumerate(individuo[:-5]):
        if m==1:
            if random.random() < indpb_move:
                if k>=S:
                    vecinos = mapa[k]
                else:
                    vecinos = [ mapa[k][i] for i in range(0,len(mapa[k])) if mapa[k][i]<=S ]
                newpos = random.choice(vecinos)
                individuo[newpos]=1
                individuo[k]=0
    # MUTACION 0-1 DE LOS NODOS
    if random.random() < indpb_01:
        if random.random() < 0.5:
            # nodes_river   = [i for i, e in enumerate(individuo[0:S]) if e != 0]
            # verticeA = random.randint(nodes_river[0],nodes_river[-1])
            # verticeB = random.randint(nodes_river[0],nodes_river[-1])
            verticeA = random.randint(0,S)
            verticeB = random.randint(0,S)
            esquinas_verticeA = [ mapa[verticeA][i] for i in range(0,len(mapa[verticeA])) if mapa[verticeA][i]>=S ]
            esquinas_verticeB = [ mapa[verticeB][i] for i in range(0,len(mapa[verticeB])) if mapa[verticeB][i]>=S ]
            esquinas          = esquinas_verticeA + esquinas_verticeB
            indices_i = [ k_to_ij(esquinas[k])[0] for k in range(0,len(esquinas)) ]
            indices_j = [ k_to_ij(esquinas[k])[1] for k in range(0,len(esquinas)) ]
            imut = random.choice(indices_i)
            jmut = random.choice(indices_j)
            indice_mutado = ij_to_k(imut,jmut)
            # if F_TERRAIN(RIVER[indice_mutado][0],RIVER[indice_mutado][1]) >= F_TERRAIN(RIVER[S-1][0],RIVER[S-1][1]):
            #     indice_mutado = random.randint(0,S-1)
            # elif F_TERRAIN(RIVER[indice_mutado][0],RIVER[indice_mutado][1]) <= F_TERRAIN(RIVER[0][0],RIVER[0][1]):
            #     indice_mutado = random.randint(0,S-1)
            # print('El nodo T ',imut,',',jmut,' se hace 1')
        else:
            indice_mutado = random.randint(0,S-1)
            # print('El nodo R ',indice_mutado,' se hace 1')
        individuo[indice_mutado] = 1
    return individuo,
    
   
def fitness_function(INDIVIDUO, analisis=False):
    PENALIZA = 1000000
    # EL CROMOSOMA TIENE TAMAÑO: S + MxN
    # LOS S PRIMEROS GENES SON NODOS A LO LARGO DEL RIO
    # LOS MxN SIGUIENTES SON NODOS DE LA MALLA DEL TERRENO
    # SEPARO EL INDIVIDUO EN LAS TRES PARTES QUE LO COMPONEN
    IND_RIVER    = INDIVIDUO[0:S]
    IND_TERRAIN  = INDIVIDUO[S:-5]
    IND_DIAMETER = INDIVIDUO[-5:]
#    # CON F_RIVER CALCULAMOS LA ALTURA DE LOS NODOS DEL RIO
#    F_TERRAIN = sp.interpolate.interp2d(VX,VY,TERRAIN[2,:,:])
    # ANTES DE SEGUIR, SI NO HAY 2 O MÁS NODOS EN EL RÍO, NO VALE
    if np.sum(IND_RIVER)<2:
        coste = PENALIZA
        return coste,
    # BUSCO LOS NODOS SITUADOS EN EL RÍO
    # DE ELLOS, EL DE MENOR Z SERÁ LA TURBINA Y EL DE MAYOR Z SERÁ LA PRESA
    nodes_river = [i for i, e in enumerate(IND_RIVER) if e != 0]
    # SACO SUS COORDENADAS X-Y A PARTIR DEL PERFIL DEL RIO
    coord_river =  np.zeros([len(nodes_river),4])
    # SACO SU COORDENADA Z DEL PERFIL DEL TERRENO
    for i in range(len(nodes_river)):
        coord_river[i][0] = 1
        coord_river[i][1] = RIVER[nodes_river[i],0]
        coord_river[i][2] = RIVER[nodes_river[i],1]
        coord_river[i][3] = F_TERRAIN(coord_river[i][1],coord_river[i][2])
    # EN COORD_RIVER ESTÁN LAS COORDENADAS DE LOS NODOS CONTENIDOS EN EL RIO
    # POR EJEMPLO, COORD_RIVER[i] ES EL PUNTO i DEFINIDO POR [X, Y, Z]
    # AHORA BUSCO LOS NODOS SITUADOS POR EL TERRENO
    nodes_terrain = [i+S for i, e in enumerate(IND_TERRAIN) if e != 0]
#    display('Individuo generado:')
#    display(nodes_river)
#    display(nodes_terrain)
    # SACO SUS COORDENADAS X-Y-Z A PARTIR DEL PERFIL DEL TERRENO
    coord_terrain = np.zeros([len(nodes_terrain),4])
    for i in range(0,len(nodes_terrain)):
        # RECORRO LOS NODOS DEL INDIVIDUO
        node = nodes_terrain[i]
#        row  = ((node) // N)
#        col  = ((node) %  N)
        row,col = k_to_ij(node)
        # EXTRAIGO SUS COORDENADAS
        coord_x = TERRAIN[0,row,col]
        coord_y = TERRAIN[1,row,col]
        coord_z = TERRAIN[2,row,col]
        # Y LAS VOY GUARDANDO
        coord_terrain[i][:] = np.array([ 0, coord_x, coord_y, coord_z])
    # JUNTO TODOS LOS NODOS (DEL RIO Y DEL TERRENO) EN UNA VARIABLE
    coord_individuo = np.concatenate((coord_river,coord_terrain))
    # ORDENO LOS NODOS POR COORDENADA Z CRECIENTE
    coord_individuo = coord_individuo[coord_individuo[:,3].argsort()]
    # SI EL PRIMER ELEMENTO (Z MAS BAJA) NO PERTENECE AL RIO, NO VALE
    # SI EL ULTIMO ELEMENTO (Z MAS ALTA) NO PERTENECE AL RIO, NO VALE
#    display(coord_individuo)
    if coord_individuo[0,0]+coord_individuo[-1,0]<2:
        coste = PENALIZA*2
        return coste,
    # CALCULO LA LONGITUD DE LA TUBERÍA Y LA ALTURA BRUTA
    L = np.sum(np.sqrt( (coord_individuo[1:,1]-coord_individuo[0:-1,1])**2 + (coord_individuo[1:,2]-coord_individuo[0:-1,2])**2 + (coord_individuo[1:,3]-coord_individuo[0:-1,3])**2))
    H = coord_individuo[-1,3]-coord_individuo[0,3]
    # CALCULO EL DIÁMETRO CORRESPONDIENTE
    Db_str = "0b"
    for bit in IND_DIAMETER:
        Db_str = Db_str + str(int(bit))
    D = (1+int(Db_str, 2)) * 10**-2
    # TOTAL DE CODOS (QUITANDO TURBINA Y PRESA)
    Nc = len(nodes_river)+len(nodes_terrain)-2
    # MODELO DE LA PLANTA HIDRÁULICA
    # POTENCIA DE LA PLANTA
    P = REND * (RHO/(2*SNOZ**2))*(H/(1/(2*G*SNOZ**2)+F*L/(D**5)))**(3/2)
    # CAUDAL TURBINADO
    Q = (H/(1/(2*G*SNOZ**2)+F*L/D**5))**(1/2)
    if P < Pmin:
        coste = PENALIZA*5
        return coste,
    # SACO PUNTOS INTERMEDIOS ENTRE LOS NODOS PARA EVALUAR ALTURAS
    resol = 1
    for i in range(len(nodes_river)+len(nodes_terrain)-1):
        # PARA CADA SEGMENTO ENTRE NODOS    i -> i+1
        coord_i = coord_individuo[i,  1:]
        coord_j = coord_individuo[i+1,1:]
        # MIDO LA DISTANCIA ENTRE ELLOS Y CALCULO LAS SUBDIVISIONES
        dL      = np.linalg.norm(coord_j - coord_i)
        subdivs = int(1+np.ceil(dL//resol))
        # CREO LOS NUEVOS PUNTOS INTERIORES DEL SEGMENTO
        coord_tramo = np.array(( np.linspace(coord_i[0],coord_j[0],subdivs),
                                 np.linspace(coord_i[1],coord_j[1],subdivs),
                                 np.linspace(coord_i[2],coord_j[2],subdivs)))
        coord_tramo = np.transpose(coord_tramo)
        # ALMACENO LOS COORD_TRAMO EN COORD_TOTAL
        if i==0:
            coord_individuo_subdiv = coord_tramo
        else:
            coord_individuo_subdiv = np.r_[coord_individuo_subdiv,coord_tramo[1:]]
    # SACO COSTE DE SOPORTES
      # EMPIEZO SACANDO LA LONGITUD DE TODOS LOS SUBINTERVALOS EN EL PLANO XY
    longitud_tramos = np.zeros(len(coord_individuo_subdiv))
    for i in range(1,len(longitud_tramos)):
        longitud_tramos[i] = np.linalg.norm(coord_individuo_subdiv[i][0:2]-coord_individuo_subdiv[i-1][0:2])
      # Y LUEGO CONSTRUYO EL VECTOR S
    rio2D_s = longitud_tramos
    for i in range(1,len(longitud_tramos)):
        longitud_tramos[i]=longitud_tramos[i]+longitud_tramos[i-1]
    rio2D_dz = np.zeros(len(longitud_tramos))
    for i in range(0,len(rio2D_dz)):
        rio2D_dz[i] = coord_individuo_subdiv[i][2] - F_TERRAIN(coord_individuo_subdiv[i][0], coord_individuo_subdiv[i][1])
    rio2D_dz_pos = rio2D_dz.copy()
    rio2D_dz_pos[np.where(rio2D_dz<=0)]=0
    rio2D_dz_neg = rio2D_dz.copy()
    rio2D_dz_neg[np.where(rio2D_dz>=0)]=0
    rio2D_dz_neg = rio2D_dz_neg*-1
    rio2D_cost_sop = Ksop*Xsop*np.power(rio2D_dz_pos*4/3, 1)
    # rio2D_cost_sop[0] = 0
    rio2D_cost_exc = Kexc*np.tan(Bexc)*np.power(rio2D_dz_neg, 2) + D*rio2D_dz_neg
    # rio2D_cost_exc[0] = 0
    C_sop = np.trapz(rio2D_cost_sop,rio2D_s)
    C_exc = np.trapz(rio2D_cost_exc,rio2D_s)
    if NewValues:
        # C_tub = (50*Nc+L)*(616.1*D**2 + 99.76*D + 13.14)
        C_tub = L*(616.1*D**2 + 99.76*D + 13.14) + Nc*(1200*D**3+50)
        # C_tub = L*(270.5*D+3.359)*D + Nc*(1711*D**2+18.93)
    else:
        C_tub = Ktub*(L+Cnod*Nc)*D**2
    C = C_tub + C_exc + C_sop
    if analisis:
        print('\n\n\n')
        print('Cost: \t',"%.2f" % C,' c.u.')
        print('C_sop', C_sop)
        print('C_exc', C_exc)
        print('C_tub', C_tub)
        print('H: \t', H, 'm')
        print('Pot: \t',"%.2f" % P,'W')
        Q = Q*1000
        print('Flow: \t',"%.2f" % Q,'L/s')
        print('Fric: \t', F, '-')
        print('Nodos: \t',Nc)
        print('Long: \t',"%.2f" % L,' m')
        print('Diam: \t',"%.2f" % D,' m')
        # PLOTEO DE MAPA TOPOGRAFICO Y SOLUCION
        fig, ax = plt.subplots()
        # Con recorte, usar VY2, TZ2. Sin recorte, usar VY, TZ.transpose()
        VY2 = VY[:45]
        TZ2 = TZ.transpose()[:45]
        XX,YY = np.meshgrid(VX,VY2)
        CS = ax.contour(XX,YY,TZ2,20,colors='grey',inline=1,alpha=1,linewidths=0.6)
        ax.clabel(CS, inline=1, fontsize=7)
        smoothRiver=True
        if smoothRiver:
            distance = np.cumsum( np.sqrt(np.sum( np.diff(RIVER, axis=0)**2, axis=1 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]
            alpha = np.linspace(0, 1, 1000)
            interpolator =  interp1d(distance, RIVER, kind='cubic', axis=0)
            interpolated_points = interpolator(alpha)
            ax.plot(*interpolated_points.T, color='blue', linewidth=2);
        else:
            ax.plot(RIVER.transpose()[0],RIVER.transpose()[1],color='blue', linewidth=2)
        coord_individuo = coord_individuo.transpose()        
        ax.plot(coord_individuo[1],coord_individuo[2],linewidth=2,color='red')
        ax.plot(coord_individuo[1][0],coord_individuo[2][0],linewidth=0,marker='v',color='red')
        ax.plot(coord_individuo[1][-1],coord_individuo[2][-1],linewidth=0,marker='^',color='red')
        plt.text(coord_individuo[1][0]-30, coord_individuo[2][0]+40, 'Dam', fontsize=12)
        plt.text(coord_individuo[1][-1]+25, coord_individuo[2][-1]+25, 'Powerhouse', fontsize=12)        
        fig.gca().set_aspect('equal', adjustable='box')
        ax.legend(('River', 'MHPP Layout'))
        plt.xlabel('x-coordinate (m)')
        plt.ylabel('y-coordinate (m)')
        # PLOTEO DE GAP
        fig2, ax2 = plt.subplots()
        ax2.plot(rio2D_s,rio2D_dz)
        plt.ylim((-4,4))
        plt.grid(True)
        plt.xlabel('s-coordinate (m)')
        plt.ylabel('Penstock-terrain gap, \u03B5 (m)')
    if SO:
        return C,
    else:
        return C,P,

# PARA SO
# Paso1: creación del problema
creator.create("Problema1", base.Fitness, weights=(-1,))
# Paso2: creación del individuo
creator.create("individuo", list, fitness=creator.Problema1)
# Creamos la caja de herramientas
toolbox = base.Toolbox() 
# Registramos nuevas funciones
toolbox.register("individuo", tools.initIterate, creator.individuo, crea_individuo_pregenerado )
toolbox.register("ini_poblacion", tools.initRepeat, list, toolbox.individuo)

def unico_objetivo_ga(c, m, ETA, Pmov, P01, P10):
    """ Los parámetros de entrada son la probabilidad de cruce y la
    probabilidad de mutación """

    NGEN   = 100
    MU     = 2000
    LAMBDA = 2000
    CXPB   = c
    MUTPB  = m
    
    # Operaciones genéticas
    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", cruce,eta=ETA)
    toolbox.register("mutate", mutacion, indpb_move=Pmov, indpb_01=P01 , indpb_10=P10 )
    toolbox.register("select", tools.selTournament, tournsize = 3)
    
    pop = toolbox.ini_poblacion(n=MU)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    logbook = tools.Logbook()
    
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB,
                                             MUTPB, NGEN, stats=stats,
                                             halloffame=hof, verbose=True)
    
    return pop, hof, logbook,

#%%############################################################################
# AJUSTE DE HIPERPARÁMETROS
###############################################################################

mapa = creaMapa()

# Abrimos dos archivos de texto para almacenar los resultados
res_individuos = open('resultados/IND.txt', "a")
res_fitness    = open('resultados/FIT.txt', "a")


# Hacemos la llamada al algoritmo
pop, hof, log = unico_objetivo_ga(0.70, 0.30, eta=0.6, Pmov=0.05, P01=0.05, P10=0.05)

# Almacenamos el logbook en un csv independiente
df_log = pd.DataFrame(log)
log_filename = 'resultados/LOG.csv'
df_log.to_csv(log_filename, index=False)

# Almacenamos la solución en los ficheros de texto
for ide, ind in enumerate(pop):
    
    res_individuos.write(str(i))
    res_individuos.write(",")
    res_individuos.write(str(ide))
    res_individuos.write(",")
    res_individuos.write(str([i for i,j in enumerate(ind) if j==1]))
    res_individuos.write("\n")
    
    res_fitness.write(str(i))
    res_fitness.write(",")
    res_fitness.write(str(ide))
    res_fitness.write(",")
    res_fitness.write(str(c))
    res_fitness.write(",")
    res_fitness.write(str(m))
    res_fitness.write(",")
    res_fitness.write(str(ind.fitness.values[0]))
    res_fitness.write("\n")
    
res_fitness.close()
res_individuos.close()

















