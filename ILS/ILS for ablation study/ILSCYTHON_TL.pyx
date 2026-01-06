import cython
import numpy as np
cimport numpy as np
import time
import gc
import pandas as pd
#import knapsack as knap
from scipy.optimize import linear_sum_assignment
from itertools import combinations
import sys
from ortools.algorithms.python import knapsack_solver

ctypedef np.int_t DTYPE_INT
ctypedef np.double_t DTYPE_FLOAT

#ctypedef np.float64_t DTYPE_FLOAT


def CREAR_RUTA_GREEDY_NP(Tmax,tij,r,p,n,k,t): # asume que no se visitan nodos sin realizar actividades (desigualdad triangular?)
    zz=np.zeros(shape=(t,n,k)) # array 3d que contiene si elgrupo de trabajo t realiza la actividad k en el nodo n
    ruta=[0] # comenzar ruta
    nv=[i for i in range(1,n)] # set de nodos no visitados
    #print(nv)
    Tactual=0 # tiempo recorrido inicial
    gr=[j for j in range(0,t)] # set de trabjadores
    actividades=[ i for i in range(0,k)]
    # ordenar no visitados de acuerdo a su ratio suma de beneficios / suma de tiempos
    Tactualnodo = 0 # tiempo actual recorrido
    Tactualretorno=0 # tiempo actual más retorno desde el último nodo
    profit=0 # beneficio acumulado
    
    sel1=5
    sel2=0
    sel3=0
    while  len(ruta) <= t and len(gr) >0 and len(nv) >0:  # iterar mientras queden trabajadores por asignar o existan nodos no visitados ( en caso de que no se encontraran nodos posibles)
          #se podría preguntar directamente si se cumple la distancia y su retorno 
          #print('***LEN***',len(ruta))
          
          if sel1==0:  
            mindist, minnodepos= np.min(tij[ruta[-1],nv]),np.argmin(tij[ruta[-1],nv])
          elif sel1==1:
            mindist, minnodepos= np.min(tij[ruta[-1],nv]+[np.sum(r[:,i,:]) for i in nv]),np.argmin(tij[ruta[-1],nv]+[np.sum(r[:,i,:]) for i in nv])
          elif sel1==2:
            mindist, minnodepos= np.min(tij[ruta[-1],nv]+[np.sum(r[:,i,:]/(t*k)) for i in nv]),np.argmin(tij[ruta[-1],nv]+[np.sum(r[:,i,:]/(t*k)) for i in nv])
          elif sel1==3:
            mindist, minnodepos= np.min(tij[ruta[-1],nv]+[np.sum(r[gr,i,:]/(len(gr)*k)) for i in nv]),np.argmin(tij[ruta[-1],nv]+[np.sum(r[gr,i,:]/(len(gr)*k)) for i in nv])#[0] #seleccionar el nodo y su distancia de acuerdo al vecino más cercano del último nodo en la ruta
          elif sel1==4:
            mindist, minnodepos= np.max([np.sum(p[i,:]) for i in nv]),np.argmax([np.sum(p[i,:]) for i in nv])#[0] #seleccionar el nodo y su distancia de acuerdo al vecino más cercano del último nodo en la ruta 
          elif sel1==5:
            mindist, minnodepos= np.max([np.sum(p[i,:])/tij[ruta[-1],i] for i in nv]),np.argmax([np.sum(p[i,:])/tij[ruta[-1],i] for i in nv])#[0] #seleccionar el nodo y su distancia de acuerdo al vecino más cercano del último nodo en la ruta
          elif sel1==6:
            mindist, minnodepos= np.max([np.sum(p[i,:])/np.sum(r[gr,i,:]/(len(gr)*k)) for i in nv]),np.argmax([np.sum(p[i,:])/np.sum(r[gr,i,:]/(len(gr)*k)) for i in nv])#[0] #seleccionar el nodo y su distancia de acuerdo al vecino más cercano del último nodo en la ruta
          elif sel1==7:
            mindist, minnodepos= np.max([np.sum(p[i,:])/np.sum(r[gr,i,:]/(t*k)) for i in nv]),np.argmax([np.sum(p[i,:])/np.sum(r[gr,i,:]/(t*k)) for i in nv])#[0] #seleccionar el nodo y su distancia de acuerdo al vecino más cercano del último nodo en la ruta  
          elif sel1==8:
            mindist, minnodepos= np.max([np.max(p[i,:])/np.sum(r[gr,i,:]/(t*k)) for i in nv]),np.argmax([np.max(p[i,:])/np.sum(r[gr,i,:]/(t*k)) for i in nv])#[0] #seleccionar el nodo y su distancia de acuerdo al vecino más cercano del último nodo en la ruta  
          elif sel1==9:
            mindist, minnodepos= np.max([np.max(p[i,:])/tij[ruta[-1],i] for i in nv]),np.argmax([np.max(p[i,:])/tij[ruta[-1],i] for i in nv])#[0] #seleccionar el nodo y su distancia de acuerdo al vecino más cercano del último nodo en la ruta  
            
        
        
            #print(mindist)
            #print(minnodepos)
       
          #print('***DIS,POS***')
          #print( mindist, minnodepos)
          minnode = nv[minnodepos] # nodo asociado al vecino más cercano
          #print('***NODO***')
          #print( minnode)
          #print('****dis****')
          mindis=tij[ruta[-1],minnode]
          #print(mindis)
          #ruta.append(minnode) # agregar a la ruta el nodo seleccionado
          Tactualnodo = Tactualnodo + mindis # tiempo actual de entrada al último nodo de la ruta
          Tactualretorno=Tactualnodo + tij[minnode,0] # calculo de tiempo de retorno al nodo 1 desde el último nodo de  la ruta
          #print('***TACTUAL,TRETORNO')
          #print( Tactualnodo,Tactualretorno)
          nv.pop(minnodepos) # remover el nodo seleccionado de la lista de nodos por visitar
          #print('***NVISITADOS****')
          #print( nv)
          if Tactualretorno < Tmax:
            #ruta.append(minnode) # agregar a la ruta el nodo seleccionado
            if sel3==0:
                rr=[np.sum(r[i,minnode,:]) for i in gr] # calcula la suma de tiempos de un grupo de trabajo t en el nodo actual
                min_tindex=np.argmin(rr)#[0] # índice con la menor suma de tiempos en un nodo para realizar todas las actividades
            else:
                rr=[np.min(r[i,minnode,:]) for i in gr] # calcula el minimo de los tiempos de un grupo de trabajo t en el nodo actual
                min_tindex=np.argmin(rr)#[0] # índice co
            
            
            tsel=gr[min_tindex] # grupo de trabajo t seleccionado
            #print('****GR ACTUAL****')
            #print(gr)
            #print('***TINDEX,TSELECCIONADO***')
            #print(min_tindex,tsel)     
            act=[i for i in range(k)] # lista de actividades, que se ira reduciendo
            candidatosk=[ i for i in act if Tactualnodo + r[tsel,minnode,i] <= Tmax]
            #print('***CANDIDATOS***')
            #print(candidatosk)
            if len(candidatosk) >0: #si hay al menso un candidato valido
                ruta.append(minnode) # se agrega el nodo a la ruta ya que es viable
                gr.pop(min_tindex) # se elimina de los grupos de trabajo por utilizar, ya que  hay al menos una actividad que podrá realizar
                Tactualnodoact=Tactualnodo # asociar el tiempo actual de llegada al nodo como el  tiempo  en que finalizará las actividades en el nodo actual en base a  las actividades que el grupo de trabajo t realice
                for j in range(k): #range(k) while len(act) >0


                    candidatosk=[ i for i in act if Tactualnodoact + r[tsel,minnode,i] <= Tmax]  # ver actividades candidatas que cumplan que si se realizan, no se supera Tmax
                    #print('***CANDIDATOS_for***')
                    #print(candidatosk)
                    if len(candidatosk) >0: # en caso de existir candidatos
                        if sel2==0:
                            max_zindex=np.argmax(p[minnode,candidatosk])  # greedy por maximo beneficio
                        elif sel2==1:
                            max_zindex=np.argmin(r[tsel,minnode,candidatosk]) # greedy por menor duración
                        elif sel2==2:
                            max_zindex=np.argmax(p[minnode,candidatosk]/r[tsel,minnode,candidatosk]) # greedy por  mayor ratio beneficio / tiempo
                        #print('****p****')
                        #print(p[minnode,candidatosk])
                        zsel=candidatosk[max_zindex] # actividad z seleccionada  #act[max_zindex]
                        #print('****ZINDEX,ZSEL')
                        #print(max_zindex,zsel)
                        zz[tsel,minnode,zsel]=1 # indicar que el grupo de trabajo tsel realizará la actividad max_indexz en el nodo minnode (último nodo de la ruta)
                        #act.pop(max_zindex) # remover la actividad seleccionada de las candidatas
                        act.remove(zsel)  # remover la actividad seleccionada de las candidatas
                        #print('*****ACT****')
                        #print(act)
                        Tactualnodoact=Tactualnodoact + r[tsel,minnode,zsel]  # Se calcula el tempo actual de finalización del grupo de trabajo t en el nodo actual minnode
                        profit= profit + p[minnode,zsel]# ctualizar el beneficio obtenido de forma global por la ruta
                        #print('**TACTUAL,PROFIT***')
                        #print(Tactualnodoact,profit,p[minnode,zsel])
                        #print('*********I*********')
                    else:
                        #print('OUT')
                        break # salir del ciclo for si ya no se puede realizar más actividades en el nodo minnode con el grupo de trabajo seleccionado tsel
          #else:
            #print('------PROBAR OTRO NODO-----')
          #print('****,RUTA,****')
          #print(ruta)
    #profit= np.array(zz)* 
    ruta.append(0)
    return np.int32(ruta), profit,zz




def calcdisentrada(ruta,tij):
    cdef int i
    u=np.array([0.0 for i in range(len(ruta))])
   
    for i in range(len(ruta)-1): # calculo de los tiempos de entrada
        u[i+1]=(u[i] + tij[ruta[i],ruta[i+1]])  # tiempo de entrada anterior + distancia tij
       
    return u#u[1:] # retornar tiempos sin el nodo base






def calcdisentrada_swap(ruta,u, tij, ind):
    cdef int s
    # Actualizar tiempos para el nodo invertido y los nodos posteriores
    for s in range(ind , len(ruta)):
           
            u[s] = u[s - 1] + tij[ruta[s - 1], ruta[s]]
      











def xswap_aleatorio_en_ruta_infactible(ruta,tij,u):
    
   
    if len(ruta) >3:    
        di,dj = np.random.choice(range(1,len(ruta)-2), 2, replace=False)#random.choice(combinaciones) 
        
        ruta[di], ruta[dj] = ruta[dj], ruta[di]

        ink=min(di,dj)
        
        calcdisentrada_swap(ruta,u, tij, ink)
    return #ruta,uu 




def xcambio_aleatorio_fueraderuta_simple_infactible(ruta, tij, z, n, u, nodos):
    cdef int i,j
    nv = list(set(range(n)) - set(ruta))  # Nodos no visitados
   
    
    i = np.random.randint(1, len(ruta) - 1)  # Posición en la ruta (excepto el depósito)
    j = np.random.choice(nv)  # Seleccionar un nuevo nodo no visitado
    temp=ruta[i]
    
    ruta[i] = j  # Cambiar el nodo en la ruta
    
    
    nodos[np.where(nodos == temp)[0][0]] = j

   
    z[:, j, :] = z[:, temp, :].copy()
    z[:, temp, :] = 0

    
    calcdisentrada_swap(ruta,u, tij, i)
    
    return  #ruta, uu, z, nodos  # Devolver la ruta, u, z y nodos si no se pudo hacer un cambio factible


def xls_assignment0(ruta, t, rik, nodos, zz):
    cdef int i,j
    medians = np.median(rik[:, ruta[1:-1], :], axis=2)
    
    zz[:,:,:]=0  
    
   
    nodos[:]=0  
    row_ind, col_ind = linear_sum_assignment(medians)
   
    
    for i, j in zip(row_ind, col_ind):
       
        nodos[i] = ruta[j+1]
    
    
    
    
    return #col_ind, nodos


def xswap_grupodetrabajo(ruta, tij, rik, z, t, n, nodos):
    i, j = np.random.choice(range(t), 2, replace=False)  # Seleccionar aleatoriamente dos grupos de trabajo
    
    z[i,:,:], z[j,:,:] = z[j,:,:].copy(), z[i,:,:].copy()
    
    nodos[i], nodos[j] = nodos[j], nodos[i]

    return #nodos




def anadir_nodo_infactible(ruta,tij,n,nodos):  # se podría agregar u y devolver
  
    nv=list(np.setdiff1d(np.arange(n),ruta)) # nodos sin visitar
    
    i,j = np.random.choice(range(1,len(ruta)), 1)[0], np.random.choice(nv, 1)[0] # seleccionar posición y nodo a ingresar
    
    ruta=np.insert(ruta,i,j)  # insertar nodo j en posicion i en la ruta
    
    
    u=calcdisentrada(ruta,tij)  # calcular los tiempos de entrada
    
    nodos[np.where(nodos==0)[0][0]]=j
    
    return ruta,u#,z,nodos  # retornar la ruta modificada, u,z y nodos








def xtwoopt_v1_f_opt(np.ndarray[DTYPE_INT, ndim=1] ruta, 
                              np.ndarray[DTYPE_FLOAT, ndim=1] u, 
                             np.ndarray[DTYPE_FLOAT, ndim=2] tij):
    
    cdef int n = ruta.shape[0]  
    cdef int i, j               
    cdef double ul, sma, sm     
    cdef bint imp = True        
    cdef np.ndarray[DTYPE_FLOAT, ndim=1] ut = np.copy(u)  

    ul = u[-1]  # Último elemento de `u`
    sm = np.sum(u)  # Suma total inicial de `u`

    # Inicio del algoritmo 2-opt
    while imp:
        imp = False
        for i in range(n - 2):
            for j in range(i + 2, n - 1):  # Evitar índices adyacentes
                # Revertir segmento de ruta
                ruta[i + 1:j + 1] = ruta[i + 1:j + 1][::-1]
                
                
                calcdisentrada_2(ruta, u, tij, i, j)  # Asegúrate de que esta función esté optimizada

               
                sma = np.sum(u)

                # Verificar condiciones de mejora
                if u[-1] < ul or (u[-1] <= ul and sma < sm):
                    imp = True
                    return  # Salida al encontrar mejora
                else:
                    # Restaurar valores originales si no mejora
                    u[:] = ut
                    ruta[i + 1:j + 1] = ruta[i + 1:j + 1][::-1]

            if imp:
                break  # Salir del bucle externo si hay mejora

    return




def calcdisentrada_2(np.ndarray[DTYPE_INT, ndim=1] ruta, 
                     np.ndarray[DTYPE_FLOAT, ndim=1] u, 
                     np.ndarray[DTYPE_FLOAT, ndim=2] tij, 
                     int i, int j):
    
    cdef int k
    cdef int n = len(ruta)
    
    # Update the time for the node at i+1
    u[i + 1] = u[i] + tij[ruta[i], ruta[i + 1]]
    
    # Update the rest of the times for nodes from i+2 to n-1
    for k in range(i + 2, n):
        u[k] = u[k - 1] + tij[ruta[k - 1], ruta[k]]




def xtiempogrupot(
    np.ndarray[DTYPE_INT, ndim=1] ruta,
    np.ndarray[DTYPE_FLOAT, ndim=2] tij,
    np.ndarray[DTYPE_FLOAT, ndim=3] rik,
    np.ndarray[DTYPE_FLOAT, ndim=3] z,
    int t,
    int n,
    np.ndarray[DTYPE_FLOAT, ndim=1] u,
    np.ndarray[DTYPE_INT, ndim=1] tiempot,
    np.ndarray[DTYPE_INT, ndim=1] nodo
):
    # Obtener solo los nodos intermedios (sin inicio y fin)
    cdef np.ndarray[DTYPE_INT, ndim=1] nodos_intermedios = ruta[1:-1]
    
    # Vectorización del cálculo de suma de actividades por grupo
    cdef np.ndarray[DTYPE_FLOAT, ndim=2] actividades_por_grupo = z[:, nodos_intermedios, :].sum(axis=2)
    
    # Vectorización del proceso completo
    cdef int i, idx_nodo, nodo_asignado
    cdef DTYPE_INT max_actividades
    #cdef DTYPE_FLOAT tiempo_actividades
    
    for i in range(t):
        # Encontrar el nodo donde trabaja el grupo i
        max_actividades = actividades_por_grupo[i].max()
        
        if max_actividades > 0:
            # Obtener índice del nodo donde trabaja el grupo
            idx_nodo = actividades_por_grupo[i].argmax()
            nodo_asignado = nodos_intermedios[idx_nodo]
            
            # Calcular tiempo de finalización
            tiempo_actividades = (z[i, nodo_asignado, :] * rik[i, nodo_asignado, :]).sum()
            
            nodo[i] = nodo_asignado
            tiempot[i] = u[idx_nodo + 1] + tiempo_actividades
        else:
            nodo[i] = 0
            tiempot[i] = 0
    
    return










 


def xILS_11_NP_XX( DTYPE_FLOAT Tmax, np.ndarray[DTYPE_FLOAT, ndim=2] tij, np.ndarray[DTYPE_FLOAT, ndim=3] r, 
    np.ndarray[DTYPE_FLOAT, ndim=2] p, DTYPE_INT n, DTYPE_INT k, DTYPE_INT cantidadt, DTYPE_INT rng,
    DTYPE_INT niter, DTYPE_FLOAT alpha, DTYPE_FLOAT a1, DTYPE_FLOAT a2, 
    DTYPE_FLOAT b1, DTYPE_FLOAT c0, DTYPE_FLOAT cota, DTYPE_INT timelimit
):
    
    
    #cdef int i
    cdef DTYPE_INT i, it, hh, iterMax, infac, eli, las, nuasin, ksum, idx
    cdef DTYPE_FLOAT profit_sol, best_profit, initprofit, profit, rd, rd1, rd0
    cdef DTYPE_FLOAT tlim, tini, titer, tiempototal, gap
    cdef np.ndarray[DTYPE_INT, ndim=1] best_sol, sol, nodos#, ruta_arr
    cdef np.ndarray[DTYPE_INT, ndim=1]  tiempot, best_tiempot
    cdef np.ndarray[DTYPE_FLOAT, ndim=1] u, best_u
    cdef np.ndarray[DTYPE_FLOAT, ndim=3] zz, best_z
    np.random.seed(rng)
    #random.seed(rng)
    MEJORA=0
   
    start = time.time()
 
    sol,best_profit,zz= CREAR_RUTA_GREEDY_NP(Tmax,tij,r,p,n,k,cantidadt)
    
    u=calcdisentrada(sol,tij)#,n)
    
    #u=np.array(u)
    tiempot =np.array([0 for i in range(cantidadt)], dtype=np.int32)
    
    nodos=np.array([0 for i in range(cantidadt)], dtype=np.int32)
    
    xtiempogrupot(sol,tij,r,zz,cantidadt,n,u,tiempot,nodos)#,nodos,sol)  (ruta,t,rik,zz,5,7,uu,tiempot,nodos)

    
    best_profit=xLS_knapsackORTOOLS(Tmax,p,tij,r,u,zz,cantidadt,n,nodos,sol)
  
    xtwoopt_v1_fact(sol,u,tij,Tmax,r,zz,cantidadt,nodos,tiempot)#tiempot,Tmax,zz,cantidadt,n,tij,nodos,r)
   


    best_profit=xLS_knapsackORTOOLS(Tmax,p,tij,r,u,zz,cantidadt,n,nodos,sol)
   
  
    xtiempogrupot(sol,tij,r,zz,cantidadt,n,u,tiempot,nodos)
    
    
   
   

    initprofit=best_profit
    

    best_sol = sol.copy()
    best_z=zz.copy()
    best_u=u.copy()
    best_tiempot=tiempot.copy()
    best_nodos=nodos.copy()
   
    profit_sol=best_profit
    iterMax = niter
    infac=0
    eli=0
    i=0
    tlim=0#time.time()
    las=1
    nuasin=0
    ksum=0
    while  tlim < timelimit: #i <= iterMax or
        #print(i,type(u))
        i=i+1
        tini=time.time()
    
        rd=np.random.random() #-0.6
       
        if rd <= a1:
            
             
             it=0
                
             xswap_grupodetrabajo(sol, tij, r, zz, cantidadt, n, nodos)
            
        elif rd <= a1 + a2:
             
             
             xswap_aleatorio_en_ruta_infactible(sol, tij, u)
             it=0
        elif rd <= 1:#0.6:
             
             
             xcambio_aleatorio_fueraderuta_simple_infactible(sol, tij, zz, n, u, nodos)
             it=1
            
        
        rd1=np.random.random()
        if len(sol) < (cantidadt+2) and rd1 <= c0:
            sol,u=anadir_nodo_infactible(sol,tij,n,nodos)
            it=1
            #u=np.array(u) 
            ksum=ksum+1
            #print('nueva sol',sol,u,nodos,ksum)
        
        
        
        
        
        
        rd0=np.random.random()
       
            
            
        if rd0 <= b1 :
           
            hh=0
                 
            xtwoopt_v1_f_opt(sol, u, tij)
            #xtwoopt_v1_f_full_ciclo(sol, u, tij,i)
            

        elif rd0 <= 1 :
                     
            hh=1
                               
            sol,zz=xcheapest_ins_dist_mod_profit(Tmax,p,tij,r,u,zz,cantidadt,n,sol,nodos) 
            xtwoopt_v1_f_opt(sol, u, tij)
            it=1




        if (np.any(u[:-1] >= Tmax)) or (u[-1]> Tmax):
            #print('INFAC U',i)
            sol,zz=xcheapest_ins_dist_mod_dist(Tmax,p,tij,r,u,zz,cantidadt,n,sol,nodos) 
            it=1
            #if np.all(u <= Tmax):
        if (np.all(u[:-1] < Tmax)) and (u[-1]<= Tmax):
            #print('MEJORO INFAC U',i)
            if it==1 or las==1:
                xls_assignment0(sol, cantidadt, r, nodos, zz)
                nuasin=nuasin+1


            profit_sol=xLS_knapsackORTOOLS(Tmax,p,tij,r,u,zz,cantidadt,n,nodos,sol)

            xtiempogrupot(sol,tij,r,zz,cantidadt,n,u,tiempot,nodos)
            
        else:
            profit_sol=0
            
        
            
           
            
        las=0      
        #test=np.array([np.sum(zz[i,:,:]) for i in range(cantidadt)])
        

        #if np.any(test >5):
        #    print('PROBLEMAS',i)
        #    break
       
        
       
        #if len(np.where(nodos==0)[0]) >=1 or len(sol)> (cantidadt +2):
        elementos_a_borrar = np.setdiff1d(sol[1:-1], nodos)  
        if len(elementos_a_borrar) >=1:
           
            sol = sol[~np.isin(sol, elementos_a_borrar)]
           
            u=calcdisentrada(sol,tij)#,n)
            #u=np.array(u)
            if (len (sol) > 2) and (np.all(u[:-1] < Tmax)) and (u[-1]<= Tmax):
               
                xtiempogrupot(sol,tij,r,zz,cantidadt,n,u,tiempot,nodos)
                profit_sol=xLS_knapsackORTOOLS(Tmax,p,tij,r,u,zz,cantidadt,n,nodos,sol)
                elementos_a_borrar2 = np.setdiff1d(sol[1:-1], nodos)  
               
                if (np.any(tiempot > Tmax)) or  (len(elementos_a_borrar2) >=1):
                   profit_sol=0
            else:
                profit_sol=0
           
        
        
        
     
        if best_profit < profit_sol:
            best_sol = sol.copy()
            best_profit = profit_sol
           
            MEJORA=MEJORA+1
            
            best_z=zz.copy()
            best_u=u.copy()
            best_tiempot=tiempot.copy()
            best_nodos=nodos.copy()
          
            
     
        if (best_profit - profit_sol) / best_profit > alpha:
            sol = best_sol.copy()
           
            zz=best_z.copy()
            u=best_u.copy()
            tiempot=best_tiempot.copy()
            nodos=best_nodos.copy()
            profit_sol=best_profit
        
        titer=time.time() -tini
        tlim=tlim+titer
        
    end = time.time()
    tiempototal = end-start
    gap=(cota-best_profit )/cota
   
    #print("Profit  : %d" % best_profit)
    #print("Ruta  : ",  best_sol)
    #print("Tiempos de entrada  :", best_u)
    #print("Tiempos de término en cada nodo  :", best_tiempot)
    #print("Tiempo : %f" % tiempototal)
    #print('NODOSP')
    #print(nodos)
    #print("ZZ : ",  best_z[:,best_sol[1:-1],:])
    
   
    return initprofit,best_profit,tiempototal,gap,MEJORA#,best_z








def xtwoopt_v1_fact   (np.ndarray[DTYPE_INT, ndim=1] ruta, np.ndarray[DTYPE_FLOAT, ndim=1] u, np.ndarray[DTYPE_FLOAT, ndim=2] tij,
                       DTYPE_FLOAT Tmax, np.ndarray[DTYPE_FLOAT, ndim=3] rik, np.ndarray[DTYPE_FLOAT, ndim=3] z,
                       DTYPE_INT t, np.ndarray[DTYPE_INT, ndim=1] nodos, np.ndarray[DTYPE_INT, ndim=1] tiempotx): 
                            
    cdef int n = ruta.shape[0]  # Longitud de la ruta
    cdef int i, j               # Variables de iteración
    cdef double ul, sma, sm     # Variables para evaluaciones
    cdef bint imp = True        # Bandera para mejoras
    cdef np.ndarray[DTYPE_FLOAT, ndim=1] ut = np.copy(u)  # Copia del vector u

    ul = u[-1]  # Último elemento de `u`
    sm = np.sum(u)  # Suma total inicial de `u`

    # Inicio del algoritmo 2-opt
    while imp:
        imp = False
        for i in range(n - 2):
            for j in range(i + 2, n - 1):  # Evitar índices adyacentes
                # Revertir segmento de ruta
                ruta[i + 1:j + 1] = ruta[i + 1:j + 1][::-1]
                
                # Llamada a la función `calcdisentrada_2` para calcular distancias
                calcdisentrada_2(ruta, u, tij, i, j)  # Asegúrate de que esta función esté optimizada
                xtiempogrupot(ruta,tij,rik,z,t,n,u,tiempotx,nodos)
                # Nuevos cálculos para evaluar mejora
                sma = np.sum(u)

                # Verificar condiciones de mejora
                
                if (u[-1] < ul and np.all(tiempotx <= Tmax))or  (u[-1] <= ul and sma < sm and np.all(tiempotx <= Tmax)):
              
                    imp = True
                    return  # Salida al encontrar mejora
                else:
                    # Restaurar valores originales si no mejora
                    u[:] = ut
                    ruta[i + 1:j + 1] = ruta[i + 1:j + 1][::-1]

            if imp:
                break  # Salir del bucle externo si hay mejora

    return






def xLS_knapsackORTOOLS(DTYPE_FLOAT tmax,
    np.ndarray[DTYPE_FLOAT, ndim=2] pik,
    np.ndarray[DTYPE_FLOAT, ndim=2] tij,
    np.ndarray[DTYPE_FLOAT, ndim=3] rik,
    np.ndarray[DTYPE_FLOAT, ndim=1] u,
    np.ndarray[DTYPE_FLOAT, ndim=3] zz,
    DTYPE_INT t,
    DTYPE_INT n,
    np.ndarray[DTYPE_INT, ndim=1] nodos,
    np.ndarray[DTYPE_INT, ndim=1] ruta):
    
    
    
    cdef DTYPE_FLOAT total = 0
    cdef int i
    cdef list zsol
    cdef DTYPE_FLOAT profit
    
    
    
    
    
    
    
   
    #total=0
    #print(zz[:,ruta[1:-1],:])
    #print(ruta)
    #print(nodos)
    for i in range(t):# (len(ruta)-2):#(t):
      

      if nodos[i]!=0:
          
          

          solver = knapsack_solver.KnapsackSolver(knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,'a',)
          
            
            
           
          cap= tmax-u[np.where(np.array(ruta)[1:-1]==np.int32(nodos[i]))[0][0]+1]
          #print(cap)
        
          #print(cap,type(cap))
            
          if cap <=0:
            zz[i,np.int32(nodos[i]),:]=0
            profit=0
          else: 
                  cap=[np.int32(cap)]
                  peso=[list(np.int32(rik[i,np.int32(nodos[i]),:]))]
                  ben=list(np.int32(pik[np.int32(nodos[i]),:]))
                  
          
            
                 
                  solver.init(ben, peso, cap)
                  
                  profit = solver.solve()
                  #print(profit)

                  zsol = []
                
                

                  for o in range(len(ben)):
                     if solver.best_solution_contains(o):
                        zsol.append(o)
                  #print(zsol)
                  #print('antes',zz[i,nodos[i],:])
                  zz[i,np.int32(nodos[i]),:]=0
                  
                  zz[i,np.int32(nodos[i]),zsol]=1
                  #print('despues',zz[i,nodos[i],:])
                  if profit <0:
                        profit=0
        
          total=total+profit
    #print(zz[:,ruta[1:-1],:])
    return  total# zz,total


def xcheapest_ins_dist_mod_profit(
    DTYPE_FLOAT tmax,
    np.ndarray[DTYPE_FLOAT, ndim=2] pik,
    np.ndarray[DTYPE_FLOAT, ndim=2] tij,
    np.ndarray[DTYPE_FLOAT, ndim=3] rik,
    np.ndarray[DTYPE_FLOAT, ndim=1] u,
    np.ndarray[DTYPE_FLOAT, ndim=3] zz,
    DTYPE_INT t,
    DTYPE_INT n,
    np.ndarray[DTYPE_INT, ndim=1] ruta,
    np.ndarray[DTYPE_INT, ndim=1] nodos
):

    # Declaración de variables internas
    cdef np.ndarray[DTYPE_FLOAT, ndim=3] zc = zz.copy()
    cdef np.ndarray[DTYPE_FLOAT, ndim=1] ut = u.copy()
    #cdef np.ndarray[np.bool, ndim=1] mascara_nodos
    cdef list nv
    cdef np.ndarray[DTYPE_FLOAT, ndim=1] profitnodo
    cdef int indexnodo_elim, kj, gt, sel, index_sel, minindex, nodo_eliminado
    cdef np.ndarray[DTYPE_INT, ndim=1] ruta_act, ruta_mod
    cdef np.ndarray[DTYPE_FLOAT, ndim=1] dist, profit_sel
    cdef DTYPE_FLOAT ul, sm, sma
    #zc = zz.copy()
    # Obtener nodos no visitados usando máscaras booleanas
    mascara_nodos = np.ones(n, dtype=bool)
    mascara_nodos[ruta] = False
    nv = np.where(mascara_nodos)[0].tolist()  # Convertido a lista para mantener consistencia
   
    # Calcular beneficios por nodo vectorizadamente
    profitnodo = np.sum(zz[:, ruta[1:-1], :] * pik[ruta[1:-1], :], axis=(0, 2))
   
    # Encontrar nodo menos rentable
    indexnodo_elim = np.argmin(profitnodo)
    nodo_eliminado = ruta[indexnodo_elim + 1]
   
    # Actualizar ruta eliminando el nodo
    ruta_act = np.delete(ruta, indexnodo_elim + 1)
   
    # Guardar y actualizar asignaciones
    kj = 1
   
    #temp = zc[:, nodo_eliminado:nodo_eliminado + 1, :].copy().reshape((zc.shape[0], zc.shape[2]))
    temp = zc[:, nodo_eliminado, :].copy()
   
   
    #print(temp)
    gt = np.where(nodos == nodo_eliminado)[0][0]
   
    zc[:, nodo_eliminado, :] = 0
    
    while len(nv) > 0:
        # Calcular ratios beneficio/tiempo vectorizadamente
        #profit_sel = np.array([np.sum(pik[i, :]) / np.sum(rik[gt, i, :]) for i in nv])
        
        
        #print(profit_sel)
        
        profit_sel = np.sum(pik[nv, :], axis=1) / np.sum(rik[gt, nv, :], axis=1)
        
        #print(profit_sel)

       
        # Seleccionar mejor nodo
        index_sel = np.argmax(profit_sel)
        sel = nv[index_sel]
        nv.remove(sel)
      
        # Calcular costos de inserción
        '''
        dist = np.array([
            tij[ruta_act[j], sel] + 
            tij[sel, ruta_act[j+1]] - 
            tij[ruta_act[j], ruta_act[j+1]] 
            for j in range(len(ruta_act)-1)
        ])
        '''
        #print(dist)
        
        dist = tij[ruta_act[:-1], sel] + tij[sel, ruta_act[1:]] - tij[ruta_act[:-1], ruta_act[1:]]
        
        #print(dist)
        
        
        
        
        
        
        # Encontrar mejor posición
        minindex = np.argmin(dist)
        ruta_mod = np.insert(ruta_act, minindex + 1, sel)
        
        # Guardar estado actual
        ul = u[-1]
        sm = np.sum(u)
        
        # Actualizar tiempos de entrada
        selindex = min(indexnodo_elim + 1, minindex + 1)
        calcdisentrada_swap(ruta_mod, u, tij, selindex)
        
        sma = np.sum(u)
        
        # Verificar si la solución es mejor
        if u[-1] < ul or (u[-1] <= ul and sma < sm):
            ruta = ruta_mod
            zc[:, sel, :] = temp
               
            if kj == 1:
                nodos[np.where(nodos == nodo_eliminado)[0][0]] = sel
            
            return ruta, zc
        else:
            u[:] = ut
    
    return ruta, zz



def xcheapest_ins_dist_mod_dist(
    DTYPE_FLOAT tmax,
    np.ndarray[DTYPE_FLOAT, ndim=2] pik,
    np.ndarray[DTYPE_FLOAT, ndim=2] tij,
    np.ndarray[DTYPE_FLOAT, ndim=3] rik,
    np.ndarray[DTYPE_FLOAT, ndim=1] u,
    np.ndarray[DTYPE_FLOAT, ndim=3] zz,
    DTYPE_INT t,
    DTYPE_INT n,
    np.ndarray[DTYPE_INT, ndim=1] ruta,
    np.ndarray[DTYPE_INT, ndim=1] nodos
):
    
    # Declaración de variables internas
    cdef np.ndarray[DTYPE_FLOAT, ndim=3] zc = zz.copy()
    cdef np.ndarray[DTYPE_FLOAT, ndim=1] ut = u.copy()
    #cdef np.ndarray[np.bool, ndim=1] mascara_nodos
    cdef list nv
    cdef np.ndarray[DTYPE_FLOAT, ndim=1] dist1
    cdef int indexnodo_elim, kj, gt, sel, index_sel, minindex, nodo_eliminado
    cdef np.ndarray[DTYPE_INT, ndim=1] ruta_act, ruta_mod
    cdef np.ndarray[DTYPE_FLOAT, ndim=1] dist, profit_sel
    cdef DTYPE_FLOAT ul, sm, sma
    #zc = zz.copy()
    # Obtener nodos no visitados usando máscaras booleanas
    mascara_nodos = np.ones(n, dtype=bool)
    mascara_nodos[ruta] = False
    nv = np.where(mascara_nodos)[0].tolist()  # Convertido a lista para mantener consistencia
   
    # Calcular beneficios por nodo vectorizadamente
    dist1=np.array([-1*(tij[ruta[i-1],ruta[i]]+tij[ruta[i],ruta[i+1]]) for i in range(1,len(ruta)-1)])
    # Encontrar nodo menos rentable
    indexnodo_elim = np.argmin(dist1)
    nodo_eliminado = ruta[indexnodo_elim + 1]
   
    # Actualizar ruta eliminando el nodo
    ruta_act = np.delete(ruta, indexnodo_elim + 1)
   
    # Guardar y actualizar asignaciones
    kj = 1
   
    #temp = zc[:, nodo_eliminado:nodo_eliminado + 1, :].copy().reshape((zc.shape[0], zc.shape[2]))
    temp = zc[:, nodo_eliminado, :].copy()
   
   
    #print(temp)
    gt = np.where(nodos == nodo_eliminado)[0][0]
   
    zc[:, nodo_eliminado, :] = 0
    
    while len(nv) > 0:
        # Calcular ratios beneficio/tiempo vectorizadamente
        #profit_sel = np.array([np.sum(pik[i, :]) / np.sum(rik[gt, i, :]) for i in nv])
        profit_sel = np.sum(pik[nv, :], axis=1) / np.sum(rik[gt, nv, :], axis=1)
        # Seleccionar mejor nodo
        index_sel = np.argmax(profit_sel)
        sel = nv[index_sel]
        nv.remove(sel)
      
        # Calcular costos de inserción
        '''
        dist = np.array([
            tij[ruta_act[j], sel] + 
            tij[sel, ruta_act[j+1]] - 
            tij[ruta_act[j], ruta_act[j+1]] 
            for j in range(len(ruta_act)-1)
        ])
        '''
        dist = tij[ruta_act[:-1], sel] + tij[sel, ruta_act[1:]] - tij[ruta_act[:-1], ruta_act[1:]]
        
        
        # Encontrar mejor posición
        minindex = np.argmin(dist)
        ruta_mod = np.insert(ruta_act, minindex + 1, sel)
        
        # Guardar estado actual
        ul = u[-1]
        sm = np.sum(u)
        
        # Actualizar tiempos de entrada
        selindex = min(indexnodo_elim + 1, minindex + 1)
        calcdisentrada_swap(ruta_mod, u, tij, selindex)
        
        sma = np.sum(u)
        
        # Verificar si la solución es mejor
        if u[-1] < ul or (u[-1] <= ul and sma < sm):
            ruta = ruta_mod
            zc[:, sel, :] = temp
               
            if kj == 1:
                nodos[np.where(nodos == nodo_eliminado)[0][0]] = sel
            
            return ruta, zc
        else:
            u[:] = ut
    
    return ruta, zz

'''
def RUN_ILS_VNP_11(prob,tt,aa,nomb,niter,M,alpha,a1,a2,b1,c0):

    #profits=[]
    ncities1=[]
    #ttime=[]
    ttime= [[] for j in range(M)]
    name=[]
    inipro=[]
    gaps=[[]for j in range(M)]
    profits=[[]for j in range(M)]
    mejoras=[[]for j in range(M)]
    mipsol= [1520,4956,7922,10397,13092,14265,14184,19058,18844,21659]
    #mipsol=[1520,7444,4912,4956,5625,6388,6874,7922,10057,10397,13092,14265,16557,16854,14184,19058,18668,18844,20739,21659]
    #['br17','ft53','ftv33','ftv35','ftv38','ftv44','ftv47','ftv55','ftv64','ftv70','ftv90','ftv100','ftv110','ftv120','kro124p','ftv130','ftv140','ftv150','ftv160','ftv170']
    #['br17','ftv35','ftv55','ftv70','ftv90','ftv100','kro124p','ftv130','ftv150','ftv170']
    #[1520,4956,7922,10397,13092,14265,14184,19058,18844,21659]

    for nn in range(M):
        for i in  range(len(prob)):
                        print('Instancia',str(prob[i]))
                        pik = np.loadtxt('pik'+str(prob[i])+'-'+str(tt)+'.txt') # beneficio con recurso
                        rik = np.loadtxt('rikt'+str(prob[i])+'-'+str(tt)+'.txt') # tiempo que toma realizar actividad
                        tij = np.loadtxt('tij'+str(prob[i])+'.txt') # matriz de tiempos
                        n=np.int32(tij.shape[0])
                        #print('n nodos',n)
                        K=np.int32(pik.shape[1])
                        T=np.int32(rik.shape[0]/n)
                        #print(T)
                        Tmax= np.loadtxt('tmax'+str(prob[i])+'-'+str(aa)+'.txt') # matriz de tiempos
                        Tmax=Tmax/3
                        rik=np.reshape(rik,(T,n,K))
                        tij[tij==0]=1
                        rik[rik==0]=1
                        #agregar también que sea 1 el tiempo en un rikt
                        rng=10+nn#np.random.randint(1,10000)
                        #print('CANGTIDADT',T)
                        #print(nn,profits,profits[nn])
                        inprofit,profit,tiempo,gap,mejora=xILS_11_NP_XX(prob,Tmax,tij,rik,pik,n,K,T,rng,niter,alpha,a1,a2,b1,c0,mipsol[i])
                        print(profit)
                        print(f'{gap*100:.2f}')
                        print(f'{tiempo:.4f}')


                        gaps[nn].append(gap)
                        name.append(str(prob[i]))
                        inipro.append(inprofit)
                        profits[nn].append(profit)
                        #print(profits)
                        ncities1.append(n)

                        #ttime.append(end-start) 
                        ttime[nn].append(tiempo)
                        mejoras[nn].append(mejora)
                        #gc.collect()
                    
   
    
    
    agaps=np.array(gaps)
    meangapprob=np.mean(agaps,axis=0)
    meangap=np.mean(meangapprob)
    profit=np.array(profits).T
    inprofit=np.array(inipro)[:len(prob)]    
    #profitss=np.array(profits)
    Nombre=np.array(name)[:len(prob)]    
    #Tempo=np.array(ttime)
    Tempo=np.array(ttime).T
    ncities=np.array(ncities1)[:len(prob)]
    
    #print(profit.shape)
    #print(inprofit)
    #print(profit)
    #print(Nombre)
    #print(Tempo)
    #print(ncities)
    #print(meangap)
    
    print("Mejor",meangap)
    

    #Res=np.c_[Nombre,ncities,inprofit,profit,Tempo]#,mejoras]
    #writer=pd.ExcelWriter(str(nomb)+'.xlsx')
    #Res=pd.DataFrame(Res)#,columns=Columns)
    #Res.to_excel(writer,'Results')
    #writer.close()     
      
    return 
'''

def RUN_ILS_VNP_11(prob,tt,aa,nomb,niter,M,alpha,a1,a2,b1,c0,timelimit):

    #profits=[]
    ncities1=[]
    #ttime=[]
    ttime= [[] for j in range(M)]
    name=[]
    inipro=[]
    gaps=[]
    profits=[]
    #mejoras=[[]for j in range(M)]
    keys = ['br17','ft53','ftv33','ftv35','ftv38','ftv44','ftv47','ftv55','ftv64','ftv70','ftv90','ftv100','ftv110','ftv120','kro124p','ftv130','ftv140','ftv150','ftv160','ftv170']
    values = [1520,7444,4912,4956,5625,6388,6874,7922,10057,10397,13092,14265,16557,16854,14184,19058,18668,18844,20739,21659]
    mipsol = dict(zip(keys, values))
    #mipsol=[1520,7444,4912,4956,5625,6388,6874,7922,10057,10397,13092,14265,16557,16854,14184,19058,18668,18844,20739,21659]
    for nn in range(M):
        #for i in  range(len(prob)):
                        print('Instancia',str(prob))
                        pik = np.loadtxt('instancias/pik'+str(prob)+'-'+str(tt)+'.txt') # beneficio con recurso
                        rik = np.loadtxt('instancias/rikt'+str(prob)+'-'+str(tt)+'.txt') # tiempo que toma realizar actividad
                        tij = np.loadtxt('instancias/tij'+str(prob)+'.txt') # matriz de tiempos
                        n=np.int32(tij.shape[0])
                        #print('n nodos',n)
                        K=np.int32(pik.shape[1])
                        T=np.int32(rik.shape[0]/n)
                        #print(T)
                        Tmax= np.loadtxt('instancias/tmax'+str(prob)+'-'+str(aa)+'.txt') # matriz de tiempos
                        Tmax=Tmax/3
                        rik=np.reshape(rik,(T,n,K))
                        tij[tij==0]=1
                        rik[rik==0]=1
                        #agregar también que sea 1 el tiempo en un rikt
                        rng=10+nn#np.random.randint(1,10000)
                        #print('CANGTIDADT',T)
                        #print(nn,profits,profits[nn])
                        inprofit,profit,tiempo,gap,mejora=xILS_11_NP_XX(Tmax,tij,rik,pik,n,K,T,rng,niter,alpha,a1,a2,b1,c0,mipsol[str(prob)],timelimit)
                        print(profit)
                        print(f'{gap*100:.2f}')
                        print(f'{tiempo:.4f}')


                        gaps.append(gap)
                        #name.append(str(prob))
                        #inipro.append(inprofit)
                        profits.append(profit)
                        
                        ##print(profits)
                        
                        #ncities1.append(n)

                        ##ttime.append(end-start) 
                        
                        #ttime[nn].append(tiempo)
                        
                        #mejoras[nn].append(mejora)
                        
                        ##gc.collect()
                    
   
    
    
    agaps=np.array(gaps)
    #meangapprob=np.mean(agaps,axis=0)
    meangap=np.mean(agaps)
    #profit=np.array(profits).T
    #inprofit=np.array(inipro)[:len(prob)]    
    ##profitss=np.array(profits)
    #Nombre=np.array(name)[:len(prob)]    
    ##Tempo=np.array(ttime)
    #Tempo=np.array(ttime).T
    #ncities=np.array(ncities1)[:len(prob)]
    
    #print(profit.shape)
    #print(inprofit)
    #print(profit)
    #print(Nombre)
    #print(Tempo)
    #print(ncities)
    #print(meangap)
    
    print("Mejor",meangap)
    #Res=np.c_[Nombre,ncities,inprofit,profit,Tempo]#,mejoras]
   


    #writer=pd.ExcelWriter(str(nomb)+'.xlsx')
    
    ##Columns=['Name','n cities',['Solution']*12,['Time']*12]
    
    #Res=pd.DataFrame(Res)#,columns=Columns)

    #Res.to_excel(writer,'Results')
    
    ##writer.save()    
    
    #writer.close()     
      
    return 



def RUN_ILS_VNP_11_RES(prob,tt,aa,nomb,niter,M,alpha,a1,a2,b1,c0,timelimit):

    #profits=[]
    ncities1=[]
    #ttime=[]
    ttime= [[] for j in range(M)]
    name=[]
    inipro=[]
    gaps=[[]for j in range(M)]
    profits=[[]for j in range(M)]
    mejoras=[[]for j in range(M)]

    keys = ['br17','ft53','ftv33','ftv35','ftv38','ftv44','ftv47','ftv55','ftv64','ftv70','ftv90','ftv100','ftv110','ftv120','kro124p','ftv130','ftv140','ftv150','ftv160','ftv170']
    values = [1520,7444,4912,4956,5625,6388,6874,7922,10057,10397,13092,14265,16557,16854,14184,19058,18668,18844,20739,21659]
    #values2=[1520,7444,4912,4956,5625,6388,6874,7922,10057,10397,10850,13188,15799,15674,14184,16744,19495,19428,20303,20933]
    #values3= [1520,7444,4912,4956,5625,6388,6874,7922,10057,10397,12148,13217,16246,15674,14184,16744,19495,20876,20303,22366]
    
    mipsol = dict(zip(keys, values))
    #mipsol= [1520,4956,7922,10397,13092,14265,14184,19058,18844,21659]
    #mipsol=[1520,7444,4912,4956,5625,6388,6874,7922,10057,10397,13092,14265,16557,16854,14184,19058,18668,18844,20739,21659]
    #['br17','ft53','ftv33','ftv35','ftv38','ftv44','ftv47','ftv55','ftv64','ftv70','ftv90','ftv100','ftv110','ftv120','kro124p','ftv130','ftv140','ftv150','ftv160','ftv170']
    #['br17','ftv35','ftv55','ftv70','ftv90','ftv100','kro124p','ftv130','ftv150','ftv170']
    #[1520,4956,7922,10397,13092,14265,14184,19058,18844,21659]

    for nn in range(M):
        for i in  range(len(prob)):
                        print('Instancia',str(prob[i]))
                        pik = np.loadtxt('instancias/pik'+str(prob[i])+'-'+str(tt)+'.txt') # beneficio con recurso
                        rik = np.loadtxt('instancias/rikt'+str(prob[i])+'-'+str(tt)+'.txt') # tiempo que toma realizar actividad
                        tij = np.loadtxt('instancias/tij'+str(prob[i])+'.txt') # matriz de tiempos
                        n=np.int32(tij.shape[0])
                        #print('n nodos',n)
                        K=np.int32(pik.shape[1])
                        T=np.int32(rik.shape[0]/n)
                        #print(T)
                        Tmax= np.loadtxt('instancias/tmax'+str(prob[i])+'-'+str(aa)+'.txt') # matriz de tiempos
                        Tmax=Tmax/3
                        rik=np.reshape(rik,(T,n,K))
                        tij[tij==0]=1
                        rik[rik==0]=1
                        #agregar también que sea 1 el tiempo en un rikt
                        rng=10+nn#np.random.randint(1,10000)
                        #print('CANGTIDADT',T)
                        #print(nn,profits,profits[nn])
                        #inprofit,profit,tiempo,gap,mejora=xILS_11_NP_XX_ACT(Tmax,tij,rik,pik,n,K,T,rng,niter,alpha,a1,a2,b1,c0,mipsol[str(prob[i])],p1,p2,p3,p4,ls1,ls2,asig,knap,timelimit)
                        inprofit,profit,tiempo,gap,mejora=xILS_11_NP_XX(Tmax,tij,rik,pik,n,K,T,rng,niter,alpha,a1,a2,b1,c0,mipsol[str(prob[i])],timelimit)
                        print(profit)
                        print(f'{gap*100:.2f}')
                        print(f'{tiempo:.4f}')


                        gaps[nn].append(gap)
                        name.append(str(prob[i]))
                        inipro.append(inprofit)
                        profits[nn].append(profit)
                        #print(profits)
                        ncities1.append(n)

                        #ttime.append(end-start) 
                        ttime[nn].append(tiempo)
                        mejoras[nn].append(mejora)
                        #gc.collect()
                    
   
    
    
    agaps=np.array(gaps)
    meangapprob=np.mean(agaps,axis=0)
    meangap=np.mean(meangapprob)
    profit=np.array(profits).T
    inprofit=np.array(inipro)[:len(prob)]    
    #profitss=np.array(profits)
    Nombre=np.array(name)[:len(prob)]    
    #Tempo=np.array(ttime)
    Tempo=np.array(ttime).T
    ncities=np.array(ncities1)[:len(prob)]
    
    #print(profit.shape)
    #print(inprofit)
    #print(profit)
    #print(Nombre)
    #print(Tempo)
    #print(ncities)
    #print(meangap)
    
    print("Mejor",meangap)
    

    Res=np.c_[Nombre,ncities,inprofit,profit,Tempo]#,mejoras]
    #writer=pd.ExcelWriter(str(nomb)+str('PURO: ')+'.xlsx')
    writer=pd.ExcelWriter(str(nomb)+'.xlsx')
    Res=pd.DataFrame(Res)#,columns=Columns)
    Res.to_excel(writer,'Results')
    writer.close()
     
    #print(Res)
    #return 