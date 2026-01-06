import numpy as np 
import pandas as pd
from docplex.mp.model import Model
import numpy as np
import time
import gc





def ICmodelx(name,n,K,Tmax,M,pik,rik,tij,T,gh):
  
    print('NAME',name)
    m=Model(name=str(name))
    m.set_time_limit(gh)
    m.context.cplex_parameters.threads = 1
    x={ (i+1,j+1) : m.binary_var(name='x_{0}_{1}'.format(i+1,j+1)) for i in range(n) for j in range(n) } 
    z={ (i+1,k+1,t+1) : m.binary_var(name='z_{0}_{1}_{2}'.format(i+1,k+1,t+1)) for i in range(n) for k in range(K) for t in range(T) }#if i!=0}
    u={ (i+1) : m.continuous_var(name='u_{0}'.format(i+1)) for i in range(n) }# if  (j!=0)}
    w={ (i+1,t+1) : m.binary_var(name='w_{0}_{1}'.format(i+1,t+1)) for i in range(n) for t in range (T) }# if  (j!=0)}
   
    fo=m.maximize(m.sum(z[i+1,k+1,t+1]*pik[i,k] for i in range(n) for k in range(K) for t in range(T) if i!=0))
   
    for i in range(n):
        m.add_constraint(m.sum(x[i+1,j+1]   for j in range(n) if (i!=j))-m.sum(x[j+1,i+1] for j in range(n)   if (i!=j)) == 0 )
        
    m.add_constraint(m.sum(x[1,j+1]  for j in range(n)  if j!=0) == 1)
    
    for i in range(n):
        m.add_constraint(m.sum(x[i+1,j+1]   for j in range(n) if (i!=j))<=1)
    
          
    for i in range(1,n):
        
        m.add_constraint(u[i+1]+m.sum(rik[t,i,k]*z[i+1,k+1,t+1] for k in range(K) for t in range(T) ) <=Tmax  )
    
    for i in range(1,n):  
        for j in range(1,n):
            if i != j:
                m.add_constraint(u[i+1]+tij[i,j]*x[i+1,j+1]- (1-x[i+1,j+1])*Tmax   <=u[j+1]  )
                
                

    m.add_constraint(m.sum(x[i+1,j+1]*tij[i,j] for i in range(n) for j in range(n) if (i!=j) ) <=Tmax  )

    
    for i in range(1,n):
        for t in range(T):       
            
            m.add_constraint(m.sum(z[i+1,k+1,t+1] for k in range(K) ) <=K*w[i+1,t+1]  )
        
    for i in range(1,n):       
            
        m.add_constraint(m.sum(w[i+1,t+1] for t in range(T)) <=1  )
            
    for t in range(T):       
            
        m.add_constraint(m.sum(w[i+1,t+1] for i in range(1,n)) <=1  )
                   
    
            
    for i in range(1,n):
       
        m.add_constraint(m.sum(z[i+1,k+1,t+1] for k in range(K) for t in range(T) ) <=K*m.sum(x[j+1,i+1] for j in range(n) if (i!=j) ) )
    
  
    for i in range(1,n):
       
        m.add_constraint(m.sum(z[i+1,k+1,t+1] for k in range(K) for t in range(T)) >=m.sum(x[j+1,i+1] for j in range(n) if j!=i ) ) 
    
    for i in range(1,n):
       
        m.add_constraint(u[i+1] <=  (Tmax -np.min(rik[:,i,:])) *(m.sum(x[j+1,i+1] for j in range(n) if j!=i ))  )
    
    for i in range(1,n):
       
        m.add_constraint(u[i+1] >=  m.sum(x[j+1,i+1]*tij[j,i] for j in range(n) if j!=i ))  
    
    

    #m.export_as_lp(str(name)+'-MTZ.lp')
    m.print_information()
    a=time.time()
    msolution = m.solve(log_output=False)#(log_output=True)
    #print(m.get_solve_status ( ))
    b=time.time()
    
    
    if  str(m.get_solve_status ( )) =='JobSolveStatus.INFEASIBLE_SOLUTION':
        sol=0
        bestbound=0
        mipgap=100
        etime=b-a
        dx=0
        dz=0
        du=0
        dw=0
        print('INFEASIBLE')
    else:
        
        sol=msolution.get_objective_value()
        bestbound=msolution.solve_details.best_bound
        mipgap=msolution.solve_details.mip_relative_gap
        etime=msolution.solve_details.time
        dx=msolution.get_value_dict(x,keep_zeros=True, precision=1e-13)
        dz=msolution.get_value_dict(z,keep_zeros=True, precision=1e-13)
        du=msolution.get_value_dict(u,keep_zeros=True, precision=1e-13)
        dw=msolution.get_value_dict(w,keep_zeros=True, precision=1e-13)
    
       
    
    print('SOL',sol)
    

    
    '''
    #msolution.display()
    print(m.get_solve_status ( ))
    print('SOL',sol)
    print('...X.....')
    print(dx)
    print('...U.....')
    print(du)
    print('...Z.....')
    print(dz)
    print('...W.....')
    print(dw)
    #del msolution,m,x,z,y,v
    #gc.collect()
    '''
    return dx,dz,du,sol,bestbound,mipgap,etime




def ALLRUNX(prob,aa,tt,nomb,gh):
    #print(len(prob))
    xv=[]
    zv=[]
    uv=[]
    solu=[]
    bestb=[]
    mipg=[]
    ttime=[]
    name=[]
    ncities1=[]
    
    
    cnodos=[]
    acts=[]
    recs=[]
    norecs=[]
    presupuesto=[]
    costo=[]
    limtiempo=[]
    pt=[]
    at=[]
    ltour=[]
    
    
    
    
    for i in  range(len(prob)):
        #print('I',i)
        #cij = np.loadtxt('cij'+str(prob[i])+'.txt') #Costos viaje
        #bik = np.loadtxt('bik'+str(prob[i])+'.txt') # beneficio sin recurso
        pik = np.loadtxt('instancias/pik'+str(prob[i])+'-'+str(tt)+'.txt') # beneficio con recurso
        #sik = np.loadtxt('sik'+str(prob[i])+'.txt') # costo de uso recurso
        #eik = np.loadtxt('eik'+str(prob[i])+'.txt') # costo de realizar actividad
        rik = np.loadtxt('instancias/rikt'+str(prob[i])+'-'+str(tt)+'.txt') # tiempo que toma realizar actividad
        tij = np.loadtxt('instancias/tij'+str(prob[i])+'.txt') # matriz de tiempos
        #si = np.loadtxt('sm'+str(prob[i])+'.txt') # vector de inicio en cada nodo
        #fi = np.loadtxt('fm'+str(prob[i])+'.txt') # vector de término en cada nodo
        #l = np.loadtxt('lm'+str(prob[i])+'.txt') # hora que separa el día en 2 segmentos (para la cantidad de tiempo a descansar)
        n=np.int32(tij.shape[0])
        #print('n nodos',n)
        K=np.int32(pik.shape[1])
        #T=3
        T=np.int32(rik.shape[0]/n)
        #Bmax=b*np.sum(sik)
        #Bmax=Bmax.round(0)#+1000
        #cc=np.array([[cij[i,j] for j  in range(n) if j!=i] for i in range(n) ])
        #Cmax=c*np.sum(cc) +c*np.sum(eik)
        #Cmax=Cmax.round(0)#+1000
        #Tmax= opt[i]*t  #sol óptima por un multiplicador (buscar sol óptima en http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
        Tmax= np.loadtxt('instancias/tmax'+str(prob[i])+'-'+str(aa)+'.txt') # matriz de tiempos
        Tmax=Tmax/3
        #print(T,'T')
        rik=np.reshape(rik,(T,n,K))
        #print(n,K,T)
        #M=np.max(tij)
        M=Tmax
        #print('***,M,',M)
        #print('Tmax',Tmax)
        #print(tij[0,1])
        
        tij[tij==0]=1
        rik[rik==0]=1
        x,z,u,sol,bestbound,mip,etime=ICmodelx(prob[i],n,K,Tmax,M,pik,rik,tij,T,gh)
        xv.append(x) 
        zv.append(z)
        uv.append(u)
        solu.append(sol)
        bestb.append(bestbound)
        mipg.append(mip)
        ttime.append(etime)
        name.append(str(prob[i]))
        ncities1.append(n)
        pt.append(tt)
        at.append(aa)
        #pb.append(b)
        #ltour.append(VerToursData(x,n))
        #print('++++++')
        #print(sol)
        #print(bestbound)
        #print(mip)
        #print(etime)
        #print('++++++')

        
        #cn,cac,crec,cnrec,cpr,cco,clim=info(x,z,v,n,cij,eik,tij,rik,K,pik,bik,sik)
        
        
        #cnodos.append(cn)
        #acts.append(cac)
        #recs.append(crec)
        #norecs.append(cnrec)
        #presupuesto.append(cpr/Bmax)
        #costo.append(cco/Cmax)
        #limtiempo.append(clim/Tmax)
        
        #del cij,bik,pik,sik,eik,rik,tij,n,cc,x,z,v,sol
        gc.collect()
      
        
    #'''  
    Nombre=np.array(name)    
    Tempo=np.array(ttime)
    Solutione=np.array(solu)
    Best=np.array(bestb)
    Mipgap=np.array(mipg)
    ncities=np.array(ncities1)
    
    #cantnodos=np.array(cnodos)
    #cantact=np.array(acts)
    #cantrec=np.array(recs)
    #cantnorec=np.array(norecs)
    #presu=np.array(presupuesto)
    #costototal=np.array(costo)
    #tiempolim=np.array(limtiempo)
    
    #paramb=np.array(pb)    
    #paramc=np.array(pc)    
    #paramt=np.array(pt)
    #plentour=np.array(ltour)
    
    
    
    #print(Nombre,Tempo,Solutione,Best,Mipgap,ncities,cantnodos,cantact,cantrec,cantnorec,presu,costototal,tiempolim)
    #Res=np.c_[Nombre,ncities,Solutione,Best,Mipgap,Tempo,plentour,cantnodos,cantact,cantrec,cantnorec,presu,costototal,tiempolim,paramb,paramc,paramt]
    Res=np.c_[Nombre,ncities,Solutione,Best,Mipgap,Tempo]
    #Res=np.c_[Res,tiempofinal]
    #Res=np.c_[Res,cor]
    #Res=np.c_[Res,UB]
    #Res=np.c_[Res,heus]
    #print(xv)
    #print('***')
    #print(zv)
    #print('****')
    #print(uv)
    #writer=pd.ExcelWriter('CORTE1AFULLf-19.xlsx')
    writer=pd.ExcelWriter(str(nomb)+'.xlsx')
    Columns=['Name','n cities','Solution','Best Bound','Mip gap','Time']
    Res=pd.DataFrame(Res,columns=Columns)

    Res.to_excel(writer,'Results')
    #writer.save()     
    writer.close() 
    return #xv,zv,uv#,solu,bestb,mip,ttime 
    #'''
    





if __name__ == "__main__":

    ALLRUNX(['br17','ftv33','ftv35','ftv38','ftv44','ftv47','ft53','ftv55','ftv64','ftv70','ftv90','kro124p','ftv100','ftv110','ftv120','ftv130','ftv140','ftv150','ftv160','ftv170'],3,0.3,'S-ATSPAW-MTZ-18-12',3600)






