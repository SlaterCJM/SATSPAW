import sys
import ILSCYTHON_K  # Importa el módulo compilado

'''
pro = str(sys.argv[1])
alpha = float(sys.argv[2])
a1 = float(sys.argv[3])
a2 = float(sys.argv[4])
b1 = float(sys.argv[5])
c0 = float(sys.argv[6])
'''
#ILSCYTHON.RUN_ILS_VNP_11(['br17','ft53','ftv33','ftv35','ftv38','ftv44','ftv47','ftv55','ftv64','ftv70','ftv90','ftv100','ftv110','ftv120','kro124p','ftv130','ftv140','ftv150','ftv160','ftv170'],0.3,3,'ILS-11NP-2025-greedy',10000,1,alpha,a1,a2,b1,c0)


alpha = 0.013
a1 = 0.23
a2 = 0.011
b1 = 0.271
c0 = 0.156

ILSCYTHON_K.RUN_ILS_VNP_11_PK(['br17','ft53','ftv33','ftv35','ftv38','ftv44','ftv47','ftv55','ftv64','ftv70','ftv90','ftv100','ftv110','ftv120','kro124p','ftv130','ftv140','ftv150','ftv160','ftv170'],0.3,3,'ILS-11NP-2025-K=10',10000,10,alpha,a1,a2,b1,c0,10)
ILSCYTHON_K.RUN_ILS_VNP_11_PK(['br17','ft53','ftv33','ftv35','ftv38','ftv44','ftv47','ftv55','ftv64','ftv70','ftv90','ftv100','ftv110','ftv120','kro124p','ftv130','ftv140','ftv150','ftv160','ftv170'],0.3,3,'ILS-11NP-2025-K=15',10000,10,alpha,a1,a2,b1,c0,15)
ILSCYTHON_K.RUN_ILS_VNP_11_PK(['br17','ft53','ftv33','ftv35','ftv38','ftv44','ftv47','ftv55','ftv64','ftv70','ftv90','ftv100','ftv110','ftv120','kro124p','ftv130','ftv140','ftv150','ftv160','ftv170'],0.3,3,'ILS-11NP-2025-K=20',10000,10,alpha,a1,a2,b1,c0,20)