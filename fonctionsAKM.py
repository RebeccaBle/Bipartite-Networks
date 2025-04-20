import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

class graph_AKM:

    def __init__(self, nombre_patient = 1, nombre_docteur = 1):
        self.nombre_patient = nombre_patient
        self.nombre_docteur = nombre_docteur
        self.position_docteur = (np.random.uniform(0,1,size = (self.nombre_docteur)), np.random.uniform(0,1,size = (self.nombre_docteur)))
        self.position_patient = (np.random.uniform(0,1,size = (self.nombre_patient)), np.random.uniform(0,1,size = (self.nombre_patient)))
        self.effet_pat = None
        self.effet_doc = None
        self.alpha = None
        self.psi = None
        self.beta = None
        self.prix = None
        self.beta_lien = None
        self.constente =None
        self.matrice_distance = None
        self.lien = None
    
    def create_link(self, effet_pat, effet_doc, beta_lien ):

        #1 matrice distance

        self.matrice_distance = np.zeros((self.nombre_patient, self.nombre_docteur))
        for j in range(self.nombre_docteur):
            for i in range(self.nombre_patient):
                self.matrice_distance[i,j] = np.sqrt((self.position_patient[1][i]-self.position_docteur[1][j])**2+(self.position_patient[0][i]-self.position_docteur[0][j])**2)

        #2 matrice des liens

        self.beta_lien= beta_lien
        self.effet_doc = effet_doc
        self.effet_pat = effet_pat
        self.lien = np.zeros((self.nombre_patient, self.nombre_docteur))
        for j in range(self.nombre_docteur):
            for i in range(self.nombre_patient):
                lambda_ij = self.beta_lien*self.matrice_distance[i,j]-self.effet_pat[i]+self.effet_doc[j]
                self.lien[i,j] = np.random.binomial(1,1-(1/(1+np.exp(lambda_ij))))

        #3 graph

        plt.figure(figsize=(6, 6))
        plt.scatter(self.position_docteur[0], self.position_docteur[1], color='blue', alpha=0.7, label="Docteur")
        plt.scatter(self.position_patient[0], self.position_patient[1], color='red', alpha=0.7, label="Patient")
        for i in range(self.nombre_patient):
            for j in range(self.nombre_docteur):
                if self.lien[i,j] == 1:
                    plt.plot([self.position_docteur[0][j], self.position_patient[0][i]], [self.position_docteur[1][j], self.position_patient[1][i]], 'k-',alpha=0.5,color="green")
        plt.title("Points aléatoires uniformes dans [0,1]²")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.axis('square')
        plt.legend()
        plt.show()
    
    def show_links(self):
        plt.figure(figsize=(6, 6))
        plt.scatter(self.position_docteur[0], self.position_docteur[1], color='blue', alpha=0.7, label="Docteur")
        plt.scatter(self.position_patient[0], self.position_patient[1], color='red', alpha=0.7, label="Patient")
        for i in range(self.nombre_patient):
            for j in range(self.nombre_docteur):
                if self.lien[i,j] == 1:
                    plt.plot([self.position_docteur[0][j], self.position_patient[0][i]], [self.position_docteur[1][j], self.position_patient[1][i]], 'k-',alpha=0.5,color="green")
        plt.title("Points aléatoires uniformes dans [0,1]²")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.axis('square')
        plt.legend()
        plt.show()

    def solve_model(self, alpha, psi, constente, beta):

        self.alpha = alpha
        self.psi = psi
        self.beta = beta
        self.constente = constente

        #1 Create prices

        self.prix = np.zeros((self.nombre_patient,self.nombre_docteur))
        for j in range(self.nombre_docteur):
            for i in range(self.nombre_patient):
                self.prix[i,j] = self.constente + self.alpha[i] + self.psi[j] + self.beta*self.matrice_distance[i,j]

        #2 Create all matrixes

        X_tild = np.zeros((self.nombre_patient*self.nombre_docteur,2))
        for i in range(self.nombre_patient):
            for j in range(self.nombre_docteur):
                X_tild[i*self.nombre_docteur+j][0]=1
                X_tild[i*self.nombre_docteur+j][1]=self.matrice_distance[i,j]

        A_tild = np.zeros((self.nombre_patient*self.nombre_docteur,self.nombre_patient))
        for i in range(self.nombre_patient*self.nombre_docteur):
                n = i//self.nombre_docteur
                A_tild[i][n]=1
        
        B_tild = np.zeros((self.nombre_patient*self.nombre_docteur,self.nombre_docteur))
        for i in range(self.nombre_patient*self.nombre_docteur):
                n = i%self.nombre_docteur
                if n<self.nombre_docteur:
                        B_tild[i][n]=1
        
        Y_etoile = np.zeros((self.nombre_patient*self.nombre_docteur,1))
        for j in range(self.nombre_docteur):
            for i in range(self.nombre_patient):
                Y_etoile[i*self.nombre_docteur+j][0] = self.prix[i,j]
        
        S = np.zeros((int(self.lien.sum()),self.nombre_patient*self.nombre_docteur))
        l=0
        while l<int(self.lien.sum()):
            for i in range(self.nombre_patient):
                for j in range(self.nombre_docteur):
                    if self.lien[i,j] == 1:
                        S[l][i*self.nombre_docteur+j] = 1
                        l+=1
        
        A = S@A_tild
        B = S@B_tild
        B = B[:,:-1]
        X = S@X_tild
        Y = S@Y_etoile

        G = np.hstack([A, B])
        C = np.eye(int(self.lien.sum()))-G@G.T

        #3 Compute estimates

        beta_chapeau = np.linalg.inv(X.T@C@X)@X.T@C@Y

        effets_fixes = (np.linalg.pinv(G.T@G))@(G.T@(Y-X@beta_chapeau))
        alpha_chapeau = effets_fixes[:self.nombre_patient][:,0]
        psi_chapeau = effets_fixes[self.nombre_patient:]
        psi_chapeau = np.vstack([psi_chapeau,[[0]]])[:,0]

        prix_chapeau = np.zeros((self.nombre_patient,self.nombre_docteur))
        for j in range(self.nombre_docteur):
            for i in range(self.nombre_patient):
                prix_chapeau[i,j] = beta_chapeau[0] + alpha_chapeau[i] + psi_chapeau[j] + beta_chapeau[1]*self.matrice_distance[i,j]
        
        return(beta_chapeau, alpha_chapeau, psi_chapeau, prix_chapeau)
    
    def show_perf(self, alpha, psi, constente, beta):
        beta_chapeau, alpha_chapeau, psi_chapeau, prix_chapeau = self.solve_model(alpha, psi, constente, beta)

        mse_prix = int(((self.prix-prix_chapeau)**2).sum())/(self.nombre_patient*self.nombre_docteur)
        mse_alpha = ((self.alpha-alpha_chapeau[:])**2).sum()/(self.nombre_patient)
        mse_psi = ((self.psi-psi_chapeau[:])**2).sum()/(self.nombre_docteur)

        print(f"Le mse_prix vaut: {mse_prix}")
        print(f"Le mse_alpha vaut: {mse_alpha}")
        print(f"Le mse_psi vaut: {mse_psi}")

        return (mse_prix, mse_alpha, mse_psi)





