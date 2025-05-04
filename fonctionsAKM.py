import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.cluster import KMeans

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
        self.std_bruit = None
    
    def create_link(self, effet_pat=None, effet_doc=None, beta_lien=None, show=True):

        #1 matrice distance

        self.matrice_distance = np.zeros((self.nombre_patient, self.nombre_docteur))
        for j in range(self.nombre_docteur):
            for i in range(self.nombre_patient):
                self.matrice_distance[i,j] = np.sqrt((self.position_patient[1][i]-self.position_docteur[1][j])**2+(self.position_patient[0][i]-self.position_docteur[0][j])**2)

        #2 matrice des liens
        if beta_lien is not None:
            self.beta_lien= beta_lien
        if effet_doc is not None:
            self.effet_doc = effet_doc
        if effet_pat is not None:
            self.effet_pat = effet_pat
        self.lien = np.zeros((self.nombre_patient, self.nombre_docteur))
        for j in range(self.nombre_docteur):
            for i in range(self.nombre_patient):
                lambda_ij = -self.beta_lien*self.matrice_distance[i,j]-self.effet_pat[i]+self.effet_doc[j]
                self.lien[i,j] = np.random.binomial(1,1-(1/(1+np.exp(lambda_ij))))

        #3 graph
        if show: 
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

    def solve_model(self, alpha=None, psi=None, constente=None, beta=None, std_bruit= 1):

        if alpha is not None:
            self.alpha = alpha
        if psi is not None:
            self.psi = psi
        if beta is not None:
            self.beta = beta
        if constente is not None:
            self.constente = constente
        if std_bruit is not None:
            self.std_bruit = std_bruit

        #1 Create prices

        self.prix = np.zeros((self.nombre_patient,self.nombre_docteur))
        for j in range(self.nombre_docteur):
            for i in range(self.nombre_patient):
                self.prix[i,j] = self.constente + self.alpha[i] + self.psi[j] + self.beta*self.matrice_distance[i,j] +np.random.normal(0,self.std_bruit)

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
    
    def show_perf(self, alpha, psi, constente, beta, std_bruit= None, show=True):

        if std_bruit is not None:
            self.std_bruit = std_bruit
        else:
            self.std_bruit = 1

        beta_chapeau, alpha_chapeau, psi_chapeau, prix_chapeau = self.solve_model(alpha, psi, constente, beta, self.std_bruit)

        mse_prix = int(((self.prix-prix_chapeau)**2).sum())/(self.nombre_patient*self.nombre_docteur)
        mse_alpha = ((self.alpha-alpha_chapeau[:])**2).sum()/(self.nombre_patient)
        mse_psi = ((self.psi-psi_chapeau[:])**2).sum()/(self.nombre_docteur)

        if show:
            print(f"Le mse_prix vaut: {mse_prix}, il doit normalement valoir la variance du bruit = {self.std_bruit**2}")
            print(f"Le mse_alpha vaut: {mse_alpha}")
            print(f"Le mse_psi vaut: {mse_psi}")

        return (mse_prix, mse_alpha, mse_psi)

    def sparcity(self, beta_lien_min, beta_lien_max, nb_points):
        t=np.linspace(beta_lien_min,beta_lien_max, nb_points)
        x=[]
        y=[]
        z=[]
        s=[]
        for i in tqdm(t):
            self.create_link(self.effet_pat, self.effet_doc, i, show=False)
            perfs=self.show_perf(self.alpha, self.psi, self.constente, self.beta, self.std_bruit, show=False)
            x.append(perfs[0])
            y.append(perfs[1])
            z.append(perfs[2])
            s.append(int(self.lien.sum())/(self.nombre_patient*self.nombre_docteur))
        
        data = list(zip(s, x, y, z))

        # Trier par x
        data_sorted = sorted(data, key=lambda t: t[0])

        # Dézipper
        s, x, y, z = zip(*data_sorted)
        
        plt.plot(s, x, label="mse modèle")
        plt.plot(s, y, label="mse patient")
        plt.plot(s, z, label="mse docteur")
        plt.axhline(y=self.std_bruit**2, linestyle='--', label='Noise Variance')

        # Ajouter des labels et un titre
        plt.xlabel("sparcité")
        plt.ylabel("MSE")
        plt.legend()
        plt.title("Différents MSE")
        plt.grid(True)

        # Afficher le graphique
        plt.show()

    def coclustering(self, alpha=None, psi=None, constente=None, beta=None, nombre_cluster=None, std_bruit= 1, print_reg=False):

        if alpha is not None:
            self.alpha = alpha
        if psi is not None:
            self.psi = psi
        if beta is not None:
            self.beta = beta
        if constente is not None:
            self.constente = constente
        if std_bruit is not None:
            self.std_bruit = std_bruit

        #Calcul du prix selon le modèle

        self.prix = np.zeros((self.nombre_patient,self.nombre_docteur))
        for j in range(self.nombre_docteur):
            for i in range(self.nombre_patient):
                self.prix[i,j] = self.constente + self.alpha[i] + self.psi[j] + self.beta*self.matrice_distance[i,j] +np.random.normal(0,self.std_bruit)

        prix_observe= self.prix*self.lien

        #clustering 

        Moment_1_i = self.lien.sum(axis=1)
        Moment_2_i = (prix_observe).sum(axis=1)

        Moment_1_j = self.lien.sum(axis=0)
        Moment_2_j = (prix_observe).sum(axis=0)

        ecart_type_i = []
        for i in range(self.nombre_patient):
            vecteur_i = []
            for j in range(self.nombre_docteur):
                if prix_observe[i,j] >0:
                    vecteur_i.append(prix_observe[i,j])
            std = np.std(vecteur_i, ddof=1)
            ecart_type_i.append(std)

        ecart_type_j = []
        for j in range(self.nombre_docteur):
            vecteur_j = []
            for i in range(self.nombre_patient):
                if prix_observe[i,j] >0:
                    vecteur_j.append(prix_observe[i,j])
            std = np.std(vecteur_j, ddof=1)
            ecart_type_j.append(std)

        Moment_1_i_ecart = Moment_1_i/np.array(ecart_type_i)
        Moment_2_i_ecart = Moment_2_i/np.array(ecart_type_i)
        Moment_1_j_ecart = Moment_1_j/np.array(ecart_type_j)
        Moment_2_j_ecart = Moment_2_j/np.array(ecart_type_j)

        moments_patients = np.array([[Moment_1_i_ecart[i],Moment_2_i_ecart[i]] for i in range(len(Moment_1_i))])
        kmeans_patients = KMeans(n_clusters=nombre_cluster)
        labels_patients = kmeans_patients.fit_predict(moments_patients)
        moments_docteurs = np.array([[Moment_1_j_ecart[j],Moment_2_j_ecart[j]] for j in range(len(Moment_1_j))])
        kmeans_docteurs = KMeans(n_clusters=nombre_cluster)
        labels_docteurs = kmeans_docteurs.fit_predict(moments_docteurs)

        #création du data-frame

        patient_ids = np.repeat(np.arange(self.nombre_patient), self.nombre_docteur)
        doctor_ids = np.tile(np.arange(self.nombre_docteur), self.nombre_patient)

        df = pd.DataFrame({
            'patient_id': patient_ids,
            'doctor_id': doctor_ids,
            'distance': self.matrice_distance.flatten(),
            'link': self.lien.flatten(),
            'prix': self.prix.flatten(),
            'prix_observe': prix_observe.flatten()
        })

        patient_ids_petit = np.arange(self.nombre_patient)
        patient_frame = pd.DataFrame({
            'patient_id': patient_ids_petit,
            'kmeans_patients': labels_patients
        })

        df = pd.merge(df, patient_frame, on='patient_id', how='left')

        docteur_ids_petit = np.arange(self.nombre_docteur)
        docteur_frame = pd.DataFrame({
            'doctor_id': docteur_ids_petit,
            'kmeans_doctors': labels_docteurs
        })

        df = pd.merge(df, docteur_frame, on='doctor_id', how='left')

        df_kmeans_dummies = pd.get_dummies(df['kmeans_patients'], prefix='cluster_patients')

        df = pd.concat([df, df_kmeans_dummies], axis=1)

        df_kmeans_dummies = pd.get_dummies(df['kmeans_doctors'], prefix='cluster_doctors')

        df = pd.concat([df, df_kmeans_dummies], axis=1)

        # Régressions
        X_bis=pd.concat([df[["distance"]]]+[df[["cluster_patients_"+str(i)]] for i in range(nombre_cluster-1)]+[df[[f"cluster_doctors_"+str(i)]] for i in range(nombre_cluster-1)], axis=1)
        X_bis = sm.add_constant(X_bis)
        d = df["link"]
        X_bis = X_bis.astype(float)
        d = d.astype(float)
        logit_model = sm.Logit(d, X_bis)
        logit_results = logit_model.fit()

        print(f"la valeur de beta lien estimée est :{logit_results.params.iloc[1]}, elle est en réalité de {-self.beta_lien}" )

        df_observed = df[df["link"] == 1]
        X=pd.concat([df_observed[["distance"]]]+[df_observed[["cluster_patients_"+str(i)]] for i in range(nombre_cluster-1)]+[df_observed[["cluster_doctors_"+str(i)]] for i in range(nombre_cluster-1)], axis=1)
        X = sm.add_constant(X)
        y = df_observed["prix"]
        X = X.astype(float)
        y = y.astype(float)
        model = sm.OLS(y, X)              
        results = model.fit()

        print(f"la valeur de beta estimée est :{results.params.iloc[1]}, elle est en réalité de {self.beta}" )

        
        if print_reg:
            print("formation des liens:")
            print(logit_results.summary())
            print("formation du prix")
            print(results.summary())

        #corrélation des résidus

        a=[]
        b=[]
        for i in range(nombre_cluster):
            docids=list(df[df['cluster_doctors_'+str(i)]==True]['doctor_id'])
            patids=list(df[df['cluster_patients_'+str(i)]==True]['patient_id'])
            a.append(np.average([self.effet_doc[i] for i in docids]))
            b.append(np.average([self.effet_pat[i] for i in patids]))
        labels_docteurs_average=[a[i] for i in labels_docteurs]
        labels_patients_average=[b[i] for i in labels_patients]
        print("corrélation des effets fixes des docteurs avec la moyenne du groupe estimé", np.corrcoef(self.effet_doc, labels_docteurs_average)[0, 1])
        print("corrélation des effets fixes des patients avec la moyenne du groupe estimé",np.corrcoef(self.effet_pat, labels_patients_average)[0, 1])

        return(logit_results, results)















