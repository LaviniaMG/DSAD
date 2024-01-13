'''
Clasa care incapsuleaza implementarea modelului de ACP
'''
import numpy as np

class ACP:
    # vom pasa contructorului o matrice X standardizata
    def __init__(self, X):
        self.X = X

        # calcul matrice corelatie pentru X nestandardizat
        self.R = np.corrcoef(self.X, rowvar=False)  # avem variabilele pe coloane

        # standardizare valori X
        medii = np.mean(self.X, axis=0)  # variabilele se gasesc pe coloane
        abateri = np.std(self.X, axis=0)  # calcul pe coloane
        self.Xstd = (self.X - medii) / abateri

        # calcul matrice de varianta-covarianta pentru X
        self.Cov = np.cov(m=X, rowvar=False)  # variabilele sunt pe coloane
        print(self.Cov.shape)
        # calcul valori proprii si vectori proprii pentru matricea de varianta-covarianta
        self.valoriProprii, self.vectoriiProprii = np.linalg.eigh(a=self.Cov)
        print(self.valoriProprii, self.valoriProprii.shape)
        print(self.vectoriiProprii.shape)
        # sortare descrescatoare valori proprii si vectori proprii
        k_desc = [k for k in reversed(np.argsort(self.valoriProprii))]
        print(k_desc)
        self.alpha = self.valoriProprii[k_desc]
        self.A = self.vectoriiProprii[:, k_desc]
        # regularizare vectorilor proprii
        for j in range(self.A.shape[1]):
            minCol = np.min(a=self.A[:, j])
            maxCol = np.max(a=self.A[:, j])
            if np.abs(minCol) > np.abs(maxCol):
                self.A[:, j] = (-1) * self.A[:, j]

        # calcul componente principale
        self.C = self.X @ self.A
        # self.C = np.matmul(self.X, self.A)  # alternativa
        # calcul corelatie dintre variabilele observate si componentele principale
        # factor loadings
        self.Rxc = self.A * np.sqrt(self.alpha)

        self.C2 = self.C * self.C
        # self.C2 = np.square(self.C)

        # calcul calitatii reprezentarii observatilor pe axele componentelor principale
        SL = np.sum(self.C2, axis=1)  # sume pe linii
        self.CalObs= np.transpose(self.C2.T / SL)

        # contributia observatiilor la varianta componentelor principale
        self.betha = self.C2 / (self.alpha * self.X.shape[0])

        # calcul comunalitati (regasirea componetelor principale in variabilele initiale)
        Rxc2 = self.Rxc * self.Rxc
        self.Comun = np.cumsum(Rxc2, axis=1)  # sume cumulative pe linii, pentru fiecare variabila observata

    def getAlpha(self):
        # return self.valoriProprii
        return self.alpha

    def getA(self):
        # return self.vectoriiProprii
        return self.A

    def getCompPrin(self):
        return self.C

    def getFactorLoadings(self):
        return self.Rxc
    def getScoruri(self):
        # calcul scoruri
        return self.C / np.sqrt(self.alpha)
    def getCalObs(self):
        return self.CalObs

    # calcul contributie observatii la explicarea variantei axelor componentelor principale
    def getContrib(self):
        return self.C2 / (self.X.shape[0] * self.alpha)

    def getComun(self):
        Rxc2 = np.square(self.Rxc)
        return np.cumsum(a=Rxc2, axis=1)  # sume pe linii

    #Functii folosite pentru AEF

    def getCorr(self):
        return self.R
    def getXstd(self):
        return self.Xstd

    def getValProp(self):
        return self.alpha

    def getRxc(self):
        return self.Rxc

    def getBetha(self):
        return self.betha