#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#import utile as utl
import AEF as aef
import ACP as acp
#import factor_analyzer as fa
import grafice as g
from sklearn.preprocessing import StandardScaler


tabel = pd.read_csv('../dataIN/AgroFood_CO2_emissions.csv', index_col=0, na_values=':', encoding='latin1')
print(tabel)
tabel = tabel.drop(['Year'], axis=1)

obsNume = tabel.index.values
varNume = tabel.columns.values
X = tabel.values

# Înlocuirea valorilor 0 cu NaN pentru a putea calcula media fără a include 0
X = tabel.replace(0, pd.NA)

# Calcularea mediei pentru fiecare coloană
media_col = X.mean()
X.fillna(X.mean(), inplace=True)
# inlocuire celule NAN
#X = utl.inlocuireNAN(matrice_numerica)
X_df = pd.DataFrame(data=X, index=obsNume, columns=varNume)
X_df.to_csv('../dataOUT/AEF/Files/X.csv')

# verificare test de sfericitate Barlett
sfericitateBartlett = fa.calculate_bartlett_sphericity(X_df)
print(sfericitateBartlett, type(sfericitateBartlett))
if sfericitateBartlett[0] > sfericitateBartlett[1]:
    print('Exista cel putin un factor comun!')
else:
    print('Nu exista factori comuni!')
    exit(-1)

# calcul indici de factorabilitate Kaiser-Meyer-Olkin (KMO)
kmo = fa.calculate_kmo(X_df)
print(kmo, type(kmo))
if kmo[1] > 0.5:
    print('Variabile initiale pot fi exprimate prin cel putin un factor comun!')
else:
    print('Variabilele observate nu pot fi exprimate prin factori')
    exit(-2)

# corelograma indicilor KMO
vector = kmo[0]
print(vector, type(vector))
matrice = vector[:, np.newaxis]
print(matrice)
matrice_df = pd.DataFrame(data=matrice, index=varNume,
                          columns=['Indici KMO'])
matrice_df.to_csv('./dataOUT/AEF/Files/KMO.csv')
g.harta_intensitate(matrice=matrice_df, titlu='Indicii KMO')
# g.afisare()

# extragere de factori comuni
nrFactoriSemnificativi = 1
chi2TabMin = 1
for k in range(1, varNume.shape[0]):
# for k in range(1, 5):
    modelFA = fa.FactorAnalyzer(n_factors=k)
    modelFA.fit(X_df)
    factorLoadings = modelFA.loadings_  # furnizeaza factorii de corelatie
    print(factorLoadings)
    factoriSpecifici =modelFA.get_uniquenesses()  # furnizeaza factorii specifici
    print(factoriSpecifici)
    modelAEF = aef.AEF(X)
    chi2Calc, chi2Tab = modelAEF.calculTestBartlett(factorLoadings, factoriSpecifici)
    print(chi2Calc, chi2Tab)

    if np.isnan(chi2Calc) or np.isnan(chi2Tab):
        break
    if chi2Tab < chi2TabMin:
        chi2TabMin = chi2Tab
        nrFactoriSemnificativi = k

print('Nr. factori semnificativi determinati: ', nrFactoriSemnificativi)

# creare model FA cu numarul semnificativ de factori determinat
modelFitFA = fa.FactorAnalyzer(n_factors=nrFactoriSemnificativi)
modelFitFA.fit(X_df)
factorLoadingsFA = modelFitFA.loadings_  # extragre factori comuni
valPropFA = modelFitFA.get_eigenvalues()  # valori proprii furnizate de FA
print(valPropFA, type(valPropFA))

# realizare grafic valori propii model initial
g.componentePrincipale(valoriProprii=valPropFA[0],
                       titlu='Varinta explicata de valorile proprii ale modelului initial')

# realizare grafic valori proprii model cu 5 factori
g.componentePrincipale(valoriProprii=valPropFA[1],
                       titlu='Varianta explicata de valorile proprii ale modelului cu 5 factori comuni')
#g.afisare()

# corelograma factorilor de corelatie pentru modelul cu 5 factori
factori = ['F'+str(j+1) for j in range(nrFactoriSemnificativi)]
factorLoadingsFA_df = pd.DataFrame(data=factorLoadingsFA,
                                   index=varNume, columns=factori)
# salvare factor loadings in fisier CSV
factorLoadingsFA_df.to_csv('./dataOUT/AEF/Files/FactorLoadingsFA.csv')
g.corelograma(matrice=factorLoadingsFA_df, titlu='Corelegrama factorilor de corelatie din FA')
#g.afisare()

# obtinere factor loading si scoruri din ACP pe modelul initial
modelACP = acp.ACP(X)
factorLoadingsACP = modelACP.getRxc()
valPropACP = modelACP.getValProp()

# grafic valori proprii ACP
g.componentePrincipale(valoriProprii=valPropACP, titlu='Varianta explicata de valorile proprii ale modelului ACP')
#g.afisare()

# grafic factori de corelatie din ACP
componente = ['C'+str(j+1) for j in range(len(varNume))]
factorLoadingsACP_df = pd.DataFrame(data=factorLoadingsACP,
                                    index=varNume,
                                    columns=componente)
# salvare in fisier CSV
g.corelograma(matrice=factorLoadingsACP_df, titlu='Corelograma factorilor de corelatie din ACP')
#g.afisare()

# extragere scoruri din ACP
scoruriACP = modelACP.getScoruri()  # componnetele principale standardizate
scoruriACP_df = pd.DataFrame(data=scoruriACP, index=obsNume,
                             columns=componente)
g.harta_intensitate(matrice=scoruriACP_df, titlu='Scorurile din ACP')
#g.afisare()

# cerc al corelatiilor pe matricea de scoruri
g.cerculCorelatiilor(matrice=scoruriACP_df, titlu='Distributia observatiilor in spatiul componentelor C1 si C2')

# cerc al corelatiilor pe matricea de factor loadings din FA
if factorLoadingsFA_df.shape[1] < 2:
    print("Nu sunt suficiente coloane în DataFrame pentru a genera cercul corelatiilor.")
    print(factorLoadingsFA_df)
else:
    g.cerculCorelatiilor(matrice=factorLoadingsFA_df, titlu='Distributia variabilelor initiale in spatiul factorilor F1 si F2')
    g.afisare()
