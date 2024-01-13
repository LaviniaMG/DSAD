# Importul bibliotecilor necesare pentru analiza datelor și vizualizare
import numpy as np
import functii as f
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sb
import joblib
import functii as f
import grafice as g

# Încărcarea datelor dintr-un fișier CSV, setând prima coloană ca index
data = pd.read_excel('../dataIN/AgroFood_CO2_emissions.xlsx', index_col=0)

data = data.drop(['Year'], axis=1)

# Crearea directoarelor pentru salvarea rezultatelor și graficelor, dacă nu există deja
PATH_FOR_FILES = Path('../dataOUT/ACP/Files')
PATH_FOR_PLOTS = Path('../dataOUT/ACP/Plots')
PATH_FOR_PLOTS.mkdir(parents=True, exist_ok=True)
PATH_FOR_FILES.mkdir(parents=True, exist_ok=True)

# Înlocuirea valorilor lipsă (NaN) cu media coloanei respective
for entry in data:
    data[entry] = np.where(np.isnan(data[entry]), np.nanmean(data[entry]), data[entry])


# creare lista etichete observatii
numeObs = data.index.values
print(numeObs, type(numeObs))

# extragere lista d evariabile utile
numeVar = data.columns.values[1:]
print(numeVar, type(numeVar))

# nr. de variabile
m = len(numeVar)
print(m)
# nr. de observatii
n = numeObs.shape[0]
print(n)

# extragere matrice observatii-variabile cauzale
X = data[numeVar].values
print(X, X.shape, type(X))

# standardizare matrice variabile cauzale
X_std = f.standardizare(X)
print(X_std.shape, type(X_std))

# salvare matrice standardizata in fisier CSV
X_std_df = pd.DataFrame(data=X_std,
                        index=numeObs,
                        columns=numeVar)
print(X_std_df)
X_std_df.to_csv('../dataOUT/ACP/Files/Xstd.csv')
# #  e la fel ca mai sus :Scalarea datelor pentru a avea o medie de 0 și o deviație standard de 1
# scaled_data = StandardScaler().fit_transform(data)


# Calcularea variației explicate pentru un număr variabil de componente
var_ratios = []
for i in range(1, len(numeVar)):
    pca = PCA(n_components=i)
    pca.fit(X_std)
    var_ratios.append(sum(pca.explained_variance_ratio_))

# Vizualizarea variației explicate în funcție de numărul de componente
plt.plot(range(1, len(numeVar)), var_ratios)
plt.xticks(range(1, len(numeVar)))
plt.tick_params(axis='x', labelsize=8)
plt.grid()
plt.xlabel('Numărul de componente')
plt.ylabel('Rata de variație explicată')
plt.title('Rata de variație explicată în funcție de numărul de componente')
plt.savefig(PATH_FOR_PLOTS / 'varianța_explicată_de_nr_componentelor.png')

# Aplicarea PCA pe datele scalate pentru a obține componentele principale
pca = PCA(n_components=len(numeVar))
pca.fit_transform(X_std)

# Extracția valorilor proprii (alpha)
alpha = pca.explained_variance_
print(alpha)
# creare grafic varianta explicata
g.componentePrincipale(valoriProprii=alpha)
# g.afisare()

# Vizualizarea varianței explicată de fiecare componentă principală individuală
plt.figure()
plt.ylim(-0.5, 4)
plt.tick_params(axis='x', labelsize=6)
plt.plot([f'C{i+1}' for i in range(len(pca.explained_variance_))], pca.explained_variance_, 'bo-')
plt.grid()
plt.xlabel('Componente principale')
plt.ylabel('Valori proprii')
plt.title('Varianța explicată de componentele principale')
plt.axhline(y=1, color='r', linestyle='-')  # Linie orizontală la valoarea proprie de 1
plt.savefig(PATH_FOR_PLOTS / 'varianța_explicată_de_componentele_principale.png')

##SAU
# extragere componente principale
compPrin = pca.components_
componente = ['C'+str(j+1) for j in range(compPrin.shape[1])]
compPrin_df = pd.DataFrame(data=compPrin)
# salvare in fisier CSV
compPrin_df.to_csv('../dataOUT/ACP/Files/CompPrin.csv')
##

# Salvarea factorilor de corelatie (loadings) într-un fișier CSV
PC_df = pd.DataFrame(pca.components_, columns=[f'C{i+1}' for i in range(len(pca.components_))], index=numeVar)
PC_df.to_csv(PATH_FOR_FILES / 'factori_de_corelatie.csv', index_label='Factor de risc')

# Crearea și salvarea unei corelograme pentru factorii de corelatie
plt.figure(figsize=(29,29))
plt.title('Corelograma factorilor de corelatie', fontsize=40, color='k', verticalalignment='bottom')
sb.heatmap(data=PC_df, cmap='bwr', vmin=-1, vmax=1, annot=True)
plt.savefig(PATH_FOR_PLOTS / 'corelograma_factorilor_de_corelatie.png')

##SAU
# extragere factori de corelatie (factor loadings) si salvare
Rxc = pca.components_
Rxc_df = pd.DataFrame(data=Rxc, index=numeVar, columns=componente)
g.corelograma(matrice=Rxc_df, titlu='Corelograma factorilor de corelatie')
#g.afisare()
##

# Salvarea scorurilor componentelor principale (proiecții ale datelor în spațiul PCA) într-un fișier CSV
scores = pca.transform(X_std)
scores_df = pd.DataFrame(scores, columns=[f'C{i+1}' for i in range(len(scores[0]))], index=data.index)
scores_df.to_csv(PATH_FOR_FILES / 'scoruri_în_noul_spațiu.csv', index_label='Țară/Regiuine')



# Calcularea și salvarea calității reprezentării observațiilor într-un fișier CSV
quality_df = pd.DataFrame(np.square(scores), columns=[f'C{i+1}' for i in range(len(scores[0]))], index=data.index)
quality_df = quality_df.div(quality_df.sum(axis=1), axis=0)
quality_df.to_csv(PATH_FOR_FILES / 'calitatea_reprezentării_observațiilor.csv', index_label='Țară/Regiuine')


# Calcularea și salvarea contribuțiilor observațiilor la fiecare componentă într-un fișier CSV
contrib_df = pd.DataFrame(np.square(scores), columns=[f'C{i+1}' for i in range(len(scores[0]))], index=data.index)
contrib_df = contrib_df.div(contrib_df.sum(axis=0), axis=1)
contrib_df.to_csv(PATH_FOR_FILES / 'contribuțiile_observațiilor.csv', index_label='Țară/Regiuine')


# Calcularea și salvarea comunălităților (suma pătratelor factorilor de sarcină pentru fiecare variabilă) într-un fișier CSV
comm_df = pd.DataFrame(np.cumsum(np.square(pca.components_), axis=1), columns=[f'C{i+1}' for i in range(len(scores[0]))], index= numeVar)
comm_df.to_csv(PATH_FOR_FILES / 'comunalități.csv', index_label='Factor de risc')

##SAU
# extragere comunalitati
comun = np.cumsum(np.square(pca.components_), axis=1)
comun_df = pd.DataFrame(data=comun, index=numeVar, columns=componente)
g.harta_intensitate(matrice=comun_df, titlu='Harta comunalitatilor')
#g.afisare()
##

# Crearea și salvarea unei corelograme pentru scorurile componentelor principale
plt.figure(figsize=(50, 50))
plt.title('Corelograma scorurilor', fontsize=40, color='k', verticalalignment='bottom')
plt.ylabel('Țară/Regiune', fontsize=40, color='b', verticalalignment='bottom')
sb.heatmap(data=scores_df, cmap='bwr', vmin=-1, vmax=1, annot=True)
plt.savefig(PATH_FOR_PLOTS / 'corelograma_scorurilor.png')

# Graficul Scree cu "cotul" evidentiat
plt.figure()
plt.plot(range(1, len((numeVar)) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Grafic Scree')
plt.xlabel('Numărul de componente')
plt.ylabel('Varianța cumulativă explicată')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=pca.n_components, color='r', linestyle='--')
plt.grid()
plt.savefig(PATH_FOR_PLOTS / 'scree_plot.png')

def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    plt.scatter(xs * scalex, ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is not None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid()


# Aplicarea biplot-ului
plt.figure(figsize=(12, 8))
biplot(scores[:, 0:2], np.transpose(pca.components_[0:2, :]), labels=numeVar)
plt.savefig(PATH_FOR_PLOTS / 'biplot.png')
