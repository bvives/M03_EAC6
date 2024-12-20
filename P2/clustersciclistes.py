

"""
@ IOC - CE IABD
"""
import os
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

def load_dataset(path):
    """
    Carrega el dataset de registres dels ciclistes.

    Arguments:
        path -- dataset

    Returns: dataframe
    """
    try:
        dataset = pd.read_csv(path)
        logging.info(f"Dataset carregat correctament des de {path}")
        return dataset
    except FileNotFoundError as e:
        logging.error(f"El fitxer {path} no s'ha trobat: {e}")
    except pd.errors.EmptyDataError as e:
        logging.error(f"El fitxer {path} està buit: {e}")
    return None


def e_d_a(data_frame):
    """
    Exploratory Data Analysis del dataframe.

    Arguments:
        df -- dataframe

    Returns: None
    """
    # Descripció del dataframe
    print("Descripció del DataFrame:")
    print(data_frame.describe())

    # Visualització de les primeres files
    print("\nPrimeres files del DataFrame:")
    print(data_frame.head())

    # Inspecció de valors nuls
    print("\nValors nuls en el DataFrame:")
    print(data_frame.isnull().sum())

    # Exemple de visualització 1: histograma de 'tp'
    plt.figure(figsize=(10, 6))
    sns.histplot(data_frame['Temps Pujada'], bins=30, kde=True)
    plt.title('Histograma de Temps de Pujada (tp)')
    plt.xlabel('Temps de Pujada (tp)')
    plt.ylabel('Freqüència')
    plt.savefig('img/EDA_histograma_tp.png')  # Guardar la imatge
    plt.close()

    # Exemple de visualització 2: histograma de 'tb'
    plt.figure(figsize=(10, 6))
    sns.histplot(data_frame['Temps Baixada'], bins=30, kde=True)
    plt.title('Histograma de Temps de Baixada (tb)')
    plt.xlabel('Temps de Baixada (tb)')
    plt.ylabel('Freqüència')
    plt.savefig('img/EDA_histograma_tb.png')  # Guardar la imatge
    plt.close()

    # Exemple de visualització 3: scatterplot de 'tp' vs 'tb'
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_frame['Temps Pujada'], y=data_frame['Temps Baixada'])
    plt.title('Scatterplot de Temps de Pujada (tp) vs Temps de Baixada (tb)')
    plt.xlabel('Temps de Pujada (tp)')
    plt.ylabel('Temps de Baixada (tb)')
    plt.savefig('img/EDA_scatterplot_tp_vs_tb.png')  # Guardar la imatge
    plt.close()

    # Gràfica de distribució de la categoria (si hi ha una columna 'Categoria')
    if 'Categoria' in data_frame.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Categoria', data=data_frame)
        plt.title('Distribució de la Categoria')
        plt.xlabel('Categoria')
        plt.ylabel('Freqüència')
        plt.savefig('img/EDA_distribucio_categoria.png')  # Guardar la imatge
        plt.close()

    logging.info("Anàlisi exploratòria completada.")


def clean(dataframe):
    """
    Elimina les columnes que no són necessàries per a l'anàlisi dels clústers.

    Arguments:
        df -- dataframe

    Returns: dataframe
    """
    # Elimina la columna 'ID' si existeix
    if 'ID' in dataframe.columns:
        dataframe = dataframe.drop(columns=['ID'])
        logging.info("La columna 'ID' s'ha eliminat del dataframe.")

    # Eliminar la columna 'tt' (temps total) si existeix
    if 'Total Temps' in dataframe.columns:
        dataframe = dataframe.drop(columns=['Total Temps'])
        logging.info("Columna 'Total Temps' eliminada.")

    return dataframe

def extract_true_labels(dataframe):
    """
    Guardem les etiquetes dels ciclistes (BEBB, ...)
    i eliminem la columna 'Categoria' del dataframe.

    Arguments:
        df -- dataframe

    Returns: numpy ndarray (true labels), dataframe
    """
    if 'Categoria' in dataframe.columns:
        labels = dataframe['Categoria'].values
        dataframe = dataframe.drop(columns=['Categoria'])
        logging.info("Les etiquetes s'han extret \
                      i la columna 'Categoria' s'ha eliminat del dataframe.")
        return labels, dataframe

    logging.error("La columna 'Categoria' no existeix en el dataframe.")
    # Retorna un array buit i el dataframe original si no hi ha la columna 'Categoria'
    return np.array([]), dataframe


def visualitzar_pairplot(dataframe):
    """
    Genera una imatge combinant entre sí tots els parells d'atributs.
    Serveix per apreciar si es podran trobar clústers.

    Arguments:
        df -- dataframe

    Returns: None
    """
    # Generar el pairplot
    pairplot = sns.pairplot(dataframe, diag_kind='kde')

    # Assegurar que el directori 'img' existeix
    os.makedirs('img', exist_ok=True)

    # Guardar la imatge
    pairplot.savefig('img/pairplot.png')

    logging.info("Pairplot generat i guardat correctament a 'img/pairplot.png'.")


def clustering_kmeans(data, n_clusters=4):
    """
    Crea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions).
    Entrena el model i guarda'l en un fitxer.

    Arguments:
        data -- les dades: tp i tb
        n_clusters -- nombre de clústers (per defecte 4)

    Returns: model (objecte KMeans)
    """
    # Creació del model KMeans amb el nombre de clústers especificat
    model = KMeans(n_clusters=n_clusters, random_state=42)

    # Entrenament del model amb les dades
    model.fit(data)

    # Guardar el model en un fitxer pickle
    os.makedirs('model', exist_ok=True)
    with open('model/clustering_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    logging.info(f"Model KMeans creat, entrenat amb {n_clusters} clústers \
                 i guardat en model/clustering_model.pkl.")

    return model



def visualitzar_clusters(data, labels):
    """
    Visualitza els clusters en diferents colors.
    Provem diferents combinacions de parells d'atributs.

    Arguments:
        data -- el dataset sobre el qual hem entrenat
        labels -- l'array d'etiquetes a què pertanyen les dades
        (hem assignat les dades a un dels 4 clústers)

    Returns: None
    """
    # Convertir les dades a un DataFrame per facilitar la visualització
    df = pd.DataFrame(data, columns=['Temps Pujada', 'Temps Baixada'])

    # Afegir les etiquetes com una nova columna
    df['Cluster'] = labels

    # Visualitzar els parells d'atributs
    sns.pairplot(df, hue='Cluster', palette='Set1')

    # Assegurar que el directori 'img' existeixi
    os.makedirs('img', exist_ok=True)

    # Guardar la imatge
    plt.savefig('img/clusters_pairplot.png')

    plt.show()

    logging.info("Visualització dels clústers generada \
                    i guardada correctament a 'img/clusters_pairplot.png'.")


def associar_clusters_patrons(tipus, model):
    """
    Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).
    S'han trobat 4 clústers però aquesta associació encara no s'ha fet.

    Arguments:
        tipus -- un array de tipus de patrons que volem actualitzar associant els labels
        model -- model KMeans entrenat

    Returns: array de diccionaris amb l'assignació dels tipus als labels
    """
    # proposta de solució

    dicc = {'tp':0, 'tb': 1}

    logging.info('Centres:')
    for j in range(len(tipus)):
        logging.info(f"{j}:\t(tp: {model.cluster_centers_[j][dicc['tp']]:.1f}\
                     \ttb: {model.cluster_centers_[j][dicc['tb']]:.1f})")

    # Procés d'assignació
    ind_label_0 = -1
    ind_label_1 = -1
    ind_label_2 = -1
    ind_label_3 = -1

    suma_max = 0
    suma_min = 50000

    for j, center in enumerate(model.cluster_centers_):
        suma = round(center[dicc['tp']], 1) + round(center[dicc['tb']], 1)
        if suma_max < suma:
            suma_max = suma
            ind_label_3 = j
        if suma_min > suma:
            suma_min = suma
            ind_label_0 = j

    tipus[0].update({'label': ind_label_0})
    tipus[3].update({'label': ind_label_3})

    lst = [0, 1, 2, 3]
    lst.remove(ind_label_0)
    lst.remove(ind_label_3)

    if model.cluster_centers_[lst[0]][0] < model.cluster_centers_[lst[1]][0]:
        ind_label_1 = lst[0]
        ind_label_2 = lst[1]
    else:
        ind_label_1 = lst[1]
        ind_label_2 = lst[0]

    tipus[1].update({'label': ind_label_1})
    tipus[2].update({'label': ind_label_2})

    logging.info('\nHem fet l\'associació')
    logging.info('\nTipus i labels:\n%s', tipus)

    # Guardar l'associació en un fitxer pickle
    os.makedirs('model', exist_ok=True)
    with open('model/tipus_dict.pkl', 'wb') as f:
        pickle.dump(tipus, f)

    logging.info('Associació guardada en model/tipus_dict.pkl')

    return tipus


def generar_informes(df, tipus):
    """
    Generació dels informes a la carpeta informes/.
    Tenim un dataset de ciclistes i 4 clústers, i generem
    4 fitxers de ciclistes per cadascun dels clústers.

    Arguments:
        df -- dataframe
        tipus -- objecte que associa els patrons de comportament amb els labels dels clústers

    Returns: None
    """
    # Assegurem que el directori 'informes' existeixi
    os.makedirs('informes', exist_ok=True)

    # Generem els informes per cada clúster
    for tipus_patro in tipus:
        label = tipus_patro['label']
        nom_fitxer = f"informes/{tipus_patro['name']}.csv"
        df_cluster = df[df['label'] == label]
        df_cluster.to_csv(nom_fitxer, index=False)
        logging.info(f"Informe generat per al clúster {label}: {nom_fitxer}")

    logging.info("S'han generat els informes en la carpeta informes/\n")


def nova_prediccio(dades, model):
    """
    Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers.

    Arguments:
        dades -- llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
        model -- clustering model

    Returns: (dades agrupades, prediccions del model)
    """
    # Convertir les dades a un DataFrame per facilitar la predicció
    df_nous_ciclistes = pd.DataFrame(dades, columns=
                                        ['ID', 'Temps Pujada', 'Temps Baixada', 'Total Temps'])

    # Predir els clústers amb el model KMeans
    prediccions = model.predict(df_nous_ciclistes[['Temps Pujada', 'Temps Baixada']])

    # Afegir les prediccions com una nova columna al DataFrame
    df_nous_ciclistes['Cluster'] = prediccions

    # Guardar les dades agrupades per clúster
    dades_agrupades = []
    for cluster in range(model.n_clusters):
        dades_agrupades.append(df_nous_ciclistes[df_nous_ciclistes['Cluster'] == cluster])

    logging.info("Les noves prediccions s'han assignat correctament als clústers.")

    return dades_agrupades, prediccions

# ----------------------------------------------

if __name__ == "__main__":

    PATH_DATASET = './data/ciclistes.csv'

    # Carregar el dataset com una llista de llistes

    # Carregar el dataset
    df = load_dataset(PATH_DATASET)
    logging.info("Dataset carregat correctament.")
    print("Dataset carregat correctament.")

    # Anàlisi Exploratori de Dades (EDA)
    e_d_a(df)

    # Natejem el dataframe:
    df = clean(df)
    logging.info("Dataframe netejat correctament.")
    print("Dataframe netejat correctament.")

    # Extreure les etiquetes dels ciclistes i eliminar la columna 'Categoria'
    true_labels, df = extract_true_labels(df)
    logging.info("Etiquetes extretes i columna 'Categoria' eliminada.")
    print("Etiquetes extretes i columna 'Categoria' eliminada.")

    # Visualització de pairplot per veure possibles clústers
    visualitzar_pairplot(df)
    logging.info("Pairplot generat i guardat correctament.")
    print("Pairplot generat i guardat correctament.")

    # Entrenar el model KMeans i guardar-lo
    trained_model = clustering_kmeans(df)
    logging.info("Model KMeans entrenat i guardat correctament.")
    print("Model KMeans entrenat i guardat correctament.")

    # Associar els clústers als patrons de comportament
    tipus = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]
    tipus = associar_clusters_patrons(tipus, trained_model)
    logging.info("Clústers associats als patrons de comportament.")
    print("Clústers associats als patrons de comportament.")

    # Predir les etiquetes dels clústers amb el model KMeans
    cluster_labels = trained_model.predict(df)

    # Visualitzar els clústers
    visualitzar_clusters(df, cluster_labels)
    logging.info("Visualització dels clústers generada \
                 i guardada a 'img/clusters_visualization.png'.")
    print("Visualització dels clústers generada \
          i guardada a 'img/clusters_visualization.png'.")

    # Afegir la columna 'label' al dataframe amb les etiquetes dels clústers
    df['label'] = cluster_labels
    logging.info("Columna 'label' afegida al dataframe amb les etiquetes dels clústers.")
    print("Columna 'label' afegida al dataframe amb les etiquetes dels clústers.")

    # Mapejar les etiquetes numèriques a etiquetes textuals utilitzant el diccionari de tipus
    label_mapping = {t['label']: t['name'] for t in tipus}
    mapped_cluster_labels = [label_mapping[label] for label in cluster_labels]

    # Calcular els scores de homogeneïtat, completitud i mesura-V
    homogeneity = homogeneity_score(true_labels, mapped_cluster_labels)
    completeness = completeness_score(true_labels, mapped_cluster_labels)
    v_measure = v_measure_score(true_labels, mapped_cluster_labels)

    logging.info(f"Homogeneity Score: {homogeneity}")
    logging.info(f"Completeness Score: {completeness}")
    logging.info(f"V-Measure Score: {v_measure}")

    print(f"Homogeneity Score: {homogeneity}")
    print(f"Completeness Score: {completeness}")
    print(f"V-Measure Score: {v_measure}")

    # Guardar els scores en un fitxer pkl
    scores_dict = {
        'Homogeneity Score': homogeneity,
        'Completeness Score': completeness,
        'V-Measure Score': v_measure
    }
    with open('model/scores.pkl', 'wb') as f:
        pickle.dump(scores_dict, f)

    logging.info("Scores guardats correctament en model/scores.txt.")
    print("Scores guardats correctament en model/scores.txt.")

    # Generar informes dels ciclistes agrupats per clústers
    generar_informes(df, tipus)
    logging.info("Informes generats i guardats a la carpeta informes/")
    print("Informes generats i guardats a la carpeta informes/")

    # Classificació de nous valors
    nous_ciclistes = [
        [500, 3230, 1430, 4670], # BEBB
        [501, 3300, 2120, 5420], # BEMB
        [502, 4010, 1510, 5520], # MEBB
        [503, 4350, 2200, 6550] # MEMB
    ]

    # Predir els clústers per als nous ciclistes
    dades_agrupades, prediccions = nova_prediccio(nous_ciclistes, trained_model)

    # Mapejar les prediccions a etiquetes textuals
    prediccions_mapejades = [label_mapping[label] for label in prediccions]

    # Mostrar les prediccions
    for i, pred in enumerate(prediccions_mapejades):
        print(f"Ciclista {nous_ciclistes[i][0]} classificat com {pred}")

    logging.info("Classificació de nous ciclistes completada.")
    print("Classificació de nous ciclistes completada.")
    