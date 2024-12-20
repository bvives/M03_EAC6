"""Aquest Script genera una serie de dades per el seu us en clustersciclistes.py"""
import os
import logging
import csv
import numpy as np

def generar_dataset(num, ind, dicc):
    """
    Genera els temps dels ciclistes de forma aleatòria segons la informació del diccionari.
    num és el número de files/ciclistes a generar.
    ind és l'index inicial/identificador/dorsal del primer ciclista.
    """
    data = []

    for i in range(num):
        # Seleccionem aleatòriament una categoria de ciclista
        categoria = np.random.choice(dicc)
        # Generem temps de pujada i baixada segons la mitjana i desviació estàndard de la categoria
        temps_pujada = np.random.normal(categoria["mu_p"], categoria["sigma"])
        temps_baixada = np.random.normal(categoria["mu_b"], categoria["sigma"])
        total_temps = temps_pujada + temps_baixada
        ciclista = [ind + i, categoria["name"], temps_pujada, temps_baixada, total_temps]
        data.append(ciclista)
        # Mostrar el ciclista generat per consola
        print(f"Ciclista generat: ID={ciclista[0]}, Categoria={ciclista[1]},\
               Temps Pujada={ciclista[2]:.2f}, Temps Baixada={ciclista[3]:.2f},\
                  Total Temps={ciclista[4]:.2f}")

    return data

if __name__ == "__main__":
    STR_CICLISTES = 'data/ciclistes.csv'

    # BEBB: bons escaladors, bons baixadors
    # BEMB: bons escaladors, mal baixadors
    # MEBB: mal escaladors, bons baixadors
    # MEMB: mal escaladors, mal baixadors

    # Port del Cantó (18 Km de pujada, 18 Km de baixada)
    # pujar a 20 Km/h són 54 min = 3240 seg
    # pujar a 14 Km/h són 77 min = 4268 seg
    # baixar a 45 Km/h són 24 min = 1440 seg
    # baixar a 30 Km/h són 36 min = 2160 seg
    MU_P_BE = 3240 # mitjana temps pujada bons escaladors
    MU_P_ME = 4268 # mitjana temps pujada mals escaladors
    MU_B_BB = 1440 # mitjana temps baixada bons baixadors
    MU_B_MB = 2160 # mitjana temps baixada mals baixadors
    SIGMA = 240 # 240 s = 4 min

    dicc = [
        {"name":"BEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name":"BEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA},
        {"name":"MEBB", "mu_p": MU_P_ME, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name":"MEMB", "mu_p": MU_P_ME, "mu_b": MU_B_MB, "sigma": SIGMA}
    ]

    # Generem el dataset
    NUM_CICLISTES = 100  # Per exemple, 100 ciclistes
    IND_INICIAL = 1  # Índex inicial del ciclista
    data = generar_dataset(NUM_CICLISTES, IND_INICIAL, dicc)

    # Assegurem que el directori existeixi
    os.makedirs(os.path.dirname(STR_CICLISTES), exist_ok=True)

    # Escrivim el dataset en un fitxer CSV
    with open(STR_CICLISTES, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Categoria", "Temps Pujada", "Temps Baixada", "Total Temps"])
        writer.writerows(data)

    logging.info("s'ha generat data/ciclistes.csv")
