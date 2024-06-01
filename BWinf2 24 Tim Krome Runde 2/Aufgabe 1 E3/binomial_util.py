# Diese Datei enthält Hilfsfunktionen zum Berechenen von Binomialverteilungen

import math
import random
import numpy as np

def binomialpdf(*, n, p, k):
    """
    Returns:
        float: P(X=k) mit den Parametern n und p und einer binomialverteilte Größe X
    """
    # Diese Sonderfälle würden wegen der logarithmischen Implementierung der Benoulli-Formel für P(X=k) einen DomainError erzeugen und werden daher seperat abgehandelt:
    if p == 1:
        return n == k
    elif p == 0:
        return n == 0
    # Berechneter Wert wird gecached:
    bincomb = math.comb(n, k)
    if bincomb == 0:
        return 0 # Dieser Fall tritt auf, wenn k > n oder k < 0
    log_binom = math.log(bincomb) + k * math.log(p) + (n - k) * math.log(1 - p)
    return math.exp(log_binom)

def binomialdist(*, n, p, relevant_threshold=0.01):
    """
    Returns:
        list: P(x=k) für alle k und den Parametern n und p und einer binomialverteilten Größe X. Der Index in der Liste korrespondiert mit dem jeweiligen k-Wert
    """
    dist = np.zeros(n+1) # Mit Nullen gefüllte Liste erzugen
    for k in range(math.floor(n*p),-1,-1):
        pdf = binomialpdf(n=n, p=p, k=k)
        dist[k] = pdf
        if pdf < relevant_threshold:
            break
    for k in range(math.ceil(n*p),n+1,1):
        pdf = binomialpdf(n=n, p=p, k=k)
        dist[k] = pdf
        if pdf < relevant_threshold: # der restliche Bereich ist vernachlässigbar, da die Wahrscheinlichkeiten verschwinden gering werden
            break
    if len(dist[dist==0]) != 0:
        fill_rest = (1- sum(dist)) / len(dist[dist==0])
        dist[dist==0] = fill_rest
    return dist

def binomial_likeliest(*, n, p, rank=0, handle_ties="higher"):
    """
    Bestimmt das k, für das P(X=k) maximal wird (also das k, das am wahrscheinlichsten Eintritt).
    Args:
        n (int), p (float): Parameter für die Binomialverteilung
        rank (int oder str): Gibt an, welches bzw. das wieviel-wahrscheinlichste k zurückgegeben werden soll. z.B. wird im Fall rank==0 das wahrscheinlichste k zurückgegeben.
            Wenn rank == "random", dann wird den tatsächlichen Wahrscheinlichkeiten entsprechend ein k zufällig ausgewählt
        handle_ties (str): Entweder "higher", "lower" oder "random". Gibt an, ob im Falle zweier gleich wahrscheinlicher Fälle (tritt auf wenn p=0.5 und floor(n/2) <= k <= ceil(n/2)) das größere oder kleinere k zurückgegeben werden soll (wenn handle_ties=="random", dann wird mit 50%-iger Wahrscheinlichkeit zufällig gewählt).
    Returns:
        int: Das gesuchte k
        float: Die Wahrscheinlichkeit, zu der k eintritt, bzw. P(X=k)
    """
    if rank == "random":
        # Simulation der tatsächlichen Wahrscheinlichkeiten: Ein k wird zufällig ausgewählt mit den Wahrscheinlichkeitswerten der Binomialverteilung
        dist = binomialdist(n=n, p=p)
        k = random.choices(population=[i for i in range(n+1)], k=1, weights=dist)[0]
        return k, dist[k]
    erwartungswert = n*p
    if erwartungswert == int(n*p) and rank == 0:
        # Sonderfall: Wenn der Erwartungswert eine ganze Zahl ist, dann entspricht er definitionsgemäß dem wahrscheinlichsten k
        return int(n*p), binomialpdf(n=n, p=p, k=int(n*p))
    elif erwartungswert != int(n*p) and rank < 2:
        # Wenn das wahrscheinlichste oder zweitwahrscheinlichste k bestimmt werden soll, dann reicht es aus, für k die ganzen Zahlen größer und kleiner E(X) zu überprüfen
        # Denn es gilt definitionsgemäß, dass das wahrscheinlichste k entweder floor(E(X)) oder ceil(E(X)) ist. Die Binomialverteilung muss in diesem Fall nicht berechnet werden.
        # Als Ersatz wird eine mit Nullen gefüllte Liste erstellt und mit den relevanten Werten gefüllt:
        dist = [0]*(n+1)
        dist[int(math.floor(erwartungswert))] = binomialpdf(n=n, p=p, k=int(math.floor(erwartungswert)))
        dist[int(math.ceil(erwartungswert))] = binomialpdf(n=n, p=p, k=int(math.ceil(erwartungswert)))
    else:
        dist = list(binomialdist(n=n, p=p))
    sorted_dist = sorted(dist, reverse=True) # dist absteigend sortieren
    if rank > len(dist)-1:
        rank = len(dist)-1
    probability = sorted_dist[rank] # der sortierten Liste die gesuchte Wahrscheinlichkeit
    k = dist.index(probability) # über den Index der Wahrscheinlichkeit k ermitteln
    # Mit zwei gleich wahrscheinlichen Fällen umgehen:
    dist[k] = None
    if probability in dist and (handle_ties == "higher" or handle_ties == "random" and random.randint(0,1) == 0):
        k = dist.index(probability)
    return k, probability
