"""
Aufgabe: 1 (ohne Erweiterung)
Ansatz: Heuristisches Vorgehen (1. Ansatz / Trivialer Ansatz)
Beschreibung des Verfahrens: Abwechselnd ... (bis eine Abbruchbedingung erfüllt ist)
1. Leeren der Ränder mit systematischem Verfahren
2. Blasen der Blätter Richtung Mitte mit Greedy-Algorithmus, der basierend auf zwei Heuristiken die nächste Blasoperation auswählt.
Bei diesen Heuristiken handelt es sich um ...
- die durchschnittliche Blattdistanz zu Feld Q: Durchschnittlicher Manhattan-Abstand eines Blattes zu Feld Q. Maß für die Nähe der Blätter zu Feld Q.
Bei einem Schulhof, auf dem das Laub bereits nahe an Q ist, kann es schneller auf Q gebracht werden.
- die quadr. Standardabweichung (Varianz) der Blattdistanzen zu Feld Q: Maß für die Streuung / Unordnung der Blätter auf dem Schulhof.
Bei einem geordneten Schulhof kann das Laub schneller auf Q gebracht werden.
Der Greedy-Algorithmus (2.) verwendet ein gewichtetes Produkt dieser beiden Größen, um die nächste Blasoperation zu ermitteln.
Steuerung über die Gewichte möglich:
- weight_avg (Gewicht der durchschnittl. Blattdistanz zu Q):
Erhöhung sorgt dafür, dass schneller Laub auf Q gelangt, bei einer gegen unendlich gehenden Schrittanzahl gelangt aber insgesamt wenig Laub auf Q.
- weight_varianz (Gewicht der Varianz der Blattdistanzen):
Erhöhung sorgt dafür, dass insgesamt mehr Laub auf Q gelangt, das Laub gelangt dafür aber langsamer auf Q.
Dies ist NICHT meine beste Lösung. Bessere Lösung: Siehe aufgabe1_solver1.py.
"""
import numpy as np
import math
from hof import Rules, Hof

# Hof-Eigenschaften festlegen:
Q = (3,4) # Index von Feld Q festlegen
hof_size = (7,7) # Hofseitenlängen festlegen
startwert = 100 # Anfangsanzahl an Blättern pro Feld
# Wahrscheinlichkeitsmodellierung festlegen:
use_binomial = True # Festlegen, ob die Wahrscheinlichkeiten basierend auf der Binomialverteilung simuliert oder ob die Erwartungswerte verwendet werden sollen
weight_avg = 0.1 # Gewichtung des 1. Heuristik-Maßes (durchschnittlicher Laubabstand zu Feld Q)
weight_varianz = 0.9 # Gewichtung des 2. Heuristik-Maßes (Varianz der Laubabstände zu Feld Q)
# Abbruchbedingungen feslegen:
satisfied_constraint = 0.8 # Bei erreichen dieser prozentualen Laubmenge (im Verhältnis zum Gesamtlaub) wird das Programm auf jeden Fall abgebrochen
max_operations = 300 # Maximal durchgeführte Anzahl an Operationen, nach denen der Blasprozess auf jeden Fall abgebrochen wird
# Datei zum Speichern der durchgeführten Blasvorgänge festlegen:
output_file = ""

def squared_std(values : list[float]):
    """
    Returns:
        float: Die quadrierte Standardabweichung (= Varianz) der Werte in values
    """
    mean = sum(values) / len(values)
    squared_diff = [(value - mean) ** 2 for value in values]
    return sum(squared_diff)

def manhattan_distance(feld0, feld1) -> int:
    """
    Returns:
        int: Die Manhatten-Distanz zwischen den Tupeln feld0 und feld1, definiert als d = (feld1[0] - feld0[0]) + (feld1[1] - feld1[0])
    """
    return abs(feld1[0] - feld0[0]) + abs(feld1[1] - feld0[1])

class Solver1:
    def __init__(self, hof, *, satisfied_constraint, max_operations=1000, Q, weight_avg, weight_varianz):
        """
        Args:
            hof (Hof): Der Hof, auf den sich die Strategie beziehen soll
            max_operations (int): Anzahl an Blasoperationen, nach der die Strategie auf jeden Fall abbricht
            satisfied_constraint (float): Anteil des Laubs auf Q am Gesamtlaub, nach dessen Erreichen die Strategie auf jeden Fall abbricht
            Q (tuple): Der Index von Feld Q
            weight_avg (float), weight_varianz (float): Die Gewichte, mit denen die Größen durchschnittl. Blattabstand zu Q und Varianz der Blattabstände zu Q in der Greedy Heuristik gewichtet bzw. multipliziert werden
        """
        self.hof = hof
        self.max_operations = max_operations
        self.satisfied_constraint = satisfied_constraint
        self.Q = Q
        if not self.hof.does_exist(Q):
            print("Q existiert nicht.")
            exit()
        if self.hof.is_edge(Q):
            print("Q darf kein Rand-/Eckfeld sein.")
            exit()
        self.clear_edges = True
        self.sum_laub = np.sum(self.hof.felder) # Gesamtanzahl an Blättern im Schulhof / im System bestimmen
        self.weight_avg = weight_avg
        self.weight_varianz = weight_varianz
        # Variablen, die den Fortschritt beim Leeren der Ränder speichern:
        self.edge_fields_to_clear = [(x,0) for x in range(self.hof.x_size)] + [(x,self.hof.y_size-1) for x in range(self.hof.x_size)] + [(0,y) for y in range(self.hof.y_size)] +[(self.hof.x_size-1,y) for y in range(self.hof.y_size)] # Eine Liste mit allen Eckfeldern, die von der self.greedy_edges() Funktion zum Speichern aller noch zu leerenden Randfelder verwendet wird
        possible_edge_target_fields = [] # Liste für Randelfer, die beim Leeren des Rands als Zielfelder (=edge_target_field) verwendet werden könnten (Felder, auf denen das Randlaub gesammelt wird, bevor es vom Rand auf ein  Nicht-Randfeld transferiert wird ("Rand-Zielfelder" / edge_target_fields)
        if not self.hof.y_size == 3:
            possible_edge_target_fields += [(0,self.Q[1]),(self.hof.x_size-1,self.Q[1])]
        if not self.hof.x_size == 3:
            possible_edge_target_fields += [(self.Q[0],0),(self.Q[0],self.hof.y_size-1)]
        if possible_edge_target_fields == []:
            self.edge_target_field = None
        else:
            self.edge_target_field = min(possible_edge_target_fields, key=lambda x : manhattan_distance(x, self.Q))
        self.clear_edge_last_field_index = 0 # Im Prozess des Verschiebens von Laub vom Feldrand auf ein Nicht-Randfeld (Teil der self.greedy_edge Methode) wird hier die zuletzt durchgeführte Operation gespeichert
        self.edge_target_field_neighbors = [] # Nachbarfelder das "Rand-Zielfelds"

    def __str__(self) -> str:
        """
        Returns:
            str: Den Fortschritt (in Hinblick auf Erreichen von self.satisfied_constraint und self.max_operations) als formatierten String
        """
        return f"Durchgeführte Blasoperationen: {self.hof.blas_counter}" + f" (Limit: {self.max_operations})" + f"\nLaubanteil auf Feld Q: {self.hof.felder[self.Q] / self.sum_laub}" + f" (Ziel: {self.satisfied_constraint / np.sum(s.hof.felder)})"

    def blattdistanzen(self, hof, feld, *, ignore_edge=True) -> float:
        """
        Args:
            hof (Hof): Hof, für den die Blattdistanzen berechnet werden sollen
            feld (tuple): Index des Felds, zu dem die Blattdistanzen berechnet werden sollen
        Keyword Arguments:
            ignore_edge (boolean): Gibt an, ob sich auf Randfeldern befindendes Laub bei der Berücksichtigung nicht berücksichtigt werden soll
        Returns:
            float: Mittler Mannhatten-Abstand der Blätter auf dem Feld feld zu allen anderen Blättern, die sich auf dem Hof hof befinden.
        """
        num_felder = hof.x_size * hof.y_size # Gesamtanzahl der Felder auf dem Hof
        blattdistanzen = [] # Hier werden zu jedem Blatt die Abstände gespeichert. Sollten sich auf Feldern "halbe" Blätte befinden, dann wird gerundet
        for x in range(hof.x_size):
            if ignore_edge:
                if x == 0 or x == hof.x_size-1:
                    continue
            for y in range(hof.y_size):
                if ignore_edge:
                    if y == 0 or y == hof.y_size-1:
                        continue
                blattdistanzen += [manhattan_distance((x,y),feld)]*round(hof.felder[(x,y)])
        return blattdistanzen

    def edge_distance(self, feld0, feld1):
        """
        returns:
            int: Die kleinste Verbindungsstrecke zwischen den Randfeldern feld0 und feld1, die nur über Rand- und Eckfelder läuft
        """
        if feld0[0] == 0 or feld0[0] == self.hof.x_size -1:
            # -> Feld 0 liegt auf unterem Rand
            if feld0[0] == feld1[0]:
                # -> Feld 0 und Feld 1 liegen in derselben "Zeile" - in diesem Fall ist der gesuchte Abstand der x-Koordinatenunterschied zwischen den Feldern
                return abs(feld0[1] - feld1[1])
            # Idee: Die beiden Ecken ermitteln, die zu Feld0 den geringsten Abstand haben.
            # min(distance(feld0, c) + distance(c, feld1)) entspricht nämlich (für das Eckeld c, für das der Ausdruck minimal wird) dem gesuchten Abstand zwischen Feld 0 und Feld 1.
            c1 = (feld0[0],0)
            d1 = feld0[1]
            c2 = (feld0[0],self.hof.y_size-1)
            d2 = abs(self.hof.y_size-1-feld0[1])
        else:
            if feld0[1] == feld1[1]:
                return abs(feld0[0] - feld1[0])
            c1 = (0,feld0[1])
            d1 = feld0[0]
            c2 = (self.hof.x_size-1,feld0[1])
            d2 = abs(self.hof.x_size-1-feld0[0])
        return min([d1+manhattan_distance(c1,feld1),d2+manhattan_distance(c2,feld1)])

    def greedy_edges(self):
        """
        Findet die beste Blasoperation zum Leeren des Rands (systematisches Verfahren)
        """
        # Def.: Das "Rand-Zielfeld" ist das Feld, auf dem das Randlaub gesammelt wird, bevor es über die Seitenabtriebe in die Mitte geblasen wird
        if self.edge_target_field is None:
            return None # Es gibt kein Rand-Zielfeld -> greedy_edge Heuristik beenden (als nächstes wird die greedy_mid Heuristik aktiviert, mit der das Laub auf Feld Q geblasen wird)
        if self.edge_fields_to_clear == []:
            return None # Es wurden alle Randfelder geleert -> greedy_edge Heuristik beenden
        possible_blow_directions = [(0,1),(1,0),(0,-1),(-1,0)]
        field_to_clear = max(self.edge_fields_to_clear, key=lambda x : self.edge_distance(x, self.edge_target_field)) # Das Feld finden, das am weitesten von edge_target_field weg ist (dieses Feld als nächstes leeren bzw. das Laub in die Richtung, in der sich edge_target_field befindet, blasen)

        if self.hof.felder[field_to_clear] > self.hof.startwert * (1-self.satisfied_constraint):
            # -> Auf dem Feld field_to_clear liegt noch eine wesentliche Menge Laub (also mehr Laub als self.hof.startwert * (1-self.satisfied_constraint))
            # Daher werden jetzt Maßnahmen durchgeführt, um es zu leeren
            # Rand-Nachbarfelder vom Rand-Zielfeld / edge_target_field (Das Feld, auf dem das Laub gesammelt wird, bevor es über die Seitenabtriebe vom Rand entfernt wird) erittelnt
            if manhattan_distance(field_to_clear, self.edge_target_field) == 1:
                # field_to_clear ist Randfeld und Nachbarfeld vom target_feld -> Speichern in Liste
                if not field_to_clear in self.edge_target_field_neighbors:
                    self.edge_target_field_neighbors.append(field_to_clear)

            elif field_to_clear == self.edge_target_field:
                # Zyklisches Blasverfahren aktivieren, mit dem der Rand gecleared wird
                # Zum Transferieren des Laubs von edge_target_field auf das Nicht-Randfeld neben edge_target_field werden diese Operationen zyklisch durchgeführt:
                # 1. Laub von der einen Seite auf edge_target_field blasen -> Seitenabtrieb auf des Feld neben edge_target_field
                # 2. Laub von der anderen Seite auf edge_target_field blasen -> Seitenabtrieb auf des Feld neben edge_target_field
                # 3. Vom Nicht-Randfeld neben edge_target_field auf edge_target_field blasen -> Das auf edge_target_field akkumulierte Laub wird wieder auf die Felder neben edge_target_field geblasen (über die Seitenabtriebe)
                self.edge_fields_to_clear += self.edge_target_field_neighbors # Die Randfelder neben edge_fields_to_clear werden zur Liste der zu leerenden Felder zurück-hinzugefügt.
                # Ziel des zyklischen Verfahrens ist es, die Laubmenge auf edge_target_field und den Rand-Nachbarfeldern von edge_target_field kontinuierlich zu vermindern, bis sie nicht mehr wesentlich ist.
                if self.clear_edge_last_field_index == 0: # Die clear_edge_last_field_index speichert die zuletzt ausfgeführte Operation und "rotiert" zyklisch
                    self.clear_edge_last_field_index = 1 # -> Vom 1. Nachbarfeld von edge_target_field aus Laub auf edge_target_field blasen
                elif self.clear_edge_last_field_index == 1:
                    self.clear_edge_last_field_index = 2
                    # -> Vom Nicht-Randfeld aus auf edge_target_field blasen, um das Laub von edge_target_field auf die beiden Nachbarfelder von edge_target_field zu schaffen
                    # Die entsprechende Operation direkt zurückgeben (keine weiteren Berechnungen erforderlich, da sich Blasrichtung und Feld 0 aus der Position von edge_target_field ergeben)
                    if self.edge_target_field[0] == 0:
                        return dict(feld0=(1,self.edge_target_field[1]), blow_direction=(-1,0))
                    elif self.edge_target_field[0] == self.hof.x_size-1:
                        return dict(feld0=(self.hof.x_size-2,self.edge_target_field[1]), blow_direction=(1,0))
                    elif self.edge_target_field[1] == 0:
                        return dict(feld0=(self.edge_target_field[0],1), blow_direction=(0,-1))
                    elif self.edge_target_field[1] == self.hof.y_size-1:
                        return dict(feld0=(self.edge_target_field[0],self.hof.y_size-2), blow_direction=(0,1))
                else:
                    self.clear_edge_last_field_index = 0 # -> Vom 0. Nachbarfeld von edge_target_field aus Laub auf edge_target_field blasen
                field_to_clear = self.edge_target_field_neighbors[self.clear_edge_last_field_index] # Eines der Rand-Nachbarfelder von edge_target_field als field_to_clear (Feld, auf das geblasen wird) festlegen

            # Alle möglichen Blasrichtungen ausprobieren, von denen aus auf field_to_clear geblasen werden kann
            for blow_direction in possible_blow_directions:
                feld0=(field_to_clear[0]-blow_direction[0],field_to_clear[1]-blow_direction[1]) # Das Feld ermitteln, von dem aus gesehen field_to_clear Feld A ist, wenn von ihm aus geblasen wird
                if not self.hof.does_exist(feld0):
                    continue
                if self.hof.is_corner(field_to_clear):
                    # Bei einer Ecke muss rechtwinklig in die Ecke geblasen werden, um sie zu leeren
                    orthogonal_direction = self.hof.get_orthogonal_direction(blow_direction)
                    # Um also das target_field zu finden, muss die zur Blasrichtung orthogonale Richtung ermittelt werden.
                    # target_field ist dann entweder field_to_clear + orthogonal_direction oder field_to_clear - orthogonal_direction (beides ausprobieren!)
                    target_field = (field_to_clear[0]+orthogonal_direction[0],field_to_clear[1]+orthogonal_direction[1])
                    if not (self.hof.is_edge(target_field) and target_field in self.edge_fields_to_clear):
                        target_field = (field_to_clear[0]-orthogonal_direction[0],field_to_clear[1]-orthogonal_direction[1])
                        if not (self.hof.is_edge(target_field) and target_field in self.edge_fields_to_clear):
                            continue
                else:
                    # Ansonsten wird gerade auf das zu leerende Feld geblasen
                    if field_to_clear in self.edge_fields_to_clear:
                        self.edge_fields_to_clear.remove(field_to_clear)
                    target_field = (field_to_clear[0]+blow_direction[0],field_to_clear[1]+blow_direction[1]) # Das Feld, auf das das Laub gelangen soll
                    if not self.hof.does_exist(target_field):
                        continue
                if self.edge_distance(target_field, self.edge_target_field) < self.edge_distance(field_to_clear, self.edge_target_field):
                    return dict(
                        feld0=feld0,
                        blow_direction=blow_direction
                    )
        else:
            self.edge_fields_to_clear.remove(field_to_clear)
            return self.greedy_edges()

    def greedy_mid(self, *, weight_varianz, weight_avg):
        """
        Findet die beste Blasoperation zum Verschieben von Laub von Nicht-Randfeldern Richtung Q. (unter Verwendung einer Greedy-Heuristik)
        Returns:
            dict: Die als nächstes auszuführende Blasoperation.
        """
        possible_blow_directions = [(0,1),(1,0),(0,-1),(-1,0)]
        blattdistanzen = self.blattdistanzen(self.hof, self.Q, ignore_edge=False) # Berechnet alle Blattdistanzen vom Feld Q aus
        current_mw_bd = sum(blattdistanzen) / len(blattdistanzen) # Die mittlere Blattdistanz am aktuellen Hof
        best_score = float("inf") # Bester (= niedrigeste) Wert für die STABW der Blattdistanz an einem Hof, der aus einer möglichen Blasoperation resultiert
        best_op = None # Die Blasoperation, aus der der best_varianz_bd Wert resultiert ist
        for x in range(0,self.hof.x_size):
            for y in range(0,self.hof.y_size):
                for blow_direction in possible_blow_directions:
                    if not self.hof.is_edge((x,y)):
                        if self.hof.is_edge((x+blow_direction[0],y+blow_direction[1])) or self.hof.is_edge((x+2*blow_direction[0],y+2*blow_direction[1])):
                            continue # Laub auf Rand blasen ist verboten
                    if not manhattan_distance((x,y), self.Q) < manhattan_distance((x+blow_direction[0], y+blow_direction[1]), self.Q):
                        hof_copy = self.hof.copy()
                        hof_copy.rules.use_binomial=False # Beim Finden der besten Blasoperation mit den Erwartungswerten arbeiten, um die Zufallskomponente zu entfernen
                        hof_copy.blase((x,y), blow_direction)
                        blattdistanzen = self.blattdistanzen(hof_copy, self.Q, ignore_edge=False)
                        if sum(blattdistanzen) / len(blattdistanzen) < current_mw_bd: # Erstes, priorisiertes Maß der Heuristik. Nur Operationen, die dieses Maß verringern, werden zugelassen
                            score = squared_std(blattdistanzen) * weight_varianz + (sum(blattdistanzen) / len(blattdistanzen)) * weight_avg # Gewichtetes Produkt aus erstem und zweitem Maß bilden
                            if score < best_score:
                                # Blasoperation mit dem besten Score finden
                                best_score = score
                                best_op = dict(feld0=tuple((x,y)), blow_direction=tuple(blow_direction))
        return best_op

    def step(self):
        """
        Führt den nächsten Schritt des Blasprozesses aus:
        Mithilfe von self.greedy wird die nächste Blasoperation bestimmt, die anschließend ausgeführt wird
        """
        if self.clear_edges:
            next_op = self.greedy_edges()
        else:
            next_op = self.greedy_mid(weight_varianz=self.weight_varianz, weight_avg=self.weight_avg)
        if next_op is None:
            self.clear_edges = not self.clear_edges
            if self.clear_edges:
                self.edge_fields_to_clear = [(x,0) for x in range(self.hof.x_size)] + [(x,self.hof.y_size-1) for x in range(self.hof.x_size)] + [(0,y) for y in range(self.hof.y_size)] +[(self.hof.x_size-1,y) for y in range(self.hof.y_size)]
                self.edge_target_field = min([(0,self.Q[1]),(self.hof.x_size-1,self.Q[1]),(self.Q[0],0),(self.Q[0],self.hof.y_size-1)], key=lambda x : manhattan_distance(x, self.Q))
                next_op = self.greedy_edges()
            else:
                next_op = self.greedy_mid(weight_varianz=self.weight_varianz, weight_avg=self.weight_avg)
            if next_op is not None:
                return True
        else:
            self.hof.blase(next_op["feld0"], next_op["blow_direction"])
            if self.satisfied_constraint is not None:
                if self.hof.felder[self.Q] / self.sum_laub >= self.satisfied_constraint:
                    return False
            if self.max_operations is not None:
                if self.hof.blas_counter >= self.max_operations:
                    return False
            return True

    def run(self):
        """
        Ruft self.step() solange auf, bis die Abbruchbedingung erreicht wurde
        """
        self.running_op_index = 0
        while self.step():
            pass

h = Hof(hof_size, Rules(
    use_binomial=use_binomial, binomial_rank="random", # Über use_binomial kann gesteuert werden, ob die Binomialverteilung oder die Erwartungswerte zur Wahrscheinlichkeitsmodellierung verwendet werden sollen
    A_seitenabtrieb = 0.1, B_vorne_abtrieb = 0.1, A_noB_seitenabtrieb=0.5*0.95, s1=0.9, s4=0.05 # Hier können die Blasregeln angepasst werden
    ), startwert=startwert) # Über startwert kann festgelegt werden, wie viele Blätter am Anfang auf jedem einzelnen Feld liegen
s = Solver1(h, Q=Q, weight_avg=weight_avg, weight_varianz=weight_varianz, satisfied_constraint=satisfied_constraint, max_operations=max_operations) # Hier kann Q festgelegt werden

print("AUSGANGSPUNKT:")
print("Verwendete Wahrscheinlichkeitsmodellierung:", "binomial" if s.hof.rules.use_binomial else "Erwartungswert-basiert")
print("Laubverteilung vor dem Blasprozess:")
print(s.hof)

print("\nFühre Simulation durch ...")
s.run()
print("\nERGEBNIS DER SIMULATION:")
print("Ausgeführte Blasoperationen:", s.hof.blas_counter)
print("Anteil der Blätter auf Q an der Gesamtlaubmenge:", s.hof.felder[s.Q]*100/s.sum_laub, "%")
print("Laubverteilung nach dem Blasprozess:")
print(s.hof)

if output_file != "":
    print("\nDURCHGEFÜHRTE BLASOPERATIONEN:")
    try:
        with open(output_file, "w") as f:
            f.write(s.hof.print_blas_log())
        print(f"In {output_file} gespeichert")
    except Exception as e:
        print("Konnte nicht in Datei schreiben")

s.hof.render(title=f"[S1] Laubverteilung nach dem Blasprozess\n({s.hof.blas_counter} Blasoperationen ausgeführt)", Q=s.Q)
