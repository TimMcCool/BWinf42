# Ansatz 3
# Dieses Programm kombiniert Ansatz 1 (aufgabe1_solver1.py) und Ansatz 2 (aufgabe1_solver2.py). Es ist praxisbezogener, da es schnelle Lösungen findet (weniger Blasoperationen). Es kann weniger des Laubs als Verfahren 2 auf Q schaffen, aber mehr als Verfahren 1.
# Ansatz 2 (aufgabe1_solver2.py) ist trotzdem noch besser und eher auf die Aufgabenstellung bezogen (siehe Dokumentation).
# Die zu bewertende Implementierung ist in aufgabe1_solver2.py enthalten.

import numpy as np
import math
from hof import Rules, Hof

# Hyperparameter festlegen
Q = (2,2) # Index von Feld Q festlegen
hof_size = (5,5) # Hofseitenlängen festlegen
use_binomial = True # Festlegen, ob die Wahrscheinlichkeiten basierend auf der Binomialverteilung simuliert oder ob die Erwartungswerte verwendet werden sollen
startwert = 100 # Anfangsanzahl an Blättern pro Feld
weight_avg = 0.5 # Gewichtung des 1. Heuristik-Maßes (durchschnittlicher Laubabstand zu Feld Q)
weight_varianz = 0.5 # Gewichtung des 2. Heuristik-Maßes (Varianz der Laubabstände zu Feld Q)
satisfied_constraint = 0.3 # Bei erreichen dieser prozentualen Laubmenge (im Verhältnis zum Gesamtlaub) wird das Programm auf jeden Fall abgebrochen
max_operations = 1000 # Maximal durchgeführte Anzahl an Operationen, nach denen der Blasprozess auf jeden Fall abgebrochen wird
tolerated_amount = 5 # Blattmenge, die auf nicht vollständig leerbaren Feldern als vernachlässigbar gilt
clear_edges = True # Wenn auf False gesetzt, dann werden die Ränder gar nicht gecleared, sondern vollständig ignoriert
max_muster_operations = 1000 # Anzahl an Operation, die pro Muster maximal durchgeführt werden

# Datei zum Speichern der durchgeführten Blasvorgänge festlegen:
output_file = "C:/Users/timkr/OneDrive/Projekte/BWInf 24/BWinf2 24 Tim Krome Runde 2/Aufgabe 1/Outputs/output_ansatz3_beispiel1.txt"

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

class Muster:
    """
    Implementierung des theoretischen Konzept eines Musters (siehe Dokumentation)
    """
    def __init__(self, strategy, source_fields, operations, tolerated_amount : float, *, num_max_operations=None):
        self.strategy = strategy # Die Strategie, zu der das Muster gehört
        self.hof = self.strategy.hof # Der Hof, zu dem die Strategie gehört, zu der das Muster gehört
        if num_max_operations is None:
            self.num_max_operations = strategy.max_muster_operations # Maximale Anzahl an Operationen, nach der das Muster abgebrochen wird. Wenn nicht als Argument gegeben, dann wird das der Strategie verwendet
        else:
            self.num_max_operations = num_max_operations
        self.source_fields = [strategy.rotate_field_index(f) for f in source_fields] # "Source-Felder" bzw. Felder, die das Muster leeren soll
        self.operations = [strategy.rotate_blasoperation(o) if isinstance(o, dict) else o for o in operations] # Die Operationen, die zum Muster gehören (können Blasvorgänge oder andere Muster sein)
        self.tolerated_amount = tolerated_amount # Anzahl an Blättern, die auf einem SOurce-Feld toleriert werden (das Muster wird abgebrochen, sobald auf jedem der Source-Feldern weniger als tolerated_amount Blätter sind)
        self.reset()

    def reset(self):
        """
        Zurücksetzen des Musters auf den Zustand vor seiner Ausführung
        """
        self.num_operations = 0 # Bisher durchgeführte Blasoperationen
        self.check_for_changes = not (self.hof.rules.use_binomial is True and self.hof.rules.binomial_rank == "random")
        if self.check_for_changes:
            # Es wird nach jedem vollständigen Musterdurchlauf überprüft, ob nach Veränderungen an der Gesamtlaubmenge auf den source-Feldern stattfinden
            # Diese Überprüfung findet aber nicht statt, wenn die Laubblassimulation den tatsächlichen Zufall simuliert
            self.current_sum = 0
        self.next_op_index = 0 # Index der als nächstes auszuführenden Operation

    def step(self):
        """
        Führt basierend auf self.operations die nächste Blasoperation aus
        """
        if max([self.hof.felder[index] for index in self.source_fields]) <= self.tolerated_amount:
            return False
        if isinstance(self.operations[self.next_op_index], dict):
            # -> Eine Blasoperation liegt vor, die ausgeführt wird
            self.hof.blase(self.operations[self.next_op_index]["feld0"], self.operations[self.next_op_index]["blow_direction"])
            self.next_op_index += 1
            self.num_operations += 1
        elif isinstance(self.operations[self.next_op_index], Muster):
            # -> Ein anderes Muster liegt vor, das ausgeführt wird
            run_another_step = self.operations[self.next_op_index].step()
            if run_another_step:
                return True
            self.next_op_index += 1
            self.num_operations += 1

        if self.next_op_index == len(self.operations):
            # -> Einmal durch alle Operationen des Musters durchgelaufen -> i zurücksetzen
            self.next_op_index = 0
            if self.check_for_changes:
                # Überprüfen, ob noch Veränderungen stattfinden
                new_sum = sum([self.hof.felder[index] for index in self.source_fields])
                if new_sum == self.current_sum:
                    return False
                self.current_sum = int(new_sum)
        if isinstance(self.operations[self.next_op_index], Muster):
            self.operations[self.next_op_index].reset()

        # Überprüfen, ob maximale Anzahl an Operationen erreicht wurde:
        if self.num_operations >= self.num_max_operations:
            return False

        if max([self.hof.felder[index] for index in self.source_fields]) <= self.tolerated_amount:
            return False

        return True

class Solver3:
    """
    Dient zum Durchführen von der Strategie 1 (Heuristikstrategie) - siehe hierzu Dokumentation
    """
    def __init__(self, hof, *, clear_edges=True, satisfied_constraint, max_operations=1000, Q, weight_avg, weight_varianz, tolerated_amount, max_muster_operations):
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
        self.sum_laub = np.sum(self.hof.felder) # Gesamtanzahl an Blättern im Schulhof / im System bestimmen
        self.weight_avg = weight_avg
        self.weight_varianz = weight_varianz
        # Variablen, die den Fortschritt beim Leeren der Ränder speichern:
        self.edge_fields_to_clear = [(x,0) for x in range(self.hof.x_size)] + [(x,self.hof.y_size-1) for x in range(self.hof.x_size)] + [(0,y) for y in range(self.hof.y_size)] +[(self.hof.x_size-1,y) for y in range(self.hof.y_size)] # Eine Liste mit allen Eckfeldern, die von der self.greedy_edges() Funktion zum Speichern aller noch zu leerenden Randfelder verwendet wird
        possible_edge_target_fields = [] # Liste für Randelfer, die als beim Leeren des Rands als Zielfelder verwendet werden könnten (Felder, auf denen das Randlaub gesammelt wird, bevor es vom Rand auf ein  Nicht-Randfeld transferiert wird ("Rand-Zielfelder")
        if not self.hof.y_size == 3:
            possible_edge_target_fields += [(0,self.Q[1]),(self.hof.x_size-1,self.Q[1])]
        if not self.hof.x_size == 3:
            possible_edge_target_fields += [(self.Q[0],0),(self.Q[0],self.hof.y_size-1)]
        if possible_edge_target_fields == []:
            self.edge_target_field = None
        else:
            self.edge_target_field = min(possible_edge_target_fields, key=lambda x : manhattan_distance(x, self.Q))
        self.edge_target_field_neighbors = [] # Nachbarfelder das "Rand-Zielfelds"
        self.currently_running_muster = None
        self.tolerated_amount = tolerated_amount # Die Anzahl an Blätter, die ein Muster "toleriert" bzw. bei Erreichen dieser Blattanzahl auf den Source-Feldern bricht ein Muster ab
        self.max_muster_operations = max_muster_operations # Maximale Anzahl an Operationen, die ein Muster durchführt, bevor es abbricht
        self.clear_edges = True
        self.attempt_clear_edges = clear_edges
        if not clear_edges:
            self.clear_edges = False
        self.num_rotations = 0

    def __str__(self) -> str:
        """
        Returns:
            str: Den Fortschritt (in Hinblick auf Erreichen von self.satisfied_constraint und self.max_operations) als formatierten String
        """
        return f"Durchgeführte Blasoperationen: {self.hof.blas_counter}" + f" (Limit: {self.max_operations})" + f"\Laubanteil auf Feld Q: {self.hof.felder[self.Q] / self.sum_laub}" + f" (Ziel: {self.satisfied_constraint / np.sum(s.hof.felder)})"

    def rotate_field_index(self, field_index) -> tuple:
        """
        Ermittelt den ursprünglichen Index des Felds (den es vor der Initialrotation hatte), das derzeit am Index field_index ist
        """
        x_size, y_size = int(self.hof.x_size), int(self.hof.y_size)
        for i in range(self.num_rotations):
            field_index = (y_size-1-field_index[1], field_index[0]) # field_index um 90° im Uhrzeigersinn rotieren
            helper = int(y_size) # Eine Rotation um 90° sorgt dafür, dass sich x-Größe (Breite) und y-Größe (Höhe) des Hofs tauschen
            y_size = int(x_size)
            x_size = int(helper)
        return field_index

    def rotate_direction_vector(self, vector : tuple) -> tuple:
        """
        Ermittelt die ursprüngliche Richtung des Vektors vector (den er vor der Initialrotation hatte)
        """
        for r in range(self.num_rotations):
            # Vektor um 90° im Uhrzeigersinn rotieren:
            if vector[0] == 0:
                vector = (-vector[1],0)
            elif vector[1] == 0:
                vector = (0,vector[0])
        return vector

    def rotate_blasoperation(self, blasoperation) -> dict:
        return dict(
            feld0=self.rotate_field_index(blasoperation["feld0"]),
            blow_direction=self.rotate_direction_vector(blasoperation["blow_direction"])
        )

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

    def corner_to_edge(self, source_corner_field, target_field) -> Muster:
        """
        Returns:
            Muster: Ein Muster, das bei Anwendung Laub vom Eckfeld source_corner_field auf das Randfeld target_field bläst
        """
        if not self.hof.are_adjacent(source_corner_field, target_field):
            return
        if not (self.hof.is_corner(source_corner_field) and self.hof.is_edge(target_field)):
            return
        orthogonal_direction = (target_field[0]-source_corner_field[0], target_field[1]-source_corner_field[1]) # Die Richtung, die orthogonal zur Blasrichtung ist (entspricht dem Vektor target_field-source_corner)
        # Beide zu orthogonal_direction orthogonale richtungen werden als mögliche Blasrichtungen durchprobiert:
        anti_blow_direction = self.hof.get_orthogonal_direction(orthogonal_direction) # Gegenrichtung / Gegenvektor zur Blasrichtung
        feld0 = (source_corner_field[0]+anti_blow_direction[0], source_corner_field[1]+anti_blow_direction[1]) # Erstes mögliches Startfeld
        blow_direction = (-anti_blow_direction[0], -anti_blow_direction[1])
        if not self.hof.does_exist(feld0): # Wenn das erste mögliche Startfeld nicht existiert, dann muss das andere mögliche Startfeld das Startfeld sein
            anti_blow_direction = blow_direction
            feld0 = (source_corner_field[0]+anti_blow_direction[0], source_corner_field[1]+anti_blow_direction[1])
            blow_direction = (-anti_blow_direction[0], -anti_blow_direction[1])
        muster = Muster(self, [source_corner_field], [dict(feld0=feld0, blow_direction=blow_direction)], self.tolerated_amount) # Muster Objekt erstellen
        return muster

    def edge_to_mid(self, source_edge_field, target_field) -> Muster:
        """
        Returns:
            Muster: Ein Muster, das bei Anwendung Laub vom Randfeld source_edge_field und seinen beiden auf dem Rand liegenden Nachbarn auf das Nicht-Rand-oder-Eckfeld target_field bläst
        """
        if not self.hof.are_adjacent(source_edge_field, target_field):
            return
        if (not self.hof.is_edge(source_edge_field)) or self.hof.is_edge(target_field):
            return
        orthogonal_direction = (source_edge_field[0]-target_field[0], source_edge_field[1]-target_field[1]) # Die Richtung, die vom Rand wegzeigt
        blow_direction = self.hof.get_orthogonal_direction(orthogonal_direction)
        operations = []
        operations.append(dict(feld0=target_field, blow_direction=orthogonal_direction)) # Den Blasvorgang von der Mitte zur Kante hin ermitteln (dieser verteilt das Laub durch die Seitenabtriebe auf den beiden Feldern neben source_edge_field)
        # Die Blasvorgänge, die Laub durch Seitenabtriebe auf target_field transportieren, ermitteln
        feld0_1 = (source_edge_field[0]+blow_direction[0]*2, source_edge_field[1]+blow_direction[1]*2)
        blow_direction = (-blow_direction[0], -blow_direction[1])
        if self.hof.does_exist(feld0_1):
            operations.append(dict(feld0=feld0_1, blow_direction=blow_direction))
        else:
            operations.insert(0, dict(feld0=(source_edge_field[0]-blow_direction[0]-orthogonal_direction[0], source_edge_field[1]-blow_direction[1]-orthogonal_direction[1]), blow_direction=orthogonal_direction))
        feld0_2 = (source_edge_field[0]+blow_direction[0]*2, source_edge_field[1]+blow_direction[1]*2)
        blow_direction = (-blow_direction[0], -blow_direction[1])
        if self.hof.does_exist(feld0_2):
            operations.append(dict(feld0=feld0_2, blow_direction=blow_direction))
        else:
            operations.insert(0, dict(feld0=(source_edge_field[0]-blow_direction[0]-orthogonal_direction[0], source_edge_field[1]-blow_direction[1]-orthogonal_direction[1]), blow_direction=orthogonal_direction))
        if not (self.hof.does_exist(feld0_1) or self.hof.does_exist(feld0_2)):
            return # Beide möglichen Startfelder existieren nicht -> Hof hat Kantenlänge 3 -> Kante kann nicht gecleared werden
        source_fields = [source_edge_field, (source_edge_field[0]+blow_direction[0], source_edge_field[1]+blow_direction[1]), (source_edge_field[0]-blow_direction[0], source_edge_field[1]-blow_direction[1])]
        muster = Muster(self, source_fields, operations, self.tolerated_amount)
        return muster

    def concentrate_on_Q(self):
        """
        Gibt ein Muster zurück, dass das Laub auf Q sammelt. Q muss den Index (2,2) - wenn es einen anderen Index hat, dann wird der Hof solange gedreht bis der Index passt.
        """
        # Hof rotieren, bis Q am Index (2,2) ist
        Q = self.Q
        num_rotations = 0
        for i in range(3):
            if Q == (2,2):
                break
            self.hof.felder = np.rot90(self.hof.felder, k=1, axes=(0, 1))
            self.hof.x_size = self.hof.felder.shape[0]
            self.hof.y_size = self.hof.felder.shape[1]
            Q = (Q[1], self.hof.y_size-1-Q[0])
            num_rotations += 1
        else:
            return None
        self.num_rotations = num_rotations
        self.Q = Q
        # Muster erstellen
        operations = []
        operations.append(Muster(self, [(1,2),(1,3)], [dict(feld0=(1,4), blow_direction=(0,-1))], self.tolerated_amount))
        operations.append(Muster(self, [(3,2),(3,3)], [dict(feld0=(3,4), blow_direction=(0,-1))], self.tolerated_amount))
        operations.append(Muster(self, [(2,3)], [dict(feld0=(2,4), blow_direction=(0,-1))], self.tolerated_amount))
        operations.append(Muster(self, [(1,1),(2,1),(3,1)], [
            dict(feld0=(0,1), blow_direction=(1,0)),
            dict(feld0=(4,1), blow_direction=(-1,0)),
            ], self.tolerated_amount)
        )
        operations.append(self.edge_to_mid((2,0), (2,1)))
        muster = Muster(self, [(1,1), (2,1), (3,1), (2,0)], operations, self.tolerated_amount)
        # Zuvor ausgeführte Rotation rückgängig machen
        for i in range(self.num_rotations):
            self.hof.felder = np.rot90(self.hof.felder, k=-1, axes=(0, 1))
            Q = (self.hof.y_size-1-Q[1], Q[0])
            self.hof.x_size = self.hof.felder.shape[0]
            self.hof.y_size = self.hof.felder.shape[1]
        self.Q = Q
        return muster

    def greedy_edges(self):
        if self.edge_target_field is None:
            return None # Es gibt kein Rand-Zielfeld -> Programm beenden
        if self.edge_fields_to_clear == []:
            return None
        possible_blow_directions = [(0,1),(1,0),(0,-1),(-1,0)]
        field_to_clear = max(self.edge_fields_to_clear, key=lambda x : self.edge_distance(x, self.edge_target_field))
        self.edge_fields_to_clear.remove(field_to_clear)
        if self.hof.felder[field_to_clear] < self.tolerated_amount:
            return self.greedy_edges()

        if manhattan_distance(field_to_clear, self.edge_target_field) == 1:
            if not field_to_clear in self.edge_target_field_neighbors:
                self.edge_target_field_neighbors.append(field_to_clear)
        elif field_to_clear == self.edge_target_field:
            if self.edge_target_field[0] == 0:
                target_field=(1,self.edge_target_field[1])
            elif self.edge_target_field[0] == self.hof.x_size-1:
                target_field=(self.hof.x_size-2,self.edge_target_field[1])
            elif self.edge_target_field[1] == 0:
                target_field=(self.edge_target_field[0],1)
            elif self.edge_target_field[0] == 0:
                target_field=(self.edge_target_field[0],self.hof.y_size-2)
            return self.edge_to_mid(field_to_clear, target_field)
        for blow_direction in possible_blow_directions:
            feld0=(field_to_clear[0]-blow_direction[0],field_to_clear[1]-blow_direction[1])
            if not self.hof.does_exist(feld0):
                continue
            if self.hof.is_corner(field_to_clear):
                orthogonal_direction = self.hof.get_orthogonal_direction(blow_direction)
                target_field = (field_to_clear[0]+orthogonal_direction[0],field_to_clear[1]+orthogonal_direction[1])
                if not (self.hof.is_edge(target_field) and target_field in self.edge_fields_to_clear):
                    target_field = (field_to_clear[0]-orthogonal_direction[0],field_to_clear[1]-orthogonal_direction[1])
                    if not (self.hof.is_edge(target_field) and target_field in self.edge_fields_to_clear):
                        continue
                return self.corner_to_edge(field_to_clear, target_field)
            else:
                if field_to_clear in self.edge_fields_to_clear:
                    self.edge_fields_to_clear.remove(field_to_clear)
                target_field = (field_to_clear[0]+blow_direction[0],field_to_clear[1]+blow_direction[1])
                if not self.hof.does_exist(target_field):
                    continue
            if self.edge_distance(target_field, self.edge_target_field) < self.edge_distance(field_to_clear, self.edge_target_field):
                return dict(
                    feld0=feld0,
                    blow_direction=blow_direction
                )

    def greedy_mid(self, *, weight_varianz, weight_avg):
        """
        Returns:
            dict: Die als nächstes auszuführende Blasoperation.
        """
        possible_blow_directions = [(0,1),(1,0),(0,-1),(-1,0)]
        blattdistanzen = self.blattdistanzen(self.hof, self.Q, ignore_edge=False)
        current_mw_bd = sum(blattdistanzen) / len(blattdistanzen) # Die mittlere Blattdistanz am aktuellen Hof
        best_score = float("inf") # Bester (= niedrigeste) Wert für die STABW der Blattdistanz an einem Hof, der aus einer möglichen Blasoperation resultiert
        best_op = None # Die Blasoperation, aus der der best_varianz_bd Wert resultiert ist
        unprioritised = []
        for x in range(0,self.hof.x_size):
            for y in range(0,self.hof.y_size):
                for blow_direction in possible_blow_directions:
                    if not self.hof.is_edge((x,y)):
                        if self.hof.is_edge((x+blow_direction[0],y+blow_direction[1])) or self.hof.is_edge((x+2*blow_direction[0],y+2*blow_direction[1])):
                            continue # Laub auf Rand blasen ist verboten
                    if not manhattan_distance((x,y), self.Q) < manhattan_distance((x+blow_direction[0], y+blow_direction[1]), self.Q):
                        hof_copy = self.hof.copy()
                        hof_copy.rules.use_binomial=False
                        hof_copy.blase((x,y), blow_direction)
                        blattdistanzen = self.blattdistanzen(hof_copy, self.Q, ignore_edge=False)
                        if sum(blattdistanzen) / len(blattdistanzen) < current_mw_bd:
                            score = squared_std(blattdistanzen) * weight_varianz + (sum(blattdistanzen) / len(blattdistanzen)) * weight_avg
                            if score < best_score:
                                best_score = score
                                best_op = dict(feld0=tuple((x,y)), blow_direction=tuple(blow_direction))
        return best_op

    def step(self):
        """
        Führt den nächsten Schritt des Blasprozesses aus:
        Mithilfe von self.greedy wird die nächste Blasoperation bestimmt, die anschließend ausgeführt wird
        """
        if self.satisfied_constraint is not None:
            if self.hof.felder[self.Q] / self.sum_laub >= self.satisfied_constraint:
                return False
        if self.max_operations is not None:
            if self.hof.blas_counter >= self.max_operations:
                return False
        if self.currently_running_muster is not None:
            if not self.currently_running_muster.step():
                self.currently_running_muster = None
            return True
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
                self.currently_running_muster = self.concentrate_on_Q()
                return True
        else:
            if isinstance(next_op, dict):
                self.hof.blase(next_op["feld0"], next_op["blow_direction"])
            elif isinstance(next_op, Muster):
                if next_op.step():
                    self.currently_running_muster = next_op
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
s = Solver3(h, Q=Q, weight_avg=weight_avg, weight_varianz=weight_varianz, satisfied_constraint=satisfied_constraint, # Hier kann Q festgelegt werden
    max_operations=max_operations, tolerated_amount=tolerated_amount, max_muster_operations=max_muster_operations, clear_edges=clear_edges)

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

s.hof.render(title=f"[S3] Laubverteilung nach dem Blasprozess\n({s.hof.blas_counter} Blasoperationen ausgeführt)", Q=s.Q)
