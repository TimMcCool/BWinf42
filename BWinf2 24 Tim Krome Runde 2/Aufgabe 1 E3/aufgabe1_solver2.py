# Erweiterung 3: Es gibt zwei Laubtypen, für jeden Laubtyp gelten andere Regeln

import math
from hof import Rules, Hof # Importieren der Klassen aus hof.py
import numpy as np

# Hyperparameter festlegen
Q = (1,2) # Index von Feld Q festlegen
hof_size = (8,4) # Hofseitenlängen festlegen
use_binomial = True # Festlegen, ob die Wahrscheinlichkeiten basierend auf der Binomialverteilung simuliert oder ob die Erwartungswerte verwendet werden sollen
tolerated_amount = 5 # Blattmenge, die auf nicht vollständig leerbaren Feldern als vernachlässigbar gilt
max_muster_operations = 5000 # Anzahl an Operation, die pro Muster maximal durchgeführt werden
startwert = 100 # Anfangsanzahl an Blättern pro Feld
choose_faster_path = True
# Datei zum Speichern der durchgeführten Blasvorgänge festlegen:
output_file = "C:/Users/timkr/OneDrive/Projekte/BWInf 24/BWinf2 24 Tim Krome Runde 2/Aufgabe 1 E3/Outputs/output_E3_beispiel2.txt"

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
        self.check_for_changes = not ((self.hof.rules_typ_1.use_binomial is True and self.hof.rules_typ_1.binomial_rank == "random") or (self.hof.rules_typ_2.use_binomial is True and self.hof.rules_typ_2.binomial_rank == "random"))
        if self.check_for_changes:
            # Es wird nach jedem vollständigen Musterdurchlauf überprüft, ob nach Veränderungen an der Gesamtlaubmenge auf den source-Feldern stattfinden
            # Diese Überprüfung findet aber nicht statt, wenn die Laubblassimulation den tatsächlichen Zufall simuliert
            self.current_sum = 0
        self.next_op_index = 0 # Index der als nächstes auszuführenden Operation

    def step(self):
        """
        Führt basierend auf self.operations die nächste Blasoperation aus
        """
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
                new_sum = sum([(self.hof.felder_typ_1[index] + self.hof.felder_typ_2[index]) for index in self.source_fields])
                if new_sum == self.current_sum:
                    return False
                self.current_sum = int(new_sum)
        if isinstance(self.operations[self.next_op_index], Muster):
            self.operations[self.next_op_index].reset()

        # Überprüfen, ob maximale Anzahl an Operationen erreicht wurde:
        if self.num_operations >= self.num_max_operations:
            return False

        if max(max([self.hof.felder_typ_1[index] for index in self.source_fields]), max([self.hof.felder_typ_2[index] for index in self.source_fields])) <= self.tolerated_amount:
            return False

        return True

    def run(self):
        """
        Ruft self.step() solange auf, bis die Abbruchbedingungen erfüllt sind
        """
        self.reset()
        while self.step():
            pass

class Solver2:
    """
    Dient zum Erstellen, Speichern, Ausführen von Strategien bzw. generalisierten Ablaufplänen nach Verfahren 2 (siehe Dokumentation)
    """
    def __init__(self, hof, *, tolerated_amount, max_muster_operations=100, choose_faster_path=False):
        self.hof = hof # Der Hof, auf den sich die Strategie beziehen soll
        self.tolerated_amount = tolerated_amount # Die Anzahl an Blätter, die ein Muster "toleriert" bzw. bei Erreichen dieser Blattanzahl auf den Source-Feldern bricht ein Muster ab
        if tolerated_amount <= 0:
            print("tolerated_amount muss größer als Null sein.")
            exit()
        self.strategy = [] # Hier werden die Operationen des generalisierten Programmablaufplans gespeichert
        self.max_muster_operations = max_muster_operations # Maximale Anzahl an Operationen, die ein Muster durchführt, bevor es abbricht
        self.running_op_index = 0
        # Bekommen später einen Wert zugewiesen:
        self.Q = None # Hier wird der Index von Feld Q als Tupel gespeichert werden
        self.num_rotations = 0 # Speichern der Anzahl an Rotationen, die vor Entwerfen einer Strategie auf den Hof und auf Q angewandt wurden
        self.choose_faster_path = choose_faster_path

    def add_operation(self, operation):
        """
        Fügt eine Blasoperation oder ein Muster zum Ablaufplan hinzu. Macht zuvor die Initialrotation rückgängig (beim späteren Ausführen wird der Hof in seiner ursprünglichen Ausrichtung betrachtet, die Rotation erfolgt nur, da dies die Generierung des generalisierten Ablaufplans / der Strategie erleichtert).
        """
        self.strategy.append(self.rotate_blasoperation(operation) if isinstance(operation, dict) else operation)

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

    def stringify_operation_sequence(self, sequence, *, indent=0) -> list[str]:
        """
        Wandelt die Operation in einen formatierten, gut lesbaren String um
        """
        output = []
        directions = {(1,0): "rechts", (-1,0): "links", (0,1): "unten", (0,-1): "oben"} # Die möglichen Blasrichtungen als Fließtext
        for op in sequence:
            if isinstance(op, dict):
                output.append("  "*indent+f"blase(Feld0: {op['feld0']}, nach: {directions[op['blow_direction']]})")
            elif isinstance(op, Muster):
                output.append("  "*indent+f"Muster(Sourcefelder: {[op.source_fields]}) "+ "{")
                output += self.stringify_operation_sequence(op.operations, indent=indent+1)
                output += "}"
            elif isinstance(op, str):
                output.append(op)
        return output

    def __str__(self):
        """
        Returns:
            str: Die Operationen des generalisierten Ablaufplans als formatierter String
        """
        return "\n".join(self.stringify_operation_sequence(self.strategy)) if not len(self.strategy) == 0 else "-" # Im Falle eines Hofs der Dimensionen (3,3) ist die Ausgabe leer, da es hier keine Blasoperationen gibt, mit denen die Laubmenge auf Q erhöht werden kann

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

    def clear_bottom_line(self):
        """
        1. Phase des generalisierten Ablaufs: Befreien der untersten Reihe des Hofs. Fügt die hierfür notwendigen Operationen zu self.strategy hinzu.
        """
        for start_x in range(self.hof.x_size-1,0,-1):
            self.add_operation(dict(feld0=(start_x,self.hof.y_size-1),blow_direction=(-1,0))) # Nicht-Eckfelder am unteren Rand leeren
        self.add_operation(self.corner_to_edge((self.hof.x_size-1, self.hof.y_size-1),(self.hof.x_size-1,self.hof.y_size-2))) # Ecke rechts unten leeren
        self.add_operation(self.corner_to_edge((0, self.hof.y_size-1),(0,self.hof.y_size-2))) # Ecke links unten leeren

    def move_to_top_line(self):
        """
        2. Phase des generalisierten Ablaufs: Bläst das gesamte Laub in die oberste Reihe. Fügt die hierfür notwendigen Operationen zu self.strategy hinzu.
        """
        for start_y in range(self.hof.y_size-1,1,-1):
            for start_x in range(self.hof.x_size-1,-1,-1):
                if not (start_x == self.Q[0] and (start_y <= self.Q[1]+1 or self.hof.y_size == 5)): # Feld Q und Felder über Q nicht unnötigerweise leeren, da hier nachher das Laub sowieder wieder hintransportiert werden muss
                    self.add_operation(dict(feld0=(start_x, start_y), blow_direction=(0,-1)))

    def concentrate_top_line(self):
        """
        3. Phase des generalisierten Ablaufs: Konzentriert das Laub der obersten Reihe auf dem Randfeld, das sich in der selben Spalte wie Q befindet. Fügt die hierfür notwendigen Operationen zu self.strategy hinzu.
        """
        Q = self.Q
        source_fields_for_edgeclear = [(Q[0]-1,0), (Q[0],0), (Q[0]+1,0)]
        if not (0,0) in source_fields_for_edgeclear:
            self.add_operation(self.corner_to_edge((0,0),(1,0))) # Ecke links oben clearen, wenn sie in Phase 4 nicht Source-Feld wird
        if not (self.hof.x_size-1,0) in source_fields_for_edgeclear:
            self.add_operation(self.corner_to_edge((self.hof.x_size-1,0),(self.hof.x_size-2,0))) # Ecke rechts oben clearen, wenn sie in Phase 4 nicht Source-Feld wird
        # Nicht-Eckfelder der obersten Reihe auf Source-Felder konzentrieren:
        for start_x in range(0,self.hof.x_size-1):
            if (start_x+1,0) in source_fields_for_edgeclear:
                break
            else:
                operationen = [dict(feld0=(start_x,0), blow_direction=(1,0)), dict(feld0=(start_x+2,2), blow_direction=(0,-1))]
                muster = Muster(self, [(start_x+1,0), (start_x+2,1)], operationen, self.tolerated_amount) # Muster zum Leeren des Eckfelds (start_x+1,0) und zum gleichzeitigen Beseitigen des entstehenden Seitenabtriebs auf Feld (start_x+2,0)
                self.add_operation(muster)
        for start_x in range(self.hof.x_size-1,0,-1):
            if (start_x-1,0) in source_fields_for_edgeclear:
                break
            else:
                operationen = [dict(feld0=(start_x,0), blow_direction=(-1,0)), dict(feld0=(start_x-2,2), blow_direction=(0,-1))]
                muster = Muster(self, [(start_x-1,0), (start_x-2,1)], operationen, self.tolerated_amount)
                self.add_operation(muster)

    def transfer_to_Q(self):
        """
        4. Phase des generalisierten Ablaufs: Verschiebt das gesamte Laub auf Feld Q. Fügt die hierfür notwendigen Operationen zu self.strategy hinzu.
        """
        Q = self.Q
        self.add_operation(self.edge_to_mid((Q[0],0), (Q[0],1))) # Muster hinzufügen, dass Laub vom Eckfeld (Q[0],0) auf das Feld (Q[0],1) bläst
        if Q != (Q[0],1): # Wenn (Q[0],1) == Q, dann befindet sich das Laub jetzt bereits auf Q - wenn nicht, müssen weitere Schritte ausgeführt werden
            self.add_operation(dict(feld0=(Q[0],0), blow_direction=(0,1))) # Das Laub wird vom Rand aus auf (Q[0],2) geblasen - dabei entstehen aber Seitenabtriebe. Mit diesen wird im Folgenden umgegangen.
            # Mit den in der vorherigen Blasoperation entstandenen Seitenabtrieben umgehen: Wenn möglich wird das Laub verlustfrei durch den Abtrieb bei B vorne auf das Feld (Q[0],2) geblasen. Wenn nicht, dann wird das im vorherigen Schritt abgetriebene Laub auf den Felder (Q[0]-1,3), (Q[0],3) und (Q[0]+1,3) versammelt.
            cleared_fields = []
            if self.hof.does_exist((Q[0]+3,2)):
                muster = Muster(self, [(Q[0]+1,2)], [dict(feld0=(Q[0]+3,2),blow_direction=(-1,0))], self.tolerated_amount)
                self.add_operation(muster)
                cleared_fields.append((Q[0]+1,2))
            elif not (self.hof.x_size == 5 and self.hof.y_size == 5 and Q == (2,2)): # In diesem Fall wird anders vorgegangen (siehe Code unten)
                muster = Muster(self, [(Q[0]+1,2)], [dict(feld0=(Q[0]+1,0),blow_direction=(0,1))], self.tolerated_amount)
                self.add_operation(muster)
            if self.hof.does_exist((Q[0]-3,2)):
                muster = Muster(self, [(Q[0]-1,2)], [dict(feld0=(Q[0]-3,2),blow_direction=(1,0))], self.tolerated_amount)
                self.add_operation(muster)
                cleared_fields.append((Q[0]-1,2))
            elif not (self.hof.x_size == 5 and self.hof.y_size == 5 and Q == (2,2)): # In diesem Fall wird anders vorgegangen (siehe Code unten)
                muster = Muster(self, [(Q[0]-1,2)], [dict(feld0=(Q[0]-1,0),blow_direction=(0,1))], self.tolerated_amount)
                self.add_operation(muster)
            if self.hof.does_exist((Q[0],5)):
                if Q != (Q[0],3):
                    muster = Muster(self, [(Q[0],3)], [dict(feld0=(Q[0],5),blow_direction=(0,-1))], self.tolerated_amount)
                    self.add_operation(muster)
                    cleared_fields.append((Q[0],3))
            elif Q == (Q[0],3):
                cleared_fields.append((Q[0],3))
            if len(cleared_fields) != 3:
                # Wenn dies nicht möglich war, dann wird stattdessen ein komplexeres System aus Mustern und Blasoperationen verwendet, um das Laub auf Feld Q zu befärdern
                if (self.hof.does_exist((Q[0],6)) or ((Q[0], 3) == Q and self.hof.does_exist((Q[0],5)))) and (self.hof.does_exist((Q[0]-3, 3)) or self.hof.does_exist((Q[0]+3, 3))):
                    # In diesem Sonderfall sind unter (Q[0],2) noch 4 weitere Felder, was ein verlustfreies Blasen des unteren Seitenabtriebs auf Feld (Q[0],2) ermöglicht
                    if self.hof.does_exist((Q[0]-3, 3)):
                        self.add_operation(dict(feld0=(Q[0]+2,3), blow_direction=(-1,0)))
                        self.add_operation(Muster(self, [(Q[0]-1, 3)], [dict(feld0=(Q[0]-3,3), blow_direction=(1,0))], self.tolerated_amount))
                    else:
                        self.add_operation(dict(feld0=(Q[0]-2,3), blow_direction=(1,0)))
                        self.add_operation(Muster(self, [(Q[0]+1, 3)], [dict(feld0=(Q[0]+3,3), blow_direction=(-1,0))], self.tolerated_amount))
                    self.add_operation(Muster(self, [(Q[0], 4)], [dict(feld0=(Q[0],6), blow_direction=(0,-1))], self.tolerated_amount))
                    if not Q != (Q[0], 3):
                        # Das Laub befindet sich nach den vorherigen Schritten auf Feld (Q[0], 3) - Wenn (Q[0], 3) == 3 dann müssen folglich keine weiteren Operationen mehr durchgeführt werden
                        self.add_operation(Muster(self, [(Q[0], 3)], [dict(feld0=(Q[0],5), blow_direction=(0,-1))], self.tolerated_amount))
                elif self.hof.does_exist((Q[0],5)):
                    # Wenn unter Feld (Q[0], 2) drei Felder frei sind, dann wird mit folgendem Muster kontinuierlich Laub auf Feld (Q[0], 2 geblasen)
                    operations = []
                    operations.append(dict(feld0=(Q[0]-2,3), blow_direction=(1,0)))
                    operations.append(dict(feld0=(Q[0]+2,3), blow_direction=(-1,0)))
                    operations.append(dict(feld0=(Q[0],5), blow_direction=(0,-1)))
                    source_fields = [(Q[0]-1,3), (Q[0],3), (Q[0]+1,3), (Q[0],4)]
                    self.add_operation(Muster(self, source_fields, operations, self.tolerated_amount))
                elif self.hof.x_size == 5 and self.hof.y_size == 5 and Q == (2,2):
                    # Im übrigbleibenden Fall sind unter Feld (Q[0], 2) nur zwei Felder frei, es muss also damit umgegangen werden, dass das Feld (Q[0], 4) ein Randfeld ist
                    # Da der Hof so rotiert wird, dass die lange Seite die y-Dimension ist, tritt dieser Fall nur für Höfe mit den Maßen (5,5) und einem sich in der Hofmitte befindenenden Feld Q auf
                    self.add_operation(dict(feld0=(0,2), blow_direction=(1,0)))
                    if self.choose_faster_path:
                        self.add_operation(dict(feld0=(2,4), blow_direction=(0,-1)))
                        self.num_rotations += 1
                    self.add_operation(Muster(self, [(1,2)], [dict(feld0=(1,4), blow_direction=(0,-1))], self.tolerated_amount))
                    self.add_operation(Muster(self, [(3,2)], [dict(feld0=(3,4), blow_direction=(0,-1))], self.tolerated_amount))
                    operations = []
                    operations.append(Muster(self, [(1,1),(2,1),(3,1)], [
                        dict(feld0=(0,1), blow_direction=(1,0)),
                        dict(feld0=(4,1), blow_direction=(-1,0)),
                        ], self.tolerated_amount)
                    )
                    operations.append(self.edge_to_mid((2,0), (2,1)))
                    self.add_operation(Muster(self, [(1,1), (2,1), (3,1), (2,0)], operations, self.tolerated_amount))
            if Q != (Q[0], 2):
                # Das Laub befindet sich nach Durchführen der vorherigen Schritte auf (Q[0], 2). Wenn Q[1] > 2, dann muss das Laub noch nach unten geblasen werden. Da über (Q[0], 2) zwei Felder sind, ist das nun verlustfrei durch den Abtrieb bei B vorne problemlos möglich
                for y_start in range(0,Q[1]-2):
                    self.add_operation(Muster(self, [(Q[0], y_start+2)], [dict(feld0=(Q[0], y_start), blow_direction=(0,1))], self.tolerated_amount))


    def build_strategy(self, *, Q):
        """
        Entwirft eine Strategie bzw. einen generalisierten Ablaufplan
        """
        self.strategy = []
        if not self.hof.does_exist(Q):
            print("Q existiert nicht.")
            exit()
        if self.hof.is_edge(Q):
            print("Q darf kein Rand-/Eckfeld sein.")
            exit()
        if self.hof.x_size == 3 and self.hof.y_size == 3:
            return # For Höfe, bei denen eine Dimension kleiner 3 ist, gibt es keine Lösung. Für Höfe der Dimensionen (3,3) ist es am besten, gar nichts zu machen
        # Hof rotieren, sodass sich Feld Q im Bereich oben links befindet (Initialrotation)
        num_rotations = 0
        for i in range(3):
            if self.hof.y_size == 3 and self.hof.x_size != 3:
                break
            if not self.hof.x_size == 3:
                if Q[1] <= math.ceil(self.hof.y_size/2)-1:
                    if Q[0] == 1:
                        break
                    if not ((Q[0] == 1 and Q[1] != 1) or (Q[0] == self.hof.x_size-2 and Q[1] != 1)):
                        if self.hof.x_size <= self.hof.y_size:
                            break
            self.hof.felder_typ_1 = np.rot90(self.hof.felder_typ_1, k=1, axes=(0, 1))
            self.hof.felder_typ_2 = np.rot90(self.hof.felder_typ_2, k=1, axes=(0, 1))
            self.hof.x_size = self.hof.felder_typ_1.shape[0]
            self.hof.y_size = self.hof.felder_typ_1.shape[1]
            Q = (Q[1], self.hof.y_size-1-Q[0])
            num_rotations += 1
        self.num_rotations = num_rotations
        self.Q = Q
        # Strategie speichern
        self.strategy.append("\n[Phase 1: Unterste Reihe entlauben]")
        self.clear_bottom_line()
        self.strategy.append("\n[Phase 2: Laub auf oberste Reihe blasen]")
        self.move_to_top_line()
        self.strategy.append("\n[Phase 3: Laub auf oberster Reihe konzentrieren]")
        self.concentrate_top_line()
        self.strategy.append("\n[Phase 4: Laub nach Feld Q transferieren]")
        self.transfer_to_Q()
        # Initialrotation rückgängig machen
        for i in range(self.num_rotations):
            self.hof.felder_typ_1 = np.rot90(self.hof.felder_typ_1, k=-1, axes=(0, 1))
            self.hof.felder_typ_2 = np.rot90(self.hof.felder_typ_2, k=-1, axes=(0, 1))
            Q = (self.hof.y_size-1-Q[1], Q[0])
            self.hof.x_size = self.hof.felder_typ_1.shape[0]
            self.hof.y_size = self.hof.felder_typ_1.shape[1]
        self.Q = Q

    def step(self, *, render_zwischenschritte=False):
        """
        Führt basierend auf self.strategy die nächste Blasoperation aus
        """
        if self.running_op_index < len(self.strategy):
            operation = self.strategy[self.running_op_index]
            if isinstance(operation, str):
                self.hof.blas_log.append(operation)
                if render_zwischenschritte:
                    self.hof.render(title="Laubverteilung vor:"+operation+f"\n({self.hof.blas_counter} Blasoperationen ausgeführt)", Q=self.Q)
                    self.running_op_index += 1
                    operation = self.strategy[self.running_op_index]
            if isinstance(operation, dict):
                self.hof.blase(operation["feld0"], operation["blow_direction"])
            elif isinstance(operation, Muster):
                run_another_step = operation.step()
                if run_another_step:
                    return
            self.running_op_index += 1

    def run(self, *, render_zwischenschritte=False):
        """
        Ruft self.step() solange auf, bis alle in self.strategy befindlichen Operationen abgearbeitet wurden
        """
        self.running_op_index = 0
        while self.running_op_index < len(self.strategy):
            self.step(render_zwischenschritte=render_zwischenschritte) # Führt basierend auf self.strategy die nächste Blasoperation aus

h = Hof(hof_size, rules_typ_1=Rules(
    use_binomial=use_binomial, binomial_rank="random", # Über use_binomial kann gesteuert werden, ob die Binomialverteilung oder die Erwartungswerte zur Wahrscheinlichkeitsmodellierung verwendet werden sollen
    A_seitenabtrieb = 0.1, B_vorne_abtrieb = 0.1, A_noB_seitenabtrieb=0.5*0.95, s1=0.9, s4=0.05, # Hier können die Blasregeln angepasst werden
    ), rules_typ_2=Rules(
        use_binomial=use_binomial, binomial_rank="random", # Über use_binomial kann gesteuert werden, ob die Binomialverteilung oder die Erwartungswerte zur Wahrscheinlichkeitsmodellierung verwendet werden sollen
        A_seitenabtrieb = 0.3, B_vorne_abtrieb = 0.3, A_noB_seitenabtrieb=0.5*1, s1=0.7, s4=0, # Hier können die Blasregeln angepasst werden
    ), startwert=startwert) # Über startwert kann festgelegt werden, wie viele Blätter am Anfang auf jedem einzelnen Feld liegen
s = Solver2(h, tolerated_amount=tolerated_amount, max_muster_operations=max_muster_operations, choose_faster_path=choose_faster_path) # tolerated_amount ist die tolerierte Blattanzahl auf Feldern, die nicht vollständig geleert werden können.
s.build_strategy(Q=Q) # Hier kann Feld Q festgelegt werden

print("AUSGANGSPUNKT:")
print("Verwendete Wahrscheinlichkeitsmodellierung:")
print("Typ 1:", "binomial" if s.hof.rules_typ_1.use_binomial else "Erwartungswert-basiert")
print("Typ 1:", "binomial" if s.hof.rules_typ_2.use_binomial else "Erwartungswert-basiert")
print("Laubverteilung vor dem Blasprozess:")
print(s.hof)

print("\nFühre Simulation durch ...")
s.run(render_zwischenschritte=True)
print("\nERGEBNIS DER SIMULATION:")
print("Ausgeführte Blasoperationen:", s.hof.blas_counter)
print("Anteil der Blätter auf Q an der Gesamtlaubmenge:", (s.hof.felder_typ_1[s.Q]+s.hof.felder_typ_2[s.Q])*100/(np.sum(s.hof.felder_typ_1)+np.sum(s.hof.felder_typ_2)), "%")
print("Laubverteilung nach dem Blasprozess:")
print(s.hof)

print("\nGENERALISIERTER ABLAUFPLAN:")
print(s)

if output_file != "":
    print("\nDURCHGEFÜHRTE BLASOPERATIONEN:")
    try:
        with open(output_file, "w") as f:
            f.write(s.hof.print_blas_log())
        print(f"In {output_file} gespeichert")
    except Exception as e:
        print("Konnte nicht in Datei schreiben")

s.hof.render(title=f"[S2] Laubverteilung nach dem Blasprozess\n({s.hof.blas_counter} Blasoperationen ausgeführt)", Q=s.Q)
