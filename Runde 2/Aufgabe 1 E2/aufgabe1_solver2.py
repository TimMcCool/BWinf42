# Erweiterung 2: Es blasen zwei Laubbläser gleichzeitig. Die Blaswirkungen kombinieren sich dabei (siehe Dokumentation)
# Die Funktionen und Attr. der Hof-Klasse wurden modifiziert und die Strategie angepasst (mit zwei Laubbläsern kann das Laub schneller auf B gebracht werden, wenn man die Bläser geschickt kombiniert)
import math
from hof import Rules, Hof # Importieren der Klassen aus hof.py
import numpy as np

# Hyperparameter festlegen
Q = (2,2) # Index von Feld Q festlegen
hof_size = (5,5) # Hofseitenlängen festlegen
use_binomial = True # Festlegen, ob die Wahrscheinlichkeiten basierend auf der Binomialverteilung simuliert oder ob die Erwartungswerte verwendet werden sollen
tolerated_amount = 5 # Blattmenge, die auf nicht vollständig leerbaren Feldern als vernachlässigbar gilt
max_muster_operations = 5000 # Anzahl an Operation, die pro Muster maximal durchgeführt werden
startwert = 100 # Anfangsanzahl an Blättern pro Feld
choose_faster_path = True
# Datei zum Speichern der durchgeführten Blasvorgänge festlegen:
output_file = ""

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
        if isinstance(self.operations[self.next_op_index], dict):
            # -> Eine Blasoperation liegt vor, die ausgeführt wird
            if "other" in self.operations[self.next_op_index]:
                op2 = self.operations[self.next_op_index]["other"]()
                self.hof.blase(self.operations[self.next_op_index]["feld0_b1"], self.operations[self.next_op_index]["blow_direction_b1"], op2["feld0"], op2["blow_direction"])
            else:
                self.hof.blase(self.operations[self.next_op_index]["feld0_b1"], self.operations[self.next_op_index]["blow_direction_b1"], self.operations[self.next_op_index]["feld0_b2"], self.operations[self.next_op_index]["blow_direction_b2"])
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
        self.idle_operation = [] # Hier werden Operationen gespeichert, die ausgeführt werden können, wenn einer der Laubbläser gerade nicht gebraucht wird, damit dieser Laubbläser zumindest noch ein bisschen was bringt
        self.leaves_sum = np.sum(self.hof.felder) # Gesamtlaubmenge auf dem Hof

    def next_idle_operation(self):
        """
        Wählt die "idle_operation", die im aktuellen Zustand am meisten bringt
        """
        def score(x):
            return self.hof.felder[(x["feld0"][0]+x["blow_direction"][0],x["feld0"][1]+x["blow_direction"][1])]
        if len(self.idle_operation) == 0:
            return dict(feld0=None, blow_direction=None) # Wenn es keine "Idle-Operationen" gibt
        return max(self.idle_operation, key=score)

    def add_operation(self, operation):
        """
        Fügt eine Blasoperation oder ein Muster zum Ablaufplan hinzu. Macht zuvor die Initialrotation rückgängig (beim späteren Ausführen wird der Hof in seiner ursprünglichen Ausrichtung betrachtet, die Rotation erfolgt nur, da dies die Generierung des generalisierten Ablaufplans / der Strategie erleichtert).
        """
        self.strategy.append(self.rotate_blasoperation(operation) if isinstance(operation, dict) else operation)

    def rotate_field_index(self, field_index) -> tuple:
        """
        Ermittelt den ursprünglichen Index des Felds (den es vor der Initialrotation hatte), das derzeit am Index field_index ist
        """
        if field_index is None:
            return None
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
        if vector is None:
            return vector
        for r in range(self.num_rotations):
            # Vektor um 90° im Uhrzeigersinn rotieren:
            if vector[0] == 0:
                vector = (-vector[1],0)
            elif vector[1] == 0:
                vector = (0,vector[0])
        return vector

    def rotate_blasoperation(self, blasoperation) -> dict:
        if "other" in blasoperation:
            return dict(
                feld0_b1=self.rotate_field_index(blasoperation["feld0_b1"]),
                blow_direction_b1=self.rotate_direction_vector(blasoperation["blow_direction_b1"]),
                other=blasoperation["other"]
            )
        else:
            return dict(
                feld0_b1=self.rotate_field_index(blasoperation["feld0_b1"]),
                blow_direction_b1=self.rotate_direction_vector(blasoperation["blow_direction_b1"]),
                feld0_b2=self.rotate_field_index(blasoperation["feld0_b2"]),
                blow_direction_b2=self.rotate_direction_vector(blasoperation["blow_direction_b2"])
            )

    def stringify_operation_sequence(self, sequence, *, indent=0) -> list[str]:
        """
        Wandelt die Operation in einen formatierten, gut lesbaren String um
        """
        output = []
        directions = {(1,0): "rechts", (-1,0): "links", (0,1): "unten", (0,-1): "oben", None : "-"} # Die möglichen Blasrichtungen als Fließtext
        for op in sequence:
            if isinstance(op, dict):
                if "other" in op:
                    output.append("  "*indent+f"blase(\n"+"  "*(indent+1)+f"BLÄSER1: Feld0: {op['feld0_b1']}, nach: {directions[op['blow_direction_b1']]}"+"  "*(indent+1)+f"BLÄSER2: Idle-Operation)")
                else:
                    output.append("  "*indent+f"blase(\n"+"  "*(indent+1)+f"BLÄSER1: Feld0: {op['feld0_b1']}, nach: {directions[op['blow_direction_b1']]}"+"  "*(indent+1)+f"BLÄSER2: Feld0: {op['feld0_b2']}, nach: {directions[op['blow_direction_b2']]})")
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

    def ops_corner_to_edge(self, source_corner_field, target_field):
        """
        Returns:
            dict: Die Blasoperation, die 1 Laubbläser wiederholt durchführen muss, um ein Eckfeld zu leeren
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
        return dict(feld0=feld0, blow_direction=blow_direction)

    def corners_to_edge(self, source_corner_field_b1, source_corner_field_b2, target_field_b1, target_field_b2) -> Muster:
        """
        Returns:
            Muster: Ein Muster, das bei Anwendung Laub von den Eckfeldern source_corner_field_b1 und source_corner_field_b2 auf die Randfelder target_field_b1 (gehört zu source_corner_field_b1) und target_field_b2 (gehört zum anderen Source-Feld) bläst
        """
        op1 = self.ops_corner_to_edge(source_corner_field_b1, target_field_b1)
        if target_field_b2 is not None:
            op2 = self.ops_corner_to_edge(source_corner_field_b2, target_field_b2)
            source_fields = [source_corner_field_b1, source_corner_field_b2]
        else:
            op2 = dict(feld0=None, blow_direction=None)
            source_fields = [source_corner_field_b1]

        muster = Muster(self, source_fields, [dict(feld0_b1=op1["feld0"], blow_direction_b1=op1["blow_direction"], feld0_b2=op2["feld0"], blow_direction_b2=op2["blow_direction"])], self.tolerated_amount) # Muster Objekt erstellen
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
        blow_direction = self.hof.get_orthogonal_direction(orthogonal_direction)
        # Die Blasvorgänge, die Laub durch Seitenabtriebe auf target_field transportieren, ermitteln
        feld0_1 = (source_edge_field[0]+blow_direction[0], source_edge_field[1]+blow_direction[1])
        feld0_2 = (source_edge_field[0]-blow_direction[0], source_edge_field[1]-blow_direction[1])
        muster = Muster(self, [source_edge_field], [dict(feld0_1, (-blow_direction[0], -blow_direction[1]), feld0_2, blow_direction)], self.tolerated_amount)
        return muster

    def clear_bottom_line(self):
        """
        1. Phase des generalisierten Ablaufs: Befreien der untersten Reihe des Hofs. Fügt die hierfür notwendigen Operationen zu self.strategy hinzu.
        """
        self.add_operation(dict(
            feld0_b1 = (math.floor(self.hof.x_size/2)-2,self.hof.y_size-1),
            blow_direction_b1 = (1,0),
            other = self.next_idle_operation,
        ))
        self.add_operation(dict(
            feld0_b1 = (math.floor(self.hof.x_size/2)-1,self.hof.y_size-1),
            blow_direction_b1 = (1,0),
            other = self.next_idle_operation,
        ))
        if self.hof.x_size % 2 == 1: # In diesem Fall wird 1 Feld zu wenig geleert (durch die folgende for-Schleife)
            self.add_operation(dict(
                feld0_b1 = (math.floor(self.hof.x_size/2)+0,self.hof.y_size-1),
                blow_direction_b1 = (1,0),
                other = self.next_idle_operation,
            ))
        for offset in range(math.ceil(self.hof.x_size/2),1,-1):
            self.add_operation(dict(
                feld0_b1 = (offset,self.hof.y_size-1),
                blow_direction_b1 = (-1,0),
                feld0_b2 = (self.hof.x_size-offset-1,self.hof.y_size-1),
                blow_direction_b2 = (1,0),
            ))
        self.add_operation(self.corners_to_edge((0, self.hof.y_size-1),(self.hof.x_size-1, self.hof.y_size-1),(0,self.hof.y_size-2),(self.hof.x_size-1,self.hof.y_size-2))) # Ecke rechts unten leeren

    def move_to_top_line(self):
        """
        2. Phase des generalisierten Ablaufs: Bläst das gesamte Laub in die oberste Reihe. Fügt die hierfür notwendigen Operationen zu self.strategy hinzu.
        """
        ops = []
        for start_y in range(self.hof.y_size-1,1,-1):
            for start_x in range(self.hof.x_size-1,-1,-1):
                if not (start_x == self.Q[0] and (start_y <= self.Q[1]+1 or self.hof.y_size == 5)): # Feld Q und Felder über Q nicht unnötigerweise leeren, da hier nachher das Laub sowieder wieder hintransportiert werden muss
                    ops.append(dict(feld0=(start_x, start_y), blow_direction=(0,-1)))
                    if len(ops) == 2:
                        self.add_operation(dict(
                            feld0_b1 = ops[0]["feld0"],
                            blow_direction_b1 = ops[0]["blow_direction"],
                            feld0_b2 = ops[1]["feld0"],
                            blow_direction_b2 = ops[1]["blow_direction"],
                        ))
                        ops = []
        if ops != []:
            self.add_operation(dict(
                feld0_b1 = ops[0]["feld0"],
                blow_direction_b1 = ops[0]["blow_direction"],
                feld0_b2 = None,
                blow_direction_b2 = None
            ))

    def concentrate_top_line(self):
        """
        3. Phase des generalisierten Ablaufs: Konzentriert das Laub der obersten Reihe auf dem Randfeld, das sich in der selben Spalte wie Q befindet. Fügt die hierfür notwendigen Operationen zu self.strategy hinzu.
        """
        Q = self.Q
        self.add_operation(self.corners_to_edge((0,0),(self.hof.x_size-1,0),(1,0),(self.hof.x_size-2,0)))
        # Nicht-Eckfelder der obersten Reihe auf Source-Felder konzentrieren:
        for start_x in range(0,self.hof.x_size-1):
            if start_x >= self.Q[0]-2:
                break
            else:
                self.add_operation(dict(
                    feld0_b1 = (start_x,0),
                    blow_direction_b1 = (1,0),
                    feld0_b2 = (start_x+2,2),
                    blow_direction_b2 = (0,-1),
                ))
        for start_x in range(self.hof.x_size-1,0,-1):
            if start_x <= self.Q[0]+2:
                break
            else:
                self.add_operation(dict(
                    feld0_b1 = (start_x,0),
                    blow_direction_b1 = (-1,0),
                    feld0_b2 = (start_x-2,2),
                    blow_direction_b2 = (0,-1),
                ))
        if self.hof.does_exist((self.Q[0]+2,0)):
            feld0_b1 = (self.Q[0]+2,0)
        else:
            self.add_operation(dict(
                feld0_b1 = (self.Q[0]-2,0),
                blow_direction_b1 = (1,0),
                feld0_b2 = (self.Q[0]+1,1),
                blow_direction_b2 = (0,-1),
            ))
            feld0_b1 = None
        if self.hof.does_exist((self.Q[0]-2,0)):
            feld0_b2 = (self.Q[0] -2 , 0)
        else:
            self.add_operation(dict(
                feld0_b1 = (self.Q[0]+2,0),
                blow_direction_b1 = (-1,0),
                feld0_b2 = (self.Q[0]-1,1),
                blow_direction_b2 = (0,-1),
            ))
            feld0_b2 = None
        if feld0_b1 is not None and feld0_b2 is not None:
            self.add_operation(dict(
                feld0_b1 = feld0_b1,
                blow_direction_b1 = (-1,0),
                feld0_b2 = feld0_b2,
                blow_direction_b2 = (1,0),
            ))
        elif feld0_b1 is not None:
            self.add_operation(self.corners_to_edge((self.hof.x_size-1,0),None,(self.hof.x_size-2,0), None))
        elif feld0_b2 is not None:
            self.add_operation(self.corners_to_edge((0,0),None,(1,0), None))

    def transfer_to_Q(self):
        """
        4. Phase des generalisierten Ablaufs: Verschiebt das gesamte Laub auf Feld Q. Fügt die hierfür notwendigen Operationen zu self.strategy hinzu.
        """
        operations = []
        # Großteil des Laubs von Feld (Q[0],0) auf Feld (Q[0],1) bringen
        operations.append(Muster(
            self, [(self.Q[0],0)], [dict(
                feld0_b1=(self.Q[0]-1,0), blow_direction_b1=(1,0), feld0_b2=(self.Q[0]+1,0), blow_direction_b2=(-1,0))
                ], self.leaves_sum * 0.2
        ))
        if self.Q[1] == 1:
            self.strategy += operations
        else:
            # Laub auf von (Q[0],1) auf Felder (Q[0],0) und (Q[0],2) aufteilen:
            operations.append(dict(feld0_b1=(self.Q[0]-1,1), blow_direction_b1=(1,0), feld0_b2=(self.Q[0]+1,1), blow_direction_b2=(-1,0)))
            # Bei jedem Durchlauf des Musters wird mehr Laub auf (Q[0],2) gebracht
            self.add_operation(Muster(self, [(self.Q[0],0),(self.Q[0],1)], operations, self.tolerated_amount))
            if self.Q != (self.Q[0], 2):
                # Das Laub befindet sich nach Durchführen der vorherigen Schritte auf (Q[0], 2)
                for y_start in range(0,self.Q[1]-2):
                    self.add_operation(Muster(self, [(self.Q[0], y_start+2)], [dict(feld0_b1=(self.Q[0], y_start), blow_direction_b1=(0,1), feld0_b2=None, blow_direction_b2=None)], self.tolerated_amount)) # Hier kann nur einer der beiden Bläser sinnvoll eingesetzt werden

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
        # Idle-Felder ermitteln (die Felder werden im nicht-rotierten Zustand ermittelt und gespeichert)
        if self.hof.does_exist((Q[0],Q[1]+2)):
            self.idle_operation.append(dict(feld0=(Q[0],Q[1]+2), blow_direction=(0,-1)))
        if self.hof.does_exist((Q[0],Q[1]-2)):
            self.idle_operation.append(dict(feld0=(Q[0],Q[1]-2), blow_direction=(0,1)))
        if self.hof.does_exist((Q[0]-2,Q[1])):
            self.idle_operation.append(dict(feld0=(Q[0]-2,Q[1]), blow_direction=(1,0)))
        if self.hof.does_exist((Q[0]+2,Q[1])):
            self.idle_operation.append(dict(feld0=(Q[0]+2,Q[1]), blow_direction=(-1,0)))
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
            self.hof.felder = np.rot90(self.hof.felder, k=1, axes=(0, 1))
            self.hof.x_size = self.hof.felder.shape[0]
            self.hof.y_size = self.hof.felder.shape[1]
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
            self.hof.felder = np.rot90(self.hof.felder, k=-1, axes=(0, 1))
            Q = (self.hof.y_size-1-Q[1], Q[0])
            self.hof.x_size = self.hof.felder.shape[0]
            self.hof.y_size = self.hof.felder.shape[1]
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
                if "other" in operation:
                    op2 = operation["other"]()
                    self.hof.blase(operation["feld0_b1"], operation["blow_direction_b1"], op2["feld0"], op2["blow_direction"])
                else:
                    self.hof.blase(operation["feld0_b1"], operation["blow_direction_b1"], operation["feld0_b2"], operation["blow_direction_b2"])
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

h = Hof(hof_size, Rules(
    use_binomial=use_binomial, binomial_rank="random", # Über use_binomial kann gesteuert werden, ob die Binomialverteilung oder die Erwartungswerte zur Wahrscheinlichkeitsmodellierung verwendet werden sollen
    A_seitenabtrieb = 0.1, B_vorne_abtrieb = 0.1, A_noB_seitenabtrieb=0.5, # Hier können die Blasregeln angepasst werden
    ), startwert=startwert) # Über startwert kann festgelegt werden, wie viele Blätter am Anfang auf jedem einzelnen Feld liegen
s = Solver2(h, tolerated_amount=tolerated_amount, max_muster_operations=max_muster_operations, choose_faster_path=choose_faster_path) # tolerated_amount ist die tolerierte Blattanzahl auf Feldern, die nicht vollständig geleert werden können.
s.build_strategy(Q=Q) # Hier kann Feld Q festgelegt werden

print("AUSGANGSPUNKT:")
print("Verwendete Wahrscheinlichkeitsmodellierung:", "binomial" if s.hof.rules.use_binomial else "Erwartungswert-basiert")
print("Laubverteilung vor dem Blasprozess:")
print(s.hof)

print("\nFühre Simulation durch ...")
s.run(render_zwischenschritte=True)
print("\nERGEBNIS DER SIMULATION:")
print("Ausgeführte Blasoperationen:", s.hof.blas_counter)
print("Anteil der Blätter auf Q an der Gesamtlaubmenge:", s.hof.felder[s.Q]*100/s.leaves_sum, "%")
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
