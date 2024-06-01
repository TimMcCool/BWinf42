# Diese Datei enthält die Klassen Rules und Hof (Modellierung eines Schulhofs und der Regeln, die auf ihm gelten)
# Erweiterung: Es blasen zwei Laubbläser gleichzeitig. Die Blaswirkungen kombinieren sich dabei (siehe Dokumentation)
from binomial_util import binomialpdf, binomialdist, binomial_likeliest
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def manhatten_distance(feld0, feld1) -> int:
    """
    Returns:
        int: Die Manhatten-Distanz zwischen den Tupeln feld0 und feld1, definiert als d = (feld1[0] - feld0[0]) + (feld1[1] - feld1[0])
    """
    return abs(feld1[0] - feld0[0]) + abs(feld1[1] - feld0[1])

class Rules:
    """
    Dient zum Speichern der Regeln, die auf einem Hof für den Laubblasprozess gelten.
    """

    def __init__(self, *, A_seitenabtrieb = 0.1, B_vorne_abtrieb = 0.1, A_noB_seitenabtrieb=0.5, use_binomial=True, binomial_rank="random", binomial_handle_ties="higher"):
        """
        Args:
            use_binomial (boolean): Wenn False, dann werden die Erwartungswerte verwendet bzw. die Wahrscheinlichkeiten auf "Blatt-Mengen" angewendet. Wenn True, dann werden über die Binomialverteilung die Blattmengenveränderungen bestimmt.
            binomial_rank (int oder str): Gibt an, die wievielwahrscheinlichste Blattmengenveränderung verwendet werden soll, wenn die Binomialverteilungswerte verwendet werden. Wenn dieser Wert gleich "random" ist, dann werden die tatsächlichen Wahrscheinlichkeiten simuliert. Hat nur eine Wirkung wenn use_binomial is True
            binomial_handle_ties (str): Gibt an, wie im Zweifelsfall (zwei gleich wahrscheinliche Fälle) verfahren werden soll (größeres oder kleiners k verwenden)
            A_seitenabtrieb = p_A_to_above_B = p_A_to_below_B (siehe Dokumentation)
            B_vorne_abtrieb = p_B_to_beyond_B
            A_noB_seitenabtrieb = p_A_to_above_A = p_A_to_below_A (für Sonderfall 4 und 5 relevant)
        Die default values sind die der Dokumentation entnehmbaren Werte für die Parameter s1 = 0.9 und s4 = 0.05
        """
        # Sicherstellen, dass sich alle Parameter in den zulässigen Bereichen befinden bzw. den richtigen Typ haben:
        assert 0 < A_seitenabtrieb < 0.5 # Implementierungsbedingte Einschränkung: An dieser Stelle wird festgelegt, dass niemals alle Blätter auf abgetrieben werden
        assert 0 < B_vorne_abtrieb < 1
        assert 0 < A_noB_seitenabtrieb <= 0.5 # Implementierungsbedingt wird in diesem Fall allerdings erlaubt, dass das ganze Laub zur Seite wegfliegt
        self.A_seitenabtrieb = A_seitenabtrieb
        self.B_vorne_abtrieb = B_vorne_abtrieb
        self.A_noB_seitenabtrieb = A_noB_seitenabtrieb

        assert isinstance(binomial_rank, int) or binomial_rank == "random"
        assert binomial_handle_ties == "higher" or binomial_handle_ties == "lower" or binomial_handle_ties == "random"
        self.use_binomial = use_binomial
        self.binomial_rank = binomial_rank
        self.binomial_handle_ties = binomial_handle_ties

    def __str__(self):
        """
        Returns:
            str: Die Regeln in lesbarer Form zum Ausgeben in der Konsole
        """
        return "Regeln: " + ", ".join(str(key)+" = "+str(self.__dict__[key]) for key in self.__dict__)

class Hof:
    """
    Repräsentiert einen aus Planquadraten bestehenden Hof, auf dem Laub geblasen werden kann
    """

    def __init__(self, size, rules = Rules(), *, startwert=100, blas_counter=0):
        self.x_size = size[0] # Hof quadratisch -> x_size und y_size sind gleich
        self.y_size = size[1]
        self.rules = rules
        self.startwert = startwert

        # Wenn keine Blattverteilung vorgegeben ist, Blattverteilung / Felderliste initialisieren; die Blattanzahl pro Feld wird durch startwert angegeben
        self.felder = np.full((self.x_size, self.y_size), startwert, dtype=int if self.rules.use_binomial else float)
        self.blas_counter = blas_counter # Gibt die Anzahl an durchgeführten Blasoperationen an
        self.blas_log = [] # Zur Rekonstruktion werden hier alle durchgeführten Blasoperationen gespeichert

    def render(self, *, title="", Q=None, plot_last_op=False):
        """
        Plots the fields of the hof with colors based on the amount of leaves.
        Empty fields are green, fields with less leaves than tolerated amount are orange,
        fields with more leaves than tolerated amount are red.
        The saturation of the color is proportional to the amount of leaves.
        The number indicating the amount of leaves is displayed on each field.
        """
        # Initialize the plot
        plt.figure(figsize=(self.x_size, self.y_size))

        # Iterate over each field and plot it
        max_value = np.amax(self.felder)
        for i in range(self.x_size):
            for j in range(self.y_size):
                # Determine color based on the amount of leaves
                if (i,j) == Q:
                    plt.plot([i, i+1, i+1, i, i], [j, j, j+1, j+1, j], color="black", linewidth=2)
                if self.felder[i][j] == 0:
                    plt.fill([i, i+1, i+1, i], [j, j, j+1, j+1], color="orange", alpha=1)
                else:
                    saturation = min(1, self.felder[i][j] / max_value)

                    # Plot the field with the corresponding color and saturation
                    plt.fill([i, i+1, i+1, i], [j, j, j+1, j+1], color="green", alpha=saturation)

                # Display the number indicating the amount of leaves on the field
                display_number = self.felder[i][j] if isinstance(self.felder[i][j], np.int32) else str(round(self.felder[i][j]*100)/100) # round at two decimals to make the plot more readable
                plt.text(i+0.5, j+0.5, display_number, color='black', ha='center', va='center')

        if plot_last_op:
            blas_log_only_blasops = list(filter(lambda x : isinstance(x, dict), self.blas_log)) # Alles, was keine Blasoperation ist, aus dem Blaslog entfernen
            i = len(blas_log_only_blasops)
            if i > 0:
                blasoperation = blas_log_only_blasops[-1]
                # Plot arrows
                plt.arrow(blasoperation["feld0_b1"][0]+0.5+blasoperation["blow_direction_b1"][0]/3, blasoperation["feld0_b1"][1]+0.5+blasoperation["blow_direction_b1"][1]/3, blasoperation["blow_direction_b1"][0]/3, blasoperation["blow_direction_b1"][1]/3, color='blue', width=0.02)
                plt.arrow(blasoperation["feld0_b2"][0]+0.5+blasoperation["blow_direction_b2"][0]/3, blasoperation["feld0_b2"][1]+0.5+blasoperation["blow_direction_b2"][1]/3, blasoperation["blow_direction_b2"][0]/3, blasoperation["blow_direction_b2"][1]/3, color='blue', width=0.02)
                # Plot circles with numbers
                plt.scatter(blasoperation["feld0_b1"][0]+0.5+blasoperation["blow_direction_b1"][0]/2, blasoperation["feld0_b1"][1]+0.5+blasoperation["blow_direction_b1"][1]/2, color='white', edgecolor='blue', s=200)
                plt.text(blasoperation["feld0_b1"][0]+0.5+blasoperation["blow_direction_b1"][0]/2, blasoperation["feld0_b1"][1]+0.5+blasoperation["blow_direction_b1"][1]/2, str(i), color='blue', ha='center', va='center', fontsize=8 if len(str(i)) < 3 else 4)
                plt.scatter(blasoperation["feld0_b2"][0]+0.5+blasoperation["blow_direction_b2"][0]/2, blasoperation["feld0_b2"][1]+0.5+blasoperation["blow_direction_b2"][1]/2, color='white', edgecolor='blue', s=200)
                plt.text(blasoperation["feld0_b2"][0]+0.5+blasoperation["blow_direction_b2"][0]/2, blasoperation["feld0_b2"][1]+0.5+blasoperation["blow_direction_b2"][1]/2, str(i), color='blue', ha='center', va='center', fontsize=8 if len(str(i)) < 3 else 4)

        # Set axis limits and labels
        plt.xlim(0, self.x_size)
        plt.ylim(0, self.y_size)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.gca().invert_yaxis()

        # Show the plot
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def __str__(self, *, round_digits=5):
        """
        Returns:
            str: Eine Darstellung des Hofs, der entnommen werden kann, wie viele Blätter sich auf den jedem Feld befinden
        """
        output_list = []
        for row in np.transpose(self.felder.round(decimals=round_digits)):
            line_output = ""
            for id, feld in enumerate(row):
                if id % self.x_size == self.x_size-1:
                    output_list.append(line_output + str(feld))
                else:
                    line_output += "{:<8}".format(str(feld)) + "| "
        return "\n".join(output_list)

    def print_blas_log(self):
        output = []
        directions = {(1,0): "rechts", (-1,0): "links", (0,1): "unten", (0,-1): "oben"}
        for op in self.blas_log:
            if isinstance(op, str):
                output.append(op)
            else:
                output.append(f"blase(")
                output.append(f"  BLÄSER1: Feld0: {op['feld0_b1']}, nach: {directions[op['blow_direction_b1']]})")
                output.append(f"  BLÄSER2: Feld0: {op['feld0_b2']}, nach: {directions[op['blow_direction_b2']]})")
                output.append(")")
        return "\n".join(output)

    def copy(self):
        """
        Returns:
            Hof: Eine identische Kopie des Hofs (um Pythons Objektreferenzierung zu umgehen)
        """
        return Hof((self.x_size, self.y_size), self.rules, felder=np.array(self.felder), blas_counter=int(self.blas_counter))

    def is_corner(self, feld : tuple):
        """
        Returns:
            boolean: True wenn das Feld am Index feld (ein Tupel, das Zeile und Spalte angibt) ein Eckfeld ist
        """
        return feld in [(0,0), (self.x_size-1, 0), (0, self.y_size-1), (self.x_size-1, self.y_size-1)]

    def is_edge(self, feld :tuple):
        if self.does_exist(feld):
            return feld[0] in [0, self.x_size-1] or feld[1] in [0, self.y_size-1]
        return False

    def does_exist(self, feld):
        return 0 <= feld[0] < self.x_size and 0 <= feld[1] < self.y_size

    def are_adjacent(self, feldA, feldB):
        """
        Returns:
            boolean: True wenn feldB existiert und ein Nachbarfeld von feldA ist, ansonsten False
        """
        if self.does_exist(feldB):
            return abs(feldA[0] - feldB[0]) + abs(feldA[1] - feldB[1]) == 1
        return False

    def get_orthogonal_direction(self, direction):
        return (0,1) if direction[1] == 0 else (1,0)

    def run_single_blasoperation(self, feld0 : tuple, blow_direction : tuple, blocked_fields : list[tuple], *, remaining_single_ops=0):
        """
        Führt die Blasoperation eines einzigen Laubbläsers aus und aktualisiert self.felder entsprechend.
        Args:
            feld0 (tuple), blow_direction (tuple): Index von Feld 0 und Blasrichtung
            blocked_fields (list[tuple]): Felder, auf die kein Laub gelangen darf (da sie im Einflussgebiets eines anderen Laubbläsers liegen) und die daher wie Randfelder behandelt werden
        """
        if not blow_direction in [(0,1),(0,-1),(1,0),(-1,0)]:
            return # -> Ungültige Blasrichtung. Der Laubbläser kann nur nach rechts, links, oben und unten blasen.

        # Richtung, die orthogonal zur Blasrichtung ist, ermitteln:
        orthogonal_direction = self.get_orthogonal_direction(blow_direction)
        # Feld A (Feld unmittelbar vor dem Laubbläser) ermitteln:
        feldA = (feld0[0]+blow_direction[0], feld0[1]+blow_direction[1])
        if not self.does_exist(feldA):
            return # -> Es gibt kein Feld vor dem Laubbläser bzw. der Laubbläser bläst gegen die Umrandung. Für diesen Fall ist definiert, dass sich die Verteilung des Laubs nicht verändert
        new_feldA_value = 0 # In dieser Variable wird die neue Anzahl an Blättern auf Feld A gespeichert

        # Feld B (Feld hinter Feld A) ermitteln:
        feldB = (feldA[0]+blow_direction[0], feldA[1]+blow_direction[1])
        if self.does_exist(feldB) and feldB not in blocked_fields:
            # -> Es gibt ein Feld B. Neue Anzahl an Blättern auf Feld B ermitteln:
            if self.rules.use_binomial:
                # -> Zum Modellieren der Blätteranzahlen sollen ganze Zahlen verwendet werden, die über die Binomialverteilung bestimmt werden
                A_seitenabtrieb_1, _ = binomial_likeliest(n=self.felder[feldA], p=self.rules.A_seitenabtrieb, rank=self.rules.binomial_rank, handle_ties=self.rules.binomial_handle_ties) # Anzahl an Blättern von Feld A, die auf dem einen Feld neben Feld B landen (Seitenabtrieb 1)
                A_seitenabtrieb_2, _ = binomial_likeliest(n=self.felder[feldA]-A_seitenabtrieb_1, p=self.rules.A_seitenabtrieb/(1-self.rules.A_seitenabtrieb), rank=self.rules.binomial_rank, handle_ties=self.rules.binomial_handle_ties) # Anzahl an Blättern von Feld A, die auf dem anderen Feld neben Feld B landen (Seitenabtrieb 2)
                B_vorne_abtrieb, _ = binomial_likeliest(n=self.felder[feldB], p=self.rules.B_vorne_abtrieb, rank=self.rules.binomial_rank, handle_ties=self.rules.binomial_handle_ties) # Anzahl an Blättern von Feld B, die nach vorne abgetrieben werden
            else:
                # -> Zum Modellieren der Blätteranzahlen werden die Erwartungswerte (als Fließkommazahlen) verwendet
                A_seitenabtrieb_1 = self.felder[feldA] * self.rules.A_seitenabtrieb
                A_seitenabtrieb_2 = A_seitenabtrieb_1
                B_vorne_abtrieb = self.felder[feldB] * self.rules.B_vorne_abtrieb
            new_feldB_value = self.felder[feldA] - (A_seitenabtrieb_1 + A_seitenabtrieb_2) + self.felder[feldB] - B_vorne_abtrieb # Neue Anzahl an Blättern auf Feld B

            # Nachbarfeld von Feld B, das Feld A gegenüberliegt, aktualisieren (sofern vorhanden):
            n1 = (feldB[0]+blow_direction[0], feldB[1]+blow_direction[1]) # Index des Nachbarfelds
            if self.does_exist(n1) and n1 not in blocked_fields:
                self.felder[n1] += B_vorne_abtrieb
            else:
                new_feldB_value += B_vorne_abtrieb

            # Nachbarfelder von Feld B, die Feld A nicht gegenüberliegen, aktualisieren (sofern vorhanden):
            n1 = (feldB[0]+orthogonal_direction[0], feldB[1]+orthogonal_direction[1])
            if self.does_exist(n1) and n1 not in blocked_fields:
                self.felder[n1] += A_seitenabtrieb_1
            else:
                new_feldB_value += A_seitenabtrieb_1
            n2 = (feldB[0]-orthogonal_direction[0], feldB[1]-orthogonal_direction[1])
            if self.does_exist(n2) and n2 not in blocked_fields:
                self.felder[n2] += A_seitenabtrieb_2
            else:
                new_feldB_value += A_seitenabtrieb_2

            # Feld B aktualsieren
            self.felder[feldB] = new_feldB_value
        else:
            # -> Es gibt kein Feld B.
            if self.rules.use_binomial:
                A_noB_seitenabtrieb_1, _ = binomial_likeliest(n=self.felder[feldA], p=self.rules.A_noB_seitenabtrieb, rank=self.rules.binomial_rank, handle_ties=self.rules.binomial_handle_ties)
                A_noB_seitenabtrieb_2, _ = binomial_likeliest(n=self.felder[feldA]-A_noB_seitenabtrieb_1, p=self.rules.A_noB_seitenabtrieb/(1-self.rules.A_noB_seitenabtrieb), rank=self.rules.binomial_rank, handle_ties=self.rules.binomial_handle_ties)
            else:
                A_noB_seitenabtrieb_1 = self.felder[feldA] * self.rules.A_noB_seitenabtrieb
                A_noB_seitenabtrieb_2 = A_noB_seitenabtrieb_1
            new_feldA_value = self.felder[feldA] - (A_noB_seitenabtrieb_1 + A_noB_seitenabtrieb_2)

            # Nachbarfelder von Feld A, die dem laubblasenden Hausmeister nicht gegenüberliegen, aktualisieren, sofern vorhanden:
            n1 = (feldA[0]+orthogonal_direction[0], feldA[1]+orthogonal_direction[1])
            if self.does_exist(n1) and n1 not in blocked_fields:
                self.felder[n1] += A_noB_seitenabtrieb_1
            else:
                # -> Feld A ist ein Eckfeld, ein Teil des Laubs verbleibt also auf Feld A.
                new_feldA_value += A_noB_seitenabtrieb_1
            n2 = (feldA[0]-orthogonal_direction[0], feldA[1]-orthogonal_direction[1])
            if self.does_exist(n2) and n2 not in blocked_fields:
                self.felder[n2] += A_noB_seitenabtrieb_2
            else:
                # -> Feld A ist ein Eckfeld, ein Teil des Laubs verbleibt also auf Feld A.
                new_feldA_value += A_noB_seitenabtrieb_2

        # Neuen Wert von A zurückgeben (Aktualisierung des Feldwerts erfolgt am Ende des insgesamten Blasvorgangs, damit die verschiedenen Blasvorgänge alle vom selben Ausgangspunkt ausgehen):
        return new_feldA_value

    def blase(self, feld0_b1, blow_direction_b1, feld0_b2, blow_direction_b2):
        """
        Simuliert einen Blasvorgang (mit zwei Laubbläsern) und aktualisiert self.felder auf die resultierende Blattverteilung.
        Args:
            feld0_b1 (int): Index des Felds, auf dem der Hausmeister 1 mit Bläser 1 steht
            blow_direction_b1 (tuple): Richtung, in die der Hausmeister 1 mit Bläser 1 bläst. Kann folgende Werte annehmen: (1,0) =rechts, (-1,0) (=links), (0,1) (=unten), (0,-1) (=oben)
            feld0_b2 (int): Index des Felds, auf dem der Hausmeister 2 mit Bläser 2 steht
            blow_direction_b2 (tuple): Richtung, in die der Hausmeister 2 mit Bläser 2 bläst. Kann folgende Werte annehmen: (1,0) =rechts, (-1,0) (=links), (0,1) (=unten), (0,-1) (=oben)
        """
        if feld0_b2 is None:
            feldA_b1 = (feld0_b1[0]+blow_direction_b1[0], feld0_b1[1]+blow_direction_b1[1])
            self.felder[feldA_b1] = self.run_single_blasoperation(feld0_b1, blow_direction_b1, [])
            return
        if feld0_b1 == feld0_b2:
            return None # Laubbläser dürfen nicht auf selbem Feld stehen
        pd = [(0,1),(0,-1),(1,0),(-1,0)] # Mögliche Blasrichtungen
        if not blow_direction_b1 in pd and blow_direction_b2 in pd:
            return # -> Ungültige Blasrichtung. Der Laubbläser kann nur nach rechts, links, oben und unten blasen.
        self.blas_counter += 1 # Zähler, der durchgeführte Blasoperationen zählt, erhöhen
        self.blas_log.append(dict(feld0_b1=feld0_b1, blow_direction_b1=blow_direction_b1, feld0_b2=feld0_b2, blow_direction_b2=blow_direction_b2)) # Blasoperation loggen

        feldA_b1 = (feld0_b1[0]+blow_direction_b1[0], feld0_b1[1]+blow_direction_b1[1])
        if not self.does_exist(feldA_b1): # Überprüfen, ob das ermittelte Feld tatsächlich existiert
            feldA_b1 = None
        feldA_b2 = (feld0_b2[0]+blow_direction_b2[0], feld0_b2[1]+blow_direction_b2[1])
        if not self.does_exist(feldA_b2):
            feldA_b2 = None
        if feldA_b1 == feld0_b2 and feld0_b1 == feldA_b2:
            return

        # Blasoperation für den ersten Laubbläser (b1) ausführen:
        if feldA_b2 == feldA_b1:
            half_feldA_value = round(self.felder[feldA_b2]/2)
            self.felder[feldA_b2] -= half_feldA_value
        blocked_fields = [feldA_b2]
        if manhatten_distance(feld0_b2, feld0_b1) > manhatten_distance(feld0_b1, (feld0_b2[0]+blow_direction_b2[0], feld0_b2[1]+blow_direction_b2[1])):
            blocked_fields.append(feld0_b2)
        new_feldA_b1_value = self.run_single_blasoperation(feld0_b1, blow_direction_b1, blocked_fields)
        if feldA_b2 == feldA_b1:
            self.felder[feldA_b2] = half_feldA_value
        blocked_fields = [feldA_b1]
        if manhatten_distance(feld0_b1, feld0_b2) > manhatten_distance(feld0_b2, (feld0_b1[0]+blow_direction_b1[0], feld0_b1[1]+blow_direction_b1[1])):
            blocked_fields.append(feld0_b1)
        new_feldA_b2_value = self.run_single_blasoperation(feld0_b2, blow_direction_b2, blocked_fields)

        if feldA_b2 == feldA_b1:
            self.felder[feldA_b1] = new_feldA_b1_value + new_feldA_b2_value
        else:
            if not new_feldA_b1_value is None:
                self.felder[feldA_b1] = new_feldA_b1_value
            if not new_feldA_b2_value is None:
                self.felder[feldA_b2] = new_feldA_b2_value

h = Hof((5,5),startwert=100)
h.blase((0,2),(1,0),(2,3),(-1,0))
h.render()
