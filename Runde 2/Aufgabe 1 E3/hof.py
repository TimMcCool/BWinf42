# Diese Datei enthält die Klassen Rules und Hof (Modellierung eines Schulhofs und der Regeln, die auf ihm gelten)

from binomial_util import binomialpdf, binomialdist, binomial_likeliest
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class Rules:
    """
    Dient zum Speichern der Regeln, die auf einem Hof für den Laubblasprozess gelten.
    """

    def __init__(self, *, A_seitenabtrieb = 0.1, B_vorne_abtrieb = 0.1, A_noB_seitenabtrieb=0.5*0.95, s1=0.9, s4=0.05, use_binomial=True, binomial_rank="random", binomial_handle_ties="higher"):
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
        self.s1 = s1
        self.s4 = s4
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

    def __init__(self, size, *, rules_typ_1 = Rules(), rules_typ_2 = Rules(), startwert=100, blas_counter=0):
        self.x_size = size[0] # Hof quadratisch -> x_size und y_size sind gleich
        self.y_size = size[1]
        self.rules_typ_1 = rules_typ_1
        self.rules_typ_2 = rules_typ_2
        self.startwert = startwert

        self.felder_typ_1 = np.full((self.x_size, self.y_size), startwert, dtype=int if self.rules_typ_1.use_binomial else float)
        self.felder_typ_2 = np.full((self.x_size, self.y_size), startwert, dtype=int if self.rules_typ_2.use_binomial else float)
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
        max_value_1 = np.amax(self.felder_typ_1)
        max_value_2 = np.amax(self.felder_typ_2)
        for i in range(self.x_size):
            for j in range(self.y_size):
                # Determine color based on the amount of leaves
                if (i,j) == Q:
                    plt.plot([i, i+1, i+1, i, i], [j, j, j+1, j+1, j], color="black", linewidth=2)
                if self.felder_typ_1[i][j] == 0:
                    plt.fill([i, i+0.5, i+0.5, i], [j, j, j+1, j+1], color="orange", alpha=1)
                else:
                    saturation = min(1, self.felder_typ_1[i][j] / max_value_1)

                    # Plot the field with the corresponding color and saturation
                    plt.fill([i, i+1, i+1, i], [j, j, j+1, j+1], color="green", alpha=saturation)
                if self.felder_typ_2[i][j] == 0:
                    plt.fill([i+0.5, i+1, i+1, i+0.5], [j, j, j+1, j+1], color="orange", alpha=1)
                else:
                    saturation = min(1, self.felder_typ_2[i][j] / max_value_2)

                    # Plot the field with the corresponding color and saturation
                    plt.fill([i+0.5, i+1, i+1, i+0.5], [j, j, j+1, j+1], color="green", alpha=saturation)

                # Display the number indicating the amount of leaves on the field
                display_number = self.felder_typ_1[i][j] if isinstance(self.felder_typ_1[i][j], np.int32) else str(round(self.felder_typ_1[i][j]*100)/100) # round at two decimals to make the plot more readable
                plt.text(i+0.25, j+0.5, display_number, color='black', ha='center', va='center')
                display_number = self.felder_typ_2[i][j] if isinstance(self.felder_typ_2[i][j], np.int32) else str(round(self.felder_typ_2[i][j]*100)/100) # round at two decimals to make the plot more readable
                plt.text(i+0.75, j+0.5, display_number, color='black', ha='center', va='center')

        if plot_last_op:
            blas_log_only_blasops = list(filter(lambda x : isinstance(x, dict), self.blas_log)) # Alles, was keine Blasoperation ist, aus dem Blaslog entfernen
            i = len(blas_log_only_blasops)
            if i > 0:
                blasoperation = blas_log_only_blasops[-1]
                # Plot arrows
                plt.arrow(blasoperation["feld0"][0]+0.5+blasoperation["blow_direction"][0]/3, blasoperation["feld0"][1]+0.5+blasoperation["blow_direction"][1]/3, blasoperation["blow_direction"][0]/3, blasoperation["blow_direction"][1]/3, color='blue', width=0.02)
                # Plot circles with numbers
                plt.scatter(blasoperation["feld0"][0]+0.5+blasoperation["blow_direction"][0]/2, blasoperation["feld0"][1]+0.5+blasoperation["blow_direction"][1]/2, color='white', edgecolor='blue', s=200)
                plt.text(blasoperation["feld0"][0]+0.5+blasoperation["blow_direction"][0]/2, blasoperation["feld0"][1]+0.5+blasoperation["blow_direction"][1]/2, str(i), color='blue', ha='center', va='center', fontsize=8 if len(str(i)) < 3 else 4)

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
        output_list = ["Blattverteilung der Blätter vom Typ 1:"]
        for row in np.transpose(self.felder_typ_1.round(decimals=round_digits)):
            line_output = ""
            for id, feld in enumerate(row):
                if id % self.x_size == self.x_size-1:
                    output_list.append(line_output + str(feld))
                else:
                    line_output += "{:<8}".format(str(feld)) + "| "
        output_list.append("Blattverteilung der Blätter vom Typ 2:")
        for row in np.transpose(self.felder_typ_2.round(decimals=round_digits)):
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
                output.append(f"blase(Feld0: {op['feld0']}, nach: {directions[op['blow_direction']]})")
        return "\n".join(output)

    def copy(self):
        """
        Returns:
            Hof: Eine identische Kopie des Hofs (um Pythons Objektreferenzierung zu umgehen)
        """
        return Hof((self.x_size, self.y_size), felder_typ_1=np.array(self.felder_typ_1), felder_typ_2=np.array(self.felder_typ_2), rules_typ_1=self.rules_typ_1, rules_typ_2=self.rules_typ_2, blas_counter=int(self.blas_counter))

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

    def blase_blattyp(self, feld0, blow_direction, *, ruleset, felder):
        # Richtung, die orthogonal zur Blasrichtung ist, ermitteln:
        orthogonal_direction = self.get_orthogonal_direction(blow_direction)
        # Feld A (Feld unmittelbar vor dem Laubbläser) ermitteln:
        feldA = (feld0[0]+blow_direction[0], feld0[1]+blow_direction[1])
        if not self.does_exist(feldA):
            return # -> Es gibt kein Feld vor dem Laubbläser bzw. der Laubbläser bläst gegen die Umrandung. Für diesen Fall ist definiert, dass sich die Verteilung des Laubs nicht verändert
        new_feldA_value = 0 # In dieser Variable wird die neue Anzahl an Blättern auf Feld A gespeichert

        # Feld B (Feld hinter Feld A) ermitteln:
        feldB = (feldA[0]+blow_direction[0], feldA[1]+blow_direction[1])
        if self.does_exist(feldB):
            # -> Es gibt ein Feld B. Neue Anzahl an Blättern auf Feld B ermitteln:
            if ruleset.use_binomial:
                # -> Zum Modellieren der Blätteranzahlen sollen ganze Zahlen verwendet werden, die über die Binomialverteilung bestimmt werden
                A_seitenabtrieb_1, _ = binomial_likeliest(n=felder[feldA], p=ruleset.A_seitenabtrieb, rank=ruleset.binomial_rank, handle_ties=ruleset.binomial_handle_ties) # Anzahl an Blättern von Feld A, die auf dem einen Feld neben Feld B landen (Seitenabtrieb 1)
                A_seitenabtrieb_2, _ = binomial_likeliest(n=felder[feldA]-A_seitenabtrieb_1, p=ruleset.A_seitenabtrieb/(1-ruleset.A_seitenabtrieb), rank=ruleset.binomial_rank, handle_ties=ruleset.binomial_handle_ties) # Anzahl an Blättern von Feld A, die auf dem anderen Feld neben Feld B landen (Seitenabtrieb 2)
                B_vorne_abtrieb, _ = binomial_likeliest(n=felder[feldB], p=ruleset.B_vorne_abtrieb, rank=ruleset.binomial_rank, handle_ties=ruleset.binomial_handle_ties) # Anzahl an Blättern von Feld B, die nach vorne abgetrieben werden
            else:
                # -> Zum Modellieren der Blätteranzahlen werden die Erwartungswerte (als Fließkommazahlen) verwendet
                A_seitenabtrieb_1 = felder[feldA] * ruleset.A_seitenabtrieb
                A_seitenabtrieb_2 = A_seitenabtrieb_1
                B_vorne_abtrieb = felder[feldB] * ruleset.B_vorne_abtrieb
            new_feldB_value = felder[feldA] - (A_seitenabtrieb_1 + A_seitenabtrieb_2) + felder[feldB] - B_vorne_abtrieb # Neue Anzahl an Blättern auf Feld B

            # Nachbarfeld von Feld B, das Feld A gegenüberliegt, aktualisieren (sofern vorhanden):
            if self.does_exist((feldB[0]+blow_direction[0], feldB[1]+blow_direction[1])):
                B_seitenabtrieb_1 = 0
                B_seitenabtrieb_2 = 0
                felder[(feldB[0]+blow_direction[0], feldB[1]+blow_direction[1])] += B_vorne_abtrieb
            else:
                if ruleset.use_binomial:
                    B_seitenabtrieb_1, _ = binomial_likeliest(n=B_vorne_abtrieb, p=(1-ruleset.s1)/2, rank=ruleset.binomial_rank, handle_ties=ruleset.binomial_handle_ties) # Anzahl an Blättern von Feld B, die normalerweise auf dem Feld vor B gelandet wären, nun aber auf dem einen Feld neben Feld B landen (Seitenabtrieb 1)
                    B_seitenabtrieb_2, _ = binomial_likeliest(n=B_vorne_abtrieb-B_seitenabtrieb_1, p=((1-ruleset.s1)/2)/(1-(1-ruleset.s1)/2), rank=ruleset.binomial_rank, handle_ties=ruleset.binomial_handle_ties) # Anzahl an Blättern von Feld B, die normalerweise auf dem Feld vor B gelandet wären, nun aber auf dem anderen Feld neben Feld B landen (Seitenabtrieb 2)
                    new_feldB_value += B_vorne_abtrieb - B_seitenabtrieb_1 - B_seitenabtrieb_2
                else:
                    new_feldB_value += B_vorne_abtrieb * ruleset.s1
                    B_seitenabtrieb_1 = B_vorne_abtrieb * (1-ruleset.s1) * 0.5
                    B_seitenabtrieb_2 = B_seitenabtrieb_1

            # Nachbarfelder von Feld B, die Feld A nicht gegenüberliegen, aktualisieren (sofern vorhanden):
            if self.does_exist((feldB[0]+orthogonal_direction[0], feldB[1]+orthogonal_direction[1])):
                felder[(feldB[0]+orthogonal_direction[0], feldB[1]+orthogonal_direction[1])] += A_seitenabtrieb_1 + B_seitenabtrieb_1
            else:
                new_feldB_value += A_seitenabtrieb_1 + B_seitenabtrieb_1
            if self.does_exist((feldB[0]-orthogonal_direction[0], feldB[1]-orthogonal_direction[1])):
                felder[(feldB[0]-orthogonal_direction[0], feldB[1]-orthogonal_direction[1])] += A_seitenabtrieb_2 + B_seitenabtrieb_2
            else:
                new_feldB_value += A_seitenabtrieb_2 +B_seitenabtrieb_2

            # Feld B aktualsieren
            felder[feldB] = new_feldB_value
        else:
            # -> Es gibt kein Feld B.
            if ruleset.use_binomial:
                A_noB_seitenabtrieb_1, _ = binomial_likeliest(n=felder[feldA], p=ruleset.A_noB_seitenabtrieb, rank=ruleset.binomial_rank, handle_ties=ruleset.binomial_handle_ties)
                A_noB_seitenabtrieb_2, _ = binomial_likeliest(n=felder[feldA]-A_noB_seitenabtrieb_1, p=ruleset.A_noB_seitenabtrieb/(1-ruleset.A_noB_seitenabtrieb), rank=ruleset.binomial_rank, handle_ties=ruleset.binomial_handle_ties)
            else:
                A_noB_seitenabtrieb_1 = felder[feldA] * ruleset.A_noB_seitenabtrieb
                A_noB_seitenabtrieb_2 = A_noB_seitenabtrieb_1
            new_feldA_value = felder[feldA] - (A_noB_seitenabtrieb_1 + A_noB_seitenabtrieb_2)

            # Nachbarfelder von Feld A, die dem laubblasenden Hausmeister nicht gegenüberliegen, aktualisieren, sofern vorhanden:
            if self.does_exist((feldA[0]+orthogonal_direction[0], feldA[1]+orthogonal_direction[1])):
                felder[(feldA[0]+orthogonal_direction[0], feldA[1]+orthogonal_direction[1])] += A_noB_seitenabtrieb_1
            else:
                # -> Feld A ist ein Eckfeld, ein Teil des Laubs verbleibt also auf Feld A.
                new_feldA_value += A_noB_seitenabtrieb_1
            if self.does_exist((feldA[0]-orthogonal_direction[0], feldA[1]-orthogonal_direction[1])):
                felder[(feldA[0]-orthogonal_direction[0], feldA[1]-orthogonal_direction[1])] += A_noB_seitenabtrieb_2
            else:
                # -> Feld A ist ein Eckfeld, ein Teil des Laubs verbleibt also auf Feld A.
                new_feldA_value += A_noB_seitenabtrieb_2

        # Feld A aktualisieren:
        felder[feldA] = new_feldA_value

    def blase(self, feld0, blow_direction):
        """
        Simuliert einen Blasvorgang und aktualisiert self.felder auf die resultierende Blattverteilung.
        Args:
            feld0 (int): Index des Felds, auf dem der Hausmeister steht
            blow_direction (tuple): Richtung, in die der Hausmeister bläst. Kann folgende Werte annehmen: (1,0) =rechts, (-1,0) (=links), (0,1) (=unten), (0,-1) (=oben)
        """
        if not blow_direction in [(0,1),(0,-1),(1,0),(-1,0)]:
            return # -> Ungültige Blasrichtung. Der Laubbläser kann nur nach rechts, links, oben und unten blasen.
        self.blas_counter += 1 # Zähler, der durchgeführte Blasoperationen zählt, erhöhen
        self.blas_log.append(dict(feld0=feld0, blow_direction=blow_direction)) # Blasoperation loggen

        self.blase_blattyp(feld0, blow_direction, ruleset=self.rules_typ_1, felder=self.felder_typ_1)
        self.blase_blattyp(feld0, blow_direction, ruleset=self.rules_typ_2, felder=self.felder_typ_2)
