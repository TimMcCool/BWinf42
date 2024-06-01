"""
Aufgabe: 3 (ohne Erweiterung)
Ansatz: Heuristisches Vorgehen mit Greedy-Algorithmus, der ...
- Am Anfang die Ecken der längsten Kante im kumuliert größten "Winkelsystem" (Kanten, die zueinander im 60° oder 90° Winkel stehen) als mögliche Orientierungsecken identifiziert
- Kreise mit dem erforderlichen Abstand um platzierte Orte markiert und über die Schnittpunkte dieser Kreise untereinander und mit den Polygonkanten neue potentielle Orte ermittelt und
- immer den Ort platziert, der am nächsten an seinen nächsten 6 Nachbarn dran ist (-> Erzeugen einer möglichst dichten Heuristik, die möglichst einem Sechseckmuster folgt)
- Platzierung des Gesundheitszentrum erfolgt basierend auf Flächenheuristik (größtmögliche Schnittfläche Schutzgebiet mit Polygon)
"""
from geometric_util import distance, isclose, Kreis, Edge, Polygon, Point
import matplotlib.pyplot as plt
import math
import sys

# Einlesen der Daten
# sys.argv[1] ist das erste Kommandozeilenargument
with open("C:/Users/DELL/OneDrive/Projekte/BWInf 24/BWinf2 24 Tim Krome Runde 2/Aufgabe 3/siedler1.txt", "r") as f:
    input_lines = f.read().split("\n")

# Abstände und Radien festlegen
min_distance = 20 # Mindestabstand der Orte außerhalb vom Gesundheitszentrum
min_distance_ghz = 10 # Mindestabstand der Orte innerhalb vom Gesundheitszentrum
secure_distance_ghz = 85 # Radius des vom Gesundheitszentrum geschützten Gebiets

# Koordinaten der Polygon-Eckpunkte einlesen
num_corners = int(input_lines.pop(0))
coords = []
for i in range(num_corners):
    line = input_lines.pop(0).split(" ")
    coords.append((float(line[0]), float(line[1])))

class Node(Kreis):
    """
    Repräsentiert einen im Polygon platzierten Ort mit allen an den Ort gebundenen Informationen
    """
    def __init__(self, center, *, in_ghz=False, origin="", origin_edge=None):
        super().__init__(center, min_distance_ghz if in_ghz else min_distance)
        self.in_ghz = in_ghz # Gibt an, ob sich der Ort im Einzugsgebiet des Gesundheitszentrums befinden
        self.closest_neighbors = {} # Hier werden die Distanzen zu allen "Nachbar-Nodes" gespeichert
        self.origin = origin # Hier werden Informationen über den Ursprung des Orts gespeichert
        self.origin_edge = origin_edge

class Solution(Polygon):
    """
    Speichert eine (Teil-)Lösung der Aufgabe
    """

    # Methoden zur Initialisierung des Polygons:

    def __init__(self, corners):
        super().__init__(corners)

        self.fill_ghz = True # Gibt an, ob nur der Bereich im Gesundheitszentrum gefüllt werden soll (dem ist zu Beginn der Fall)
        self.reset()

    def reset(self):
        """
        Setzt alle für die Befüllung relevanten Attribute des Objekts auf ihren Anfangszustand zurück
        """
        self.nodes = [] # Hier werden die platzierten Orte gespeichert werden
        self.last_added = None # Der zuletzt hinzugefügte Ort
        self.ghz = None # Hier wird die Position des Gesundheitszentrums gespeichert werden
        self.nodes_to_add = [] # Punkte, die hinzugefügt werden können
        self.nodes_to_add_outside_ghz = [] #Punkte, die später außerhalb des Gesundheitszentrumshinzugefügt werden können

        self.area_heuristic_cache = {} # Hier werden die berechneten Flächeninhalte der Flächenheuristik gecached, damit sie nicht unnötig mehrfach berechnet werden
        self.start_edge = None
        self.circle_union_cache = None

    # Methoden zum Ausgeben des befüllten Polygons:

    def render(self, *, title="", draw_small_circles=True, draw_big_circles=False):
        """
        Zeigt mit matplotlib eine grafische Darstellung des Polygons und den Siedlungen an
        """
        plt.title(title)
        # Gitter anzeigen:
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(which='both', linestyle='--', linewidth=0.25)
        # Kanten des Polygons plotten:
        for edge in self.edges:
            plt.plot([edge.corner1[0], edge.corner2[0]], [edge.corner1[1], edge.corner2[1]], marker=None, color='r', linestyle='-', linewidth=1)
        # Aktiviere Hilfslinien:
        plt.minorticks_on()
        # Setze die Anzahl der Gitterlinien zwischen den Hauptlinien:
        plt.locator_params(axis='x', nbins=10)
        # Orte plotten:
        for node in self.nodes:
            plt.scatter(node[0], node[1], color="b", s=4.8, zorder=2)
            if draw_small_circles:
                circle = plt.Circle(node, min_distance_ghz/2 if node.in_ghz else min_distance/2, edgecolor="b", facecolor='none', linewidth=0.2)
                plt.gca().add_patch(circle)
            if draw_big_circles:
                circle = plt.Circle(node, min_distance_ghz if node.in_ghz else min_distance, edgecolor="b", facecolor='none', linewidth=0.2)
                plt.gca().add_patch(circle)
        # Gesundheitszentrum plotten:
        if self.ghz is not None:
            plt.scatter(self.ghz[0], self.ghz[1], color="g", s=4.8)
            circle = plt.Circle(self.ghz, secure_distance_ghz, edgecolor="g", facecolor='none', linewidth=0.5, zorder=3)
            plt.gca().add_patch(circle)
        # Diagramm anzeigen
        plt.show()

    def print(self):
        """
        Gibt die platzierten Orte in der Konsole aus (auf 5 Nachkommastellen gerundet)
        """
        print(len(self.nodes), "Orte insgesamt platziert")
        print(len(list(filter(lambda x : x.in_ghz, self.nodes))), "Orte im Ghz-Einzugsgebiet platziert")
        rd = 5
        print("Position des Gesundheitszentrum:", str(((round(self.ghz[0]*(10**rd))/(10**rd),(round(self.ghz[1]*(10**rd))/(10**rd))))))
        print("Positionen der Orte:", ", ".join([str(((round(ort[0]*(10**rd))/(10**rd),(round(ort[1]*(10**rd))/(10**rd))))) for ort in self.nodes]))

    # Methoden zum Umgang mit dem Gesundheitszentrum:

    def is_in_ghz_range(self, point):
        """
        Gibt zurück, ob sich der Punkt point im Einzugsgebiet des Gesundheitszentrums befindet
        """
        if self.ghz is None:
            return True
        d = distance(point, self.ghz)
        return d < secure_distance_ghz or isclose(d, secure_distance_ghz)

    def implement_ghz(self, ghz):
        """
        self.ghz wird auf die Position des Gesundheitszentrums gesetzt und die Schnittpunkte des Ghz-Einzugsgebiets mit den Polygonkanten werden ermittelt
        Args:
            ghz (tuple oder Point): Position des Gesundheitszentrums
        """
        self.ghz = ghz
        self.ghz_intersection_points = []
        for edge in self.edges:
            # Schnittpunkte des Gesundheitszentrum-Schutzradius mit den Polygonkanten ermitteln
            intersections = edge.schnittpunkt_kreis(Kreis(tuple(self.ghz), secure_distance_ghz))
            for s in intersections:
                self.ghz_intersection_points.append(tuple(s))

    def implement_ghz_intersections(self):
        """
        Die Schnittpunkte des Ghz-Einzugsgebiets mit den Polygonkanten werden zur nodes_to_add Liste hinzugefügt, wenn sie nicht durch andere bereits platzierte Orte blockiert werden
        Dieser Schritt erfolgt erst jetzt, da nach der Implementierung des Gesundheitszentrums noch Orte hinzugefügt werden, die möglicherweise die Schnittpunkte als mögliche Orte blockieren
        """
        for point in self.ghz_intersection_points:
            new_node = Node(point, in_ghz=self.is_in_ghz_range(point))
            if not self.is_blocked(new_node):
                self.nodes_to_add.append(new_node)

    def find_ghz(self):
        # Nutzt die aktuelle Ortskonfiguration zum finden einer initialen Ghz-Position: Gibt den Ort zurück, von dem aus die meisten Nachbarorte im Ghz-Sicherheitsabstand liegen
        best_node = None
        best_score = -1
        for node in self.nodes:
            # Iteriert über alle im Polygon platzierten Punkte, um den zu finden, der (wenn dort ein Ghz platziert wird) am meisten andere Punkte im secure_distance_ghz-Radius abdeckt
            score = len(list(filter(lambda x : node.closest_neighbors[tuple(x)] <= secure_distance_ghz, list(node.closest_neighbors.keys()))))
            if score > best_score:
                best_score = score
                best_node = tuple(node.center)
        self.ghz_score = best_score
        return best_node

    def optimize_ghz(self, ghz_init_pos):
        # Optimiert die zuvor mit self.find_ghz() gefundene initiale Position des Ghzs,
        # indem alle Punkte, die im Kreis mit dem Radius min_distance_ghz um das initiale Ghz liegen, mit einer Genauigkeit von 1 (Diskretisierungswert) durchprobiert werden
        # Der Punkte, bei dem ein Kreis um den Punkte mit dem Radius secure_distance_ghz die größte Schnittfläche mit dem Polygon hat, wird ausgewählt
        best_node = None
        best_score = 0
        step = 1
        radius = min_distance_ghz
        # Kreis diskretisieren
        x = ghz_init_pos[0] - radius
        for i1 in range(radius*2):
            # In jeder Iteration dieser Schleife wird der x Wert um step erhöht
            # und es wird über alle für diesen x-Wert im Kreis liegenden y-Werte iteriert
            x += step
            to_sqrt = 1-((x-ghz_init_pos[0])/radius)**2
            if to_sqrt < 0:
                y_max = 0
            else:
                y_max = math.sqrt(to_sqrt) * radius # y-Bereich an im Kreis liegenden Punkten an diesem x-Wert finden

            y = ghz_init_pos[1]
            while y < ghz_init_pos[1] + y_max:
                cut_area = self.exact_cut_area(Kreis((x,y), secure_distance_ghz))
                if cut_area > best_score:
                    if self.point_in_polygon((x,y)):
                        best_node = (x,y)
                        best_score = float(cut_area)
                y += step

            y = ghz_init_pos[1]-step
            while y > ghz_init_pos[1] - y_max:
                cut_area = self.exact_cut_area(Kreis((x,y), secure_distance_ghz))
                if cut_area > best_score:
                    if self.point_in_polygon((x,y)):
                        best_node = (x,y)
                        best_score = float(cut_area)
                y -= step

        m = (sum([n[0] for n in self.nodes])/len(self.nodes), sum([n[1] for n in self.nodes])/len(self.nodes))
        cut_area = self.exact_cut_area(Kreis(m, secure_distance_ghz))
        if cut_area > best_score:
            if self.point_in_polygon(m):
                best_node = m
                best_score = float(cut_area)
        self.ghz_score = best_score # "Score" bzw. Schnittflächenanteil speichern
        if best_node is None:
            return ghz_init_pos
        else:
            return best_node

    def align_ghz(self, ghz_init_pos):
        # Passt die Position des Ghz so an, dass Eckpunkte, die in einer 60°- oder 90°-Ecke sind, leichter in die Dreiecksstruktur der Befüllung miteinbezogen werden können
        new_ghz_positions = []
        new_ghz_position_nodes = []
        for corner_id in range(len(self.corners)):
            angle = self.get_corner_angle(corner_id)
            if angle == math.pi/2 or angle == math.pi/3:
                if not self.is_in_ghz_range(self.corners[corner_id]):
                    edge1 = self.get_edge(corner_id-1)
                    edge2 = self.get_edge(corner_id)
                    s1 = edge1.schnittpunkt_kreis(Kreis(self.ghz, secure_distance_ghz))
                    s2 = edge2.schnittpunkt_kreis(Kreis(self.ghz, secure_distance_ghz))
                    if s1 == [] or s2 == []:
                        continue
                    else:
                        i1 = min(s1, key=lambda x : distance(self.get_corner(corner_id),x))
                        i2 = min(s2, key=lambda x : distance(self.get_corner(corner_id),x))
                    d1, d2 = distance(i1, self.get_corner(corner_id)), distance(i2, self.get_corner(corner_id))
                    reference_distance = min_distance_ghz/2 if angle == math.pi/2 else min_distance_ghz
                    new_d1 = round(d1/reference_distance)*reference_distance
                    new_d2 = round(d2/reference_distance)*reference_distance
                    new_i1 = edge1.schnittpunkt_kreis(Kreis(self.get_corner(corner_id), new_d1), check_validity=True)
                    if new_i1 == []:
                        continue
                    else:
                        new_i1 = new_i1[0]
                    new_i2 = edge2.schnittpunkt_kreis(Kreis(self.get_corner(corner_id), new_d2), check_validity=True)
                    if new_i2 == []:
                        continue
                    else:
                        new_i2 = new_i2[0]
                    possible_ghz_positions = Kreis(new_i1, secure_distance_ghz).schnittpunkt_kreis(Kreis(new_i2, secure_distance_ghz))
                    for ghz_position in possible_ghz_positions:
                        if not self.is_in_ghz_range(ghz_position):
                            continue
                        new_score = self.exact_cut_area(Kreis(ghz_position, secure_distance_ghz))
                        if new_d1 == min_distance_ghz and new_d2 == min_distance_ghz:
                            new_score += Polygon([new_i1, new_i2, self.corners[corner_id]]).triangle_area()
                        if new_score >= self.ghz_score:
                            self.implement_ghz(ghz_position)
                            break
                    else:
                        continue

                    if new_d1 == min_distance_ghz and new_d2 == min_distance_ghz:
                        new_ghz_positions.append(ghz_position)
                        new_ghz_position_nodes.append(self.corners[corner_id])

        return new_ghz_positions, new_ghz_position_nodes

    # Methoden zum Hinzufügen von Orten zum Polygon:

    def add_definite_nodes(self):
        """
        Fügt Eckenpunkte hinzu, die sich an spitzen Innenwinkeln <60° befinden und in dem durch die Platzierung wegfallenden Bereich keine Schnittpunkte mit anderen Kanten als den aus dem Scheitelpunkt hervorgehenden Kanten haben.
        Dieser Schritt kann erst jetzt erfolgen, da die Platzierung des Gesundheitszentrums die Größe des bei der PLatzierung wegfallenden Bereichs beeinflusst (wegen min_distance != min_distance_ghz)
        """
        for corner_id in range(len(self.corners)):
            # Hinzufügen von gesichert zu platzierenden Orten
            if self.get_corner_angle(corner_id) <= math.pi * 1/3:
                new_node = Node(self.corners[corner_id], in_ghz=self.is_in_ghz_range(self.corners[corner_id]), origin="definite_node")
                # Überprüfen, ob im wegfallenden Bereich andere Ecken sind
                contains_corner = False
                for corner2 in self.corners:
                    if corner2 == self.corners[corner_id]:
                        continue
                    d = distance(corner2, self.corners[corner_id])
                    if d < new_node.radius:
                        contains_corner = True
                        break
                if contains_corner:
                    continue
                # Überprüfen, ob der wegfallende Bereich andere Ecken schneidet
                num_intersecting_edges = 0
                for edge in self.edges:
                    result = edge.schnittpunkt_kreis(new_node, check_validity=True)
                    for r in list(result):
                        if not distance(r,new_node.center) < new_node.radius:
                            result.remove(r)
                    if len(result) > 0:
                        num_intersecting_edges += 1
                if num_intersecting_edges > 2:
                    continue
            else:
                continue
            self.add_node(new_node)
            self.last_added = new_node
            self.successors()

    def add_node(self,new_node, *, update_nodes_to_add=True):
        """
        Fügt den Ort new_node hinzu und berechnet die Distanzen
        """
        for node in self.nodes:
            d = distance(new_node, node)
            new_node.closest_neighbors[tuple(node)] = d
            node.closest_neighbors[tuple(new_node)] = d
        self.nodes.append(new_node)
        if new_node in self.nodes_to_add:
            self.nodes_to_add.remove(new_node)
        self.last_added = new_node
        if update_nodes_to_add:
            self.update_nodes_to_add()

    def remove_node(self,old_node):
        """
        Entfernt den Ort old_node und entfernt alle Distanz-Verknüpfungen anderer Orte zum entfernten Ort
        """
        self.nodes.remove(old_node)
        for node in self.nodes:
            if tuple(old_node) in node.closest_neighbors:
                node.closest_neighbors.pop(tuple(old_node))
        old_node.closest_neighbors = {}
        self.nodes_to_add.append(old_node)


    # Methoden zum Aktualisieren der Menge an hinzufügbaren Orten:

    def successors(self):
        """
        Schnittpunkte vom letzten hinzugefügten Ortsumkreis mit den bereits existierenden Ortsumkreisen und den Begrenzungsflächenkanten finden
        """
        if self.last_added is not None:
            # Schnittpunkte mit den Ortsumkreisen der bereits existierenden Orte finden
            for node in self.nodes:
                if node.center == self.last_added.center:
                    continue
                # Verschiedene Ausschlusskriterien überprüfen, die es nicht möglich machen, dass Punkt node sich mit self.last_added schneidet (um unnötige Berechnungen zu vermeiden)
                if max(abs(node[0] - self.last_added[0]), abs(node[1] - self.last_added[1])) > self.last_added.radius + node.radius:
                    continue
                if tuple(node) in self.last_added.closest_neighbors:
                    if self.last_added.closest_neighbors[tuple(node)] > self.last_added.radius + node.radius:
                        continue
                result = self.last_added.schnittpunkt_kreis(node) # Schnittpunkte mit dem Ortsumkreis von node ermitteln
                for point in result:
                    if self.point_in_polygon(point): # Überprüfen, ob der Schnittpunkt im Polygon liegt
                        in_ghz = self.is_in_ghz_range(point)
                        if (not in_ghz and self.fill_ghz):
                            self.nodes_to_add_outside_ghz.append(Node(point, in_ghz=in_ghz))
                        else:
                            new_node = Node(point, in_ghz=in_ghz)
                            if (not self.is_blocked(new_node)) and (not new_node in self.nodes_to_add):
                                self.nodes_to_add.append(new_node)
            # Schnittpunkte mit den Kanten finden
            for edge in self.edges:
                result = edge.schnittpunkt_kreis(self.last_added, check_validity=False)
                for point in result:
                    if self.point_in_polygon(point): # Überprüfen, ob der Schnittpunkt im Polygon liegt
                        in_ghz = self.is_in_ghz_range(point)
                        new_node = Node(point, in_ghz=in_ghz, origin_edge=edge)
                        if (not in_ghz and self.fill_ghz):
                            self.nodes_to_add_outside_ghz.append(new_node)
                        elif not new_node in self.nodes_to_add:
                            if not self.is_blocked(new_node):
                                self.nodes_to_add.append(new_node)
            # Falls der Bereich des Ghz befüllt wird: Schnittpunkte mit dem Ghz-Einzugesbereich-Umfang finden
            if self.fill_ghz and self.ghz is not None:
                result = self.last_added.schnittpunkt_kreis(Kreis(self.ghz, secure_distance_ghz))
                for point in result:
                    if self.point_in_polygon(point): # Überprüfen, ob der Schnittpunkt im Polygon liegt
                        new_node = Node(point, in_ghz=self.is_in_ghz_range(point))
                        if not self.is_blocked(new_node) and (not (not new_node.in_ghz and self.fill_ghz)) and not (new_node in self.nodes_to_add):
                            self.nodes_to_add.append(new_node)

    def update_nodes_to_add(self):
        # Überprüft für alle Punkte aus nodes_to_add, ob sie nach Hinzufügen von self.last_added noch hinzugefügt werden können
        if not self.last_added is None:
            for point in list(self.nodes_to_add):
                if max(abs(point[0] - self.last_added[0]), abs(point[1] - self.last_added[1])) > point.radius + self.last_added.radius:
                    continue
                d = distance(point, self.last_added)
                use_ghz_distance = self.last_added.in_ghz or point.in_ghz # Wenn einer der beiden Orte im Gesundheitsz. ist, dann dürfen die Gesundheitsz.-Abstandsregeln verwendet werden
                if not (d > (min_distance_ghz if use_ghz_distance else min_distance) or isclose(d, min_distance_ghz if use_ghz_distance else min_distance)):
                    self.nodes_to_add.remove(point)

    def add_corners_to_nodes_to_add(self):
        # Fügt die Ecken des Polygons als potentiell hinzufügbare Orte zu self.nodes_to_add hinzu, wenn sie im gerade zu befüllenden Bereich liegen und nicht durch bereits platzierte Orte blockiert werden
        for corner in self.corners:
            new_node = Node(corner, in_ghz=self.is_in_ghz_range(corner))
            if self.fill_ghz and self.ghz is not None:
                if not new_node.in_ghz:
                    continue
            if not self.is_blocked(new_node):
                self.nodes_to_add.append(new_node)

    # Andere Util-Methoden:

    def is_blocked(self, new_point):
        """
        Gibt zurück, ob ein Ort new_point im Polygon platziert werden kann, oder ob bereits andere, zu nahe Orte existieren
        """
        for node in self.nodes:
            if max(abs(node[0] - new_point[0]), abs(node[1] - new_point[1])) > new_point.radius + node.radius:
                continue
            d = distance(node, new_point)
            use_ghz_distance = new_point.in_ghz or node.in_ghz # Wenn einer der beiden Orte im Gesundheitsz. ist, dann dürfen die Gesundheitsz.-Abstandsregeln verwendet werden
            if not (d > (min_distance_ghz if use_ghz_distance else min_distance) or isclose(d, min_distance_ghz if use_ghz_distance else min_distance)):
                return True
        return False

    # Heuristiken:

    def fill_area_heuristic(self):
        """
        Befüllt einen Bereich (entweder das ganze Polyon oder nur den Bereich um das Gesundheitszentrum herum) basierend auf der Flächenheuristik
        """
        while self.nodes_to_add != []:
            next_node = min(self.nodes_to_add, key=self.area_heuristic)
            self.add_node(next_node)
            self.add_to_circle_union(next_node)
            self.successors()

    def area_heuristic(self, x):
        """
        Gibt den Flächenanteil der durch Ort x wegfallenden Polygonfläche an der durch Ort x wegfallenden Gesamtfläche zurück
        """
        if tuple(x) in self.area_heuristic_cache and (not self.last_added is None):
            if distance(self.last_added, x) < 2*(min_distance_ghz if x.in_ghz else min_distance):
                self.area_heuristic_cache[tuple(x)] = self.exact_cut_area(x, exclude_union=True, intersect_with=Kreis(self.ghz, secure_distance_ghz) if (self.fill_ghz and self.ghz is not None) else None, radius_reduction=0.1) / (math.pi * (x.radius**2))
        else:
            self.area_heuristic_cache[tuple(x)] = self.exact_cut_area(x, exclude_union=True, intersect_with=Kreis(self.ghz, secure_distance_ghz) if (self.fill_ghz and self.ghz is not None) else None, radius_reduction=0.1) / (math.pi * (x.radius**2))
        return self.area_heuristic_cache[tuple(x)]

    def fill_distance_heuristic(self):
        """
        Befüllt einen Bereich (entweder das ganze Polyon oder nur den Bereich um das Gesundheitszentrum herum) basierend auf der Distanzheuristik
        """
        while self.nodes_to_add != []:
            best_addition = None
            best_score = float("inf")
            source = []
            if self.start_edge is not None:
                source = list(filter(lambda x : self.start_edge.is_identic_to(x.origin_edge), self.nodes_to_add))
            if source == []:
                source = list(filter(lambda x : x.origin != "definite_node", self.nodes_to_add))
            if source == []:
                source = self.nodes_to_add
            for new_node in source:
                self.add_node(new_node, update_nodes_to_add=False)
                score = self.distance_heuristic()
                if score < best_score:
                    best_score = score
                    best_addition = new_node
                self.remove_node(new_node)
            if best_addition is None:
                return
            self.add_node(best_addition)
            self.successors()

    def distance_heuristic(self):
        """
        Bewertet nicht eine Position x, sondern den Gesamtzustand, in dem sich das Polygon befindet:
        Gibt den durchschnittlichen kumulierten Abstand eines Ortes zu seinen 6 nächsten Nachbarorten zurück
        """
        score = 0
        for node in self.nodes:
            score += sum(sorted(list(node.closest_neighbors.values()))[:6])/6
        return score / len(self.nodes)

    # Speichern / Laden von Zwischenzuständen:
    def to_waypoint(self):
        # Speichert einen Zwischenzustand. Wird von diversen Befüllungsstrategien verwendet, die verschiedene Optionen ausprobieren
        waypoint = {}
        waypoint["nodes"] = list(self.nodes)
        waypoint["nodes_to_add"] = list(self.nodes_to_add)
        waypoint["nodes_to_add_outside_ghz"] = list(self.nodes_to_add_outside_ghz)
        waypoint["area_heuristic_cache"] = dict(self.area_heuristic_cache)
        waypoint["last_added"] = [self.last_added]
        waypoint["ghz"] = tuple(self.ghz)
        waypoint["ghz_intersection_points"] = list(self.ghz_intersection_points)
        return waypoint

    def load_waypoint(self, waypoint):
        # Lädt den Zwischenzustand waypoint
        self.nodes = list(waypoint["nodes"])
        self.nodes_to_add = list(waypoint["nodes_to_add"])
        self.nodes_to_add_outside_ghz = list(waypoint["nodes_to_add_outside_ghz"])
        self.area_heuristic_cache = dict(waypoint["area_heuristic_cache"])
        self.last_added = waypoint["last_added"][0]
        self.ghz = tuple(waypoint["ghz"])
        self.ghz_intersection_points = list(waypoint["ghz_intersection_points"])

    # Analyse:

    def get_start_point(self, *, new_ghz_positions=[], extra_start_nodes=[]):

        def adjust_section_to_ghz(self, edge):
            if self.ghz is None:
                return edge
            s = edge.schnittpunkt_kreis(Kreis(self.ghz, secure_distance_ghz))
            if len(s) == 2:
                if edge.corner1 in extra_start_nodes:
                    return Edge(edge.corner1, s[1])
                if edge.corner2 in extra_start_nodes:
                    return Edge(s[0], edge.corner2)
                return Edge(s[0], s[1])
            if len(s) == 1:
                if (self.is_in_ghz_range(edge.corner1) and self.fill_ghz) or ((not self.is_in_ghz_range(edge.corner1)) and (not self.fill_ghz)):
                    if edge.corner2 in extra_start_nodes:
                        return Edge(edge.corner1, edge.corner2)
                    else:
                        return Edge(edge.corner1, s[0])
                else:
                    if edge.corner1 in extra_start_nodes:
                        return Edge(edge.corner1, edge.corner2)
                    else:
                        return Edge(s[0], edge.corner2)
            else:
                if (self.is_in_ghz_range(edge.corner1) and self.fill_ghz) or ((not self.is_in_ghz_range(edge.corner1)) and (not self.fill_ghz)):
                    return edge
                else:
                    return None

        class Winkelsystem:

            def __init__(self):
                self.cut_edges = []
                self.lines = []
                self.all_cut_edges = []

            def add_cut_edge(self, cut_edge):
                for i in range(len(self.lines)):
                    if self.lines[i].is_identic_to(cut_edge):
                        self.cut_edges[i].append(cut_edge)
                        self.all_cut_edges.append(cut_edge)
                        break
                else:
                    self.cut_edges.append([cut_edge])
                    self.lines.append(cut_edge)
                    self.all_cut_edges.append(cut_edge)

        cut_edges = [adjust_section_to_ghz(self, edge) for edge in self.edges]
        cut_edges = list(filter(lambda x : x is not None, cut_edges))

        winkelsysteme_trigonal = {}
        winkelsysteme_orthogonal = {}

        for cut_edge in cut_edges:
            alpha = cut_edge.steigungswinkel()
            angle = round((alpha % (math.pi/3)) * 100)/100
            if angle in winkelsysteme_trigonal:
                winkelsysteme_trigonal[angle].add_cut_edge(cut_edge)
            else:
                w = Winkelsystem()
                w.add_cut_edge(cut_edge)
                winkelsysteme_trigonal[angle] = w
            angle = round((alpha % (math.pi/2)) * 100)/100
            if angle in winkelsysteme_orthogonal:
                winkelsysteme_orthogonal[angle].add_cut_edge(cut_edge)
            else:
                w = Winkelsystem()
                w.add_cut_edge(cut_edge)
                winkelsysteme_orthogonal[angle] = w
        for cut_edge in cut_edges:
            alpha = cut_edge.steigungswinkel()
            angle = round((alpha % (math.pi/3)) * 100)/100
            if angle in winkelsysteme_trigonal:
                if not cut_edge in winkelsysteme_trigonal[angle].all_cut_edges:
                    winkelsysteme_trigonal[angle].append(cut_edge)
            if angle in winkelsysteme_orthogonal:
                if not cut_edge in winkelsysteme_trigonal[angle].all_cut_edges:
                    winkelsysteme_trigonal[angle].append(cut_edge)

        def score_line_edges(w, line_edges):
            cumulative_length = 0
            for edge in line_edges:
                cumulative_length += edge.length()
            return cumulative_length

        def score_winkelsystem(w):
            cumulative_length = 0
            for edge in w.all_cut_edges:
                cumulative_length += edge.length()
            return cumulative_length

        max_t = max(list(winkelsysteme_trigonal.values()), key=score_winkelsystem)
        max_o = max(list(winkelsysteme_orthogonal.values()), key=score_winkelsystem)
        if score_winkelsystem(max_t) >= score_winkelsystem(max_o):
            source_edges, source_sections = max_t.lines, max_t.cut_edges
        else:
            source_edges, source_sections = max_o.lines, max_o.cut_edges
        best_source_edges = sorted(source_edges, key=lambda x : score_line_edges(w, source_sections[source_edges.index(x)]), reverse=True)
        for start_edge in best_source_edges:
            self.start_edge = start_edge
            sections = source_sections[source_edges.index(start_edge)]
            best_sections = sorted(sections, key=lambda x : x.length(), reverse=True)
            for section in best_sections:
                option1 = section.corner1
                option2 = section.corner2
                if option1 in self.corners:
                    angle1 = self.get_corner_angle(self.corners.index(option1))
                    if angle1 < math.pi/3:
                        option1 = None
                else:
                    angle1 = 0
                if option2 in self.corners:
                    angle2 = self.get_corner_angle(self.corners.index(option2))
                    if angle2 < math.pi/3:
                        option2 = None
                else:
                    angle2 = 0
                if option1 is None and option2 is None:
                    continue
                if angle1 > angle2 and option1 is not None:
                    return Node(option1, in_ghz=self.is_in_ghz_range(option1))
                elif isclose(angle1, angle2):
                    if option1 is None:
                        return Node(option1, in_ghz=self.is_in_ghz_range(option1))
                    else:
                        return Node(option2, in_ghz=self.is_in_ghz_range(option2))
                elif option2 is not None:
                    return Node(option2, in_ghz=self.is_in_ghz_range(option2))

# Neues Solution Objekt initialisieren
p = Solution(coords)

# Gesundheitszentrum/Ghz-Position finden und optimieren
p.add_corners_to_nodes_to_add()
p.fill_area_heuristic() # Eine initiale Punktplatzierung erzeugen, basierend auf der dann das Gesundheitszentrum platziert werden kann (wird beim Zurücksetzen mit reset() wieder gelöscht)
ghz = p.find_ghz()
ghz = p.optimize_ghz(ghz)
p.reset() # Zurücksetzen auf Anfangszustand
p.implement_ghz(ghz)
new_ghz_positions, extra_start_nodes = p.align_ghz(ghz)

# Orte hinzufügen, die auf jeden Fall dort zu platzieren sind
p.add_definite_nodes()

# Ghz-Einzugsgebiet befüllen
p.area_heuristic_cache = {}
# Überprüfen, ob der Ghz-Einzugsbereich Kanten schneidet oder nicht:
ghz_area_uncut = len(set(p.ghz_intersection_points)) == len(p.ghz_intersection_points) / 2
if ghz_area_uncut:
    for corner in p.corners:
        if corner in p.ghz_intersection_points:
            continue
        if p.is_in_ghz_range(corner):
            ghz_area_uncut = False
            break
if ghz_area_uncut:
    # Wenn der Ghz-Einzugsbereich keine Kanten schneidet, dann wird direkt der ganze Bereich befüllt, hierbei werden je nach Bereich die entsprechenden Abstandsregeln eingehalten
    pass # -> kein Befüllen des reinen Ghz-Bereichs
else:
    start_point = p.get_start_point(new_ghz_positions=new_ghz_positions, extra_start_nodes=extra_start_nodes)
    move_ghz = False
    if start_point is not None:
        if tuple(start_point) in extra_start_nodes:
            move_ghz = True
            ghz = new_ghz_positions[extra_start_nodes.index(tuple(start_point))]
            p.implement_ghz(ghz)
    # Schnittpunkte des Ghz-Einzugsbereich mit Kanten zu den potentiell hinzufügbaren Orten hinzufügen
    p.implement_ghz_intersections()
    if start_point is not None:
        if not p.is_blocked(start_point): # Möglicherweise wurde an der Position start_point bereits von add_definite_nodes ein Ort platziert (dieser Fall kann auftreten, wenn der Eck-Innenwinkel an start_node 60° beträgt)
            p.add_node(start_point)
            p.successors()
    p.add_corners_to_nodes_to_add()
    if move_ghz:
        before_filling_process = p.to_waypoint()
        results = []
        possible_start_edges = list(filter(lambda edge : edge.corner1 == tuple(start_point) or edge.corner2 == tuple(start_point), p.edges))
        for start_edge in possible_start_edges:
            p.start_edge = start_edge
            p.fill_distance_heuristic()
            results.append(p.to_waypoint())
            p.load_waypoint(before_filling_process)
        p.load_waypoint(max(results, key=lambda x : len(x["nodes"])))
    else:
        p.fill_distance_heuristic()

# Restliche Bereiche befüllen
p.fill_ghz = False
p.nodes_to_add = p.nodes_to_add_outside_ghz
p.add_corners_to_nodes_to_add()
for node in list(p.nodes_to_add):
    if p.is_blocked(node):
        p.nodes_to_add.remove(node)
p.fill_distance_heuristic()
p.print()
p.render(title="[H2] Finaler Besiedlungsplan")
