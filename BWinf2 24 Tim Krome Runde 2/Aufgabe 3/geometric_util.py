# Diese Datei enthält geometrische Hilfsfunktionen und Klassen

import math
from copy import deepcopy
import shapely.geometry as shapely_geometry # Library shapely wird ausschließlich zur Berechnung von Schnittflächen verwendet

def isclose(num1, num2, *, digits=9):
    """
    Gibt an, ob die beiden Werte nahe aneinanderliegen (d.h. Unterschied kleiner als 10^(-digits))
    """
    return abs(num2-num1) < (10**(-digits))

def distance(point1 : tuple, point2):
    """
    Gibt die Distanz zwischen zwei Punkten zurück (2D)
    Args:
    - point1 (tuple), point2 (tuple): Die beiden Punkte, müssen als Tupel der Länge zwei (oder als Point Objekte vorliegen)
    """
    return math.sqrt((point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2)

def crossproduct2d(vector1 : tuple, vector2 : tuple):
    """
    Gibt das 2D-Kreuzprodukt der beiden 2D-Vektoren zurück. Die Vektoren müssen als Tupel der Länge 2 vorliegen.
    """
    return vector1[0] * vector2[1] - vector2[0] * vector1[1]

class Point:
    """
    Repräsentiert einen Punkt, bestehend aus x- und y-Koordinate
    Die Klasse wurde als Ersatz für tuple() erstellt, damit eine angepasste Gleichheitsfunktion (__eq__) eingerichtet werden kann, die zum Vergleichen Pythobs isclose Funktion verwendet (Umgang mit Pythons Fließkommazahl-Genauigkeit)
    """
    def __init__(self, x,y):
        self.center = (x,y)

    def __str__(self):
        return str(self.center)

    def __eq__(self,other):
        if isinstance(other, tuple):
            other = Point(*other)
        elif isinstance(other, Point):
            return isclose(self.center[0], other.center[0]) and isclose(self.center[1], other.center[1])
        else:
            return False

    def __getitem__(self,i):
        return self.center[i]

    def __tuple__(self):
        return self.center

class Kreis(Point):
    """
    Repräsentiert einen Kreis
    Implement geometrisch-analytische Methoden zum Umgang mit Kreisen (Berechnung der Schnittpunkte zweier Kreise)
    """
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def schnittpunkt_kreis(self, other) -> list:
        """
        Gibt in einer Liste die Schnittpunkte zurück, die dieser Kreis mit einem anderen Kreis hat
        Args:
        - other (Kreis): der andere Kreis
        """
        other_center = other.center
        if isclose(other.radius, self.radius):
            # Kreise haben den gleichen Radius -> Punkt s ist Mitte der Verbindungsstrecke (self.center to other center), b ist der Abstand von Punkt self.center zu Punkt s
            s = ((self.center[0] + other_center[0]) / 2, (self.center[1] + other_center[1])/2)
            b = distance(s, self.center)
        else:
            d = distance(self.center, other.center)
            if isclose(d, 0):
                return [] # Identischer Kreismittelpunkt -> keine oder unendlich viele Schnittpnkte
            to_acos = (other.radius**2 - self.radius**2 - d**2)/(-2*self.radius*d)
            if not -1 <= to_acos <= 1:
                return [] # -> keine Schnittpunkte
            beta = math.acos(to_acos)
            b = math.cos(beta) * self.radius
            s = (self.center[0] + (other_center[0]-self.center[0])*(b/d), self.center[1] + (other_center[1]-self.center[1])*(b/d))
        if self.radius ** 2 - b ** 2 < 0:
            return [] # -> keine Schnittpunkte
        a = math.sqrt(self.radius ** 2 - b ** 2)

        if isclose(other_center[0], self.center[0]):
            orthogonal = 0
        else:
            steigung = (other_center[1]-self.center[1])/(other_center[0]-self.center[0])
            if isclose(other_center[1],self.center[1] ):
                return [Point(s[0], s[1]+a), Point(s[0], s[1]-a)]
            else:
                orthogonal = -1/steigung

        y_abschnitt =  -1 * s[0] * orthogonal + s[1]
        factor = math.sqrt(1 + orthogonal**2)
        solutions = []
        solutions.append(Point(s[0] + a/factor, y_abschnitt + (s[0] + a/factor) * orthogonal))
        solutions.append(Point(s[0] - a/factor, y_abschnitt + (s[0] - a/factor) * orthogonal))
        return solutions

class Edge:
    """
    Repräsentiert eine Kante eines Polygons
    Implement geometrisch-analytische Funktionen zum Umgang mit Geraden und Kanten sowie zur Interaktion mit anderen geometrischen Objekten (Kreisen und Punkten)
    """
    def __init__(self, corner1:Point, corner2:Point=None, *, m=None, c=None):
        """
        Kante definiert sich über Eckpunkte corner1 und corner2.
        Alternativ kann diese Klasse auch zum Repräsentieren einer Gerade verwendet werden, in diesem Fall wird die Steigung durch m und der y-Achsenabschnitt durch c angegebenen (y = mx + c) und es muss trotzdem mind. corner1 angegeben sein (damit im Fall m == unendlich trotzdem die Lage der Gerade rekonstruiert werden kann)
        """
        if corner2 is None:
            self.m = m
            self.c = c
            self.corner1 = corner1
            if self.m is None:
                self.corner2 = (corner1[0],corner1[1]+1)
            else:
                self.corner2 = (corner1[0]+1, self.m*(corner1[0]+1)+self.c)
        else:
            if corner1[0] > corner2[0]:
                # corner1 soll immer der Punkt mit der niedrigeren x-Koordinate sein, dies erleichtert spätere Berechnungen
                helper = tuple(corner1)
                corner1 = tuple(corner2)
                corner2 = helper
            self.corner1 = corner1
            self.corner2 = corner2
            # Steigung und y-Abschnitt der Gerade (mx + c) bestimmen, auf der die Kante liegt:
            if corner1[0] == corner2[0]:
                # y-Achsenabschnitt und Steigung nicht bestimmbar, da Kante parallel zur y-Achse
                self.m = None
                self.c = None
            else:
                self.m = ( self.corner2[1] - self.corner1[1] ) / (self.corner2[0] - self.corner1[0]) # Steigung der Gerade, auf der die Kante liegt (mx + c)
                self.c = -1 * self.corner1[0] * self.m + self.corner1[1] # y-Achsenabschnitt der Gerade, auf der die Kante liegt (mx + c)
        self.cached_length = None # Hier wird die Länge der Kante gecached, sobald sie einmal angefordert wurde (zum Vermeiden mehrfacher Berechnungen)
        self.steigungswinkel_cache = None # Hier wird der Steigungswikel der Kante gecached, sobald er einmal angefordert wurde (zum Vermeiden mehrfacher Berechnungen)

    def distance_to_edge(self, other):
        """
        Berechnet den Abstand dieser Kante zu einer anderen Kante.
        """
        if self.m is None or other.m is None:
            return abs(self.corner1[0] - other.corner1[0])
        if not isclose(self.m, other.m):
            return 0
        elif isclose(self.m, 0):
            return abs(self.corner1[1] - other.corner1[1])
        else:
            other_m = -1/self.m
            s1 = self.schnittpunkt_gerade(other_m, 0, check_validity=False)
            s2 = other.schnittpunkt_gerade(other_m, 0, check_validity=False)
            return distance(s1, s2)

    def schnittpunkt_gerade(self, other_m, other_c, *, other_corner1=None, check_validity=True) -> Point:
        """
        Bestimmt den Schnittpunkt der Gerade, auf der die Kante liegt, mit einer anderen Gerade.
        Args:
            other_m (float): Steigung der anderen Gerade
            other_c (float): y-Achsenabschnitt der anderen Gerade
            check_validity (boolean): Wenn True, dann wird überprüft, ob die ermittelten Schnittpunkte tatsächluch auf der Polygonkante liegt (wenn nicht, dann wird None zurückgegeben)
        """
        if self.m is None:
            # Diese Gerade verläuft parallel zur y-Achse
            if other_m is None:
                # Wenn die andere Gerade auch parallel zur y-Achse läuft, dann gibt es keine Schnittpunkte
                return None
            x = self.corner1[0]
            y = other_m * x + other_c
        elif other_m is None:
            if other_corner1 is None:
                # Schnittpunkt nicht bestimmbar: Es fehlen Infos zur anderen Gerade
                return None
            x = other_corner1[0]
            y = self.m * x + self.c
        elif isclose(self.m, other_m):
            # Geraden entweder parallel (keine Schnittpunkte) oder identisch (unendlich viele Schnittpunkte) -> Kein konkreter Schnittpunkt wird zurückgegeben
            return None
        else:
            if isclose(other_c, self.c):
                x = 0
            else:
                x = (self.c-other_c)/(other_m-self.m)
            y = self.m * x + self.c
        if check_validity:
            # Überprüfen, ob der ermittelte Schnittpunkt tatsächlich auf der Polygonkante liegt
            if self.corner1[0] <= x <= self.corner2[0]:
                if min((self.corner1[1], self.corner2[1])) <= y <= max((self.corner1[1], self.corner2[1])):
                    return Point(x, y)
        else:
            return Point(x, y)
        return None

    def schnittpunkt_kreis(self, kreis, *, check_validity=True) -> list:
        """
        Bestimmt die Schnittpunkte der Polygonkante mit einem Kreis.
        Die Schnittpunkte werden in einer Liste zurückgegeben.
        Args:
            kreis (Kreis): Der andere Kreis
        """
        solutions = []

        if self.m is None:
            s = (self.corner1[0], kreis.center[1])
        elif math.isclose(self.m, 0):
            s = (kreis.center[0], self.corner1[1])
        else:
            other_m = -(1/self.m)
            other_c = -1 * kreis.center[0] * other_m + kreis.center[1]
            s = self.schnittpunkt_gerade(other_m, other_c, check_validity=False)

        a = distance(kreis.center, s)
        if kreis.radius**2 - a**2 < 0:
            return []
        b = float(math.sqrt(kreis.radius**2 - a**2))
        if self.m is None:
            solutions.append(Point(self.corner1[0], s[1] + b))
            solutions.append(Point(self.corner1[0], s[1] - b))
        else:
            factor = math.sqrt(1 + self.m**2)
            solutions.append(Point(s[0] + b/factor, (s[0] + b/factor) * self.m + self.c))
            solutions.append(Point(s[0] - b/factor, (s[0] - b/factor) * self.m + self.c))

        if check_validity:
            for solution in list(solutions):
                if self.corner1[0] <= solution[0] <= self.corner2[0]:
                    if min((self.corner1[1], self.corner2[1])) <= solution[1] <= max((self.corner1[1], self.corner2[1])):
                        continue
                solutions.remove(solution)

        return solutions

    def length(self):
        if self.cached_length is None:
            self.cached_length = distance(self.corner2, self.corner1)
        return self.cached_length

    def is_identic_to(self, edge2):
        if edge2 is None:
            return False
        if self.m is None and edge2.m is None:
            return isclose(self.corner1[0], edge2.corner1[0])
        if self.m is None and edge2.m is not None or self.m is not None and edge2.m is None:
            return False
        if not isclose(self.m, edge2.m):
            return False
        return isclose(self.corner1[0] * edge2.m + edge2.c, self.corner1[1])

    def steigungswinkel(self):
        if self.steigungswinkel_cache is None:
            v1, v2 = (self.corner2[0]-self.corner1[0], self.corner2[1]-self.corner1[1]), (1, 0)
            theta = math.acos((v1[0] * v2[0] + v1[1] * v2[1])/(math.sqrt(v1[0]**2+v1[1]**2)*math.sqrt(v2[0]**2+v2[1]**2)))
            if crossproduct2d(v1, v2) > 0:
                self.steigungswinkel_cache = theta # da Ecke konvex
            else:
                self.steigungswinkel_cache = 2*math.pi - theta # da Ecke konkav
        return self.steigungswinkel_cache

    def is_on_edge(self, point):
        if self.corner1[0] <= point[0] <= self.corner2[0]:
            if min((self.corner1[1], self.corner2[1])) <= point[1] <= max((self.corner1[1], self.corner2[1])):
                if self.m is None:
                    return isclose(point[0], self.corner1[0])
                else:
                    return isclose(self.m * point[0] + self.c, point[1])
        return False

    def orthogonal_line_through_point(self, point):
        # Steigung der orthogonalen Geraden berechnen
        if self.m == 0:
            m_ortho = None  # Vertikale Linie, Steigung unendlich
        elif self.m is None:
            m_ortho = 0
        else:
            m_ortho = -1 / self.m

        # y-Achsenabschnitt der orthogonalen Geraden berechnen
        if m_ortho is None:
            c_ortho = point[1]
        else:
            c_ortho = point[1] - m_ortho * point[0]

        # Gleichung der orthogonalen Geraden zurückgeben (mx + c Form)
        return Edge(point, m=m_ortho, c=c_ortho)

class Polygon:
    """
    Repräsentiert ein Polygon
    """
    def __init__(self, corners):
        if len(corners) > 3:
            # Sicherstellen, dass die Punkte des Polygons im Uhrzeigersinn vorliegen (wenn das Polygon ein Dreieck ist, dann ist es egal - die Punkte müssen nur für die Triangulation von größeren Polygonen clockwise vorliegen):
            if sum([crossproduct2d(corners[i], corners[(i+1)%len(corners)]) for i in range(0,len(corners))]) > 0:
                corners.reverse()

        if len(corners) < 3:
            print("Polygon muss mindestens 3 Ecken haben.")
            exit()
        self.corners = corners
        self.edges = []
        for i in range(len(corners)):
            self.edges.append(Edge(self.get_corner(i), self.get_corner(i+1)))
        self.shapely_polygon = shapely_geometry.Polygon(self.corners)
        self.triangulation = None
        self.area = None
        self.circle_union_cache = None

    def get_corner(self, corner_index):
        """
        Gibt die Ecke am Index corner_index zurück. Die Indizierung erfolgt dabei zyklisch.
        """
        corner_index = corner_index % len(self.corners)
        return self.corners[corner_index]

    def get_edge(self, edge_index):
        """
        Gibt die Ecke am Index corner_index zurück. Die Indizierung erfolgt dabei zyklisch.
        """
        edge_index = edge_index % len(self.edges)
        return self.edges[edge_index]

    def get_corner_angle(self, corner_id):
        """
        Gibt den Innenwinkel des Polygons an der Ecke corner zurück.
        """
        c0, c1, c2 = self.get_corner(corner_id-1), self.get_corner(corner_id), self.get_corner(corner_id+1) # Ecke corner (c1) und die Nachbarecken von corner (c0, c2)
        v1, v2 = (c1[0]-c0[0], c1[1]-c0[1]), (c1[0]-c2[0], c1[1]-c2[1])
        theta = math.acos((v1[0] * v2[0] + v1[1] * v2[1])/(math.sqrt(v1[0]**2+v1[1]**2)*math.sqrt(v2[0]**2+v2[1]**2))) # Berechnung des Winkels (im Bogenmaß) über die Formel zur Bestimmung des Innenwinkels zwischen zwei Vektoren
        if crossproduct2d(v1, v2) > 0: # Über Kreuzprodukt der 2D-Vektoren entscheiden, ob Ecke konvex oder konkav
            return theta # da Ecke konvex
        else:
            return 2*math.pi - theta # da Ecke konkav muss für die Berechnung des Innenwinkels im Polygon der Innenwinkel zwischen den Vektoren v1 und v2 umgerechnet werden

    def triangulate(self):
        """
        Gibt eine Triangulation des Polygons zurück
        Returns:
            list<Polygon>: Eine Liste, die n-2 Dreiecke enthält (mit n := Anzahl Ecken vom zu trianguilierenden Polygon)
        """
        if self.triangulation is None:
            polygon_to_divide, self.triangulation = deepcopy(self), []
            while len(polygon_to_divide.corners) > 3:
                for corner_id in range(len(polygon_to_divide.corners)):
                    c0, c1, c2 = polygon_to_divide.get_corner(corner_id-1), polygon_to_divide.get_corner(corner_id), polygon_to_divide.get_corner(corner_id+1)
                    v1, v2 = (c1[0]-c0[0], c1[1]-c0[1]), (c1[0]-c2[0], c1[1]-c2[1])
                    if crossproduct2d(v1, v2) > 0:
                        # -> Ecke konvex
                        triangle = Polygon([c0,c1,c2])
                        for c_check in polygon_to_divide.corners:
                            if c_check in [c0,c1,c2]:
                                continue
                            if triangle.point_in_polygon(c_check):
                                triangle = None
                                break
                        if triangle is not None:
                            self.triangulation.append(triangle)
                            corners = polygon_to_divide.corners
                            corners.remove(c1)
                            polygon_to_divide = Polygon(corners)
                            break
                else:
                    self.triangulation = [] # Triangulation fehlgeschlagen -> abbrechen, um Endlosschleife zu vermeiden
                    return []
            self.triangulation.append(polygon_to_divide)
        return self.triangulation

    def triangle_area(self):
        """
        Returns:
            float: Die Fläche des Polygons, wenn es dreieckig ist
        """
        if len(self.corners) == 3:
            s = 0.5*sum([edge.length() for edge in self.edges])
            to_sqrt = s*(s-self.edges[0].length())*(s-self.edges[1].length())*(s-self.edges[2].length())
            if to_sqrt < 0:
                return 0
            return math.sqrt(to_sqrt)

    def calc_area(self):
        if self.triangulation is None:
            self.triangulation = self.triangulate()
        return sum([x.triangle_area() for x in self.triangulation])

    def point_in_polygon(self, point : Point):
        if len(self.corners) == 3:
            if min([self.corners[0][0], self.corners[1][0], self.corners[2][0]])-1 <= point[0] <= max([self.corners[0][0], self.corners[1][0], self.corners[2][0]])+1:
                if min([self.corners[0][1], self.corners[1][1], self.corners[2][1]])-1 <= point[1] <= max([self.corners[0][1], self.corners[1][1], self.corners[2][1]])+1:
                      big_area = self.triangle_area()
                      small_triangles = [Polygon([point, self.get_corner(i), self.get_corner(i+1)]) for i in range(3)]
                      small_area = sum([triangle.triangle_area() for triangle in small_triangles])
                      if isclose(big_area, small_area):
                          return True
            return False
        else:
            triangulation = self.triangulate()
            for triangle in triangulation:
                if triangle.point_in_polygon(point):
                    return True
            return False

    def exact_cut_area(self, circle : Kreis, *, exclude_union=False, intersect_with = None, radius_reduction=0) -> float:
        # Define polygon and circle
        shapely_circle = shapely_geometry.Point(tuple(circle.center)).buffer(circle.radius-radius_reduction)

        # Get the intersection
        initial_intersection = self.shapely_polygon.intersection(shapely_circle)
        if initial_intersection.is_empty:
            return 0
        if intersect_with is not None:
            initial_intersection = initial_intersection.intersection(shapely_geometry.Point(tuple(intersect_with.center)).buffer(intersect_with.radius))
            if initial_intersection.is_empty:
                return 0

        circle_union = self.circle_union_cache

        if circle_union is None or exclude_union is False:
            return initial_intersection.area
        intersection = initial_intersection.intersection(circle_union)
        if intersection.is_empty:
            return initial_intersection.area
        return initial_intersection.area - intersection.area

    def add_to_circle_union(self, object):
        if self.circle_union_cache is None:
            self.circle_union_cache = shapely_geometry.Point(tuple(object.center)).buffer(object.radius)
        else:
            self.circle_union_cache = self.circle_union_cache.union(shapely_geometry.Point(tuple(object.center)).buffer(object.radius))
