from copy import deepcopy
import sys

# Textdatei einlesen
# sys.argv[1] ist das erste Kommandozeilenargument
with open(sys.argv[1]) as f:
    input_lines = f.read().split("\n")

class Route:
    """
    Klasse zum Speichern einer möglichen Route
    """

    def __init__(self, orte, jahre, length, *, geschlossen):
        self.orte = orte
        self.jahre = jahre
        self.length = length
        self.geschlossen = geschlossen

    def __lt__(self, obj2):
        """
        Diese Funktion existiert, damit mit min(list<Route>) die Route mit der kürzersten Strecke ermittelt werden kann
        Returns:
            boolean: Ob die Route obj2 kürzer ist als die Route self bzw. das Objekt selbst
        """
        return self.length < obj2.length

    def add_punkt(self, ort, jahr, distance):
        """
        Fügt einen Punkt zur Route hinzu
        """
        self.orte.append(ort)
        self.jahre.append([jahr])
        self.length += distance

    def schliessen(self, jahr):
        """
        Schließt die Route
        """
        self.jahre[-1].append(jahr)
        self.geschlossen = True

    def oeffnen(self):
        """
        Erstellt eine Kopie der Route, die geöffnet ist
        """
        return Route(deepcopy(self.orte), deepcopy(self.jahre), int(self.length), geschlossen=False)

    def print(self):
        """
        Gibt die Route formatiert in der Konsole aus
        """
        for i in range(len(self.orte)):
            print(str(i+1)+")", self.orte[i], ", ".join(self.jahre[i]))

# Tourpunkte einlesen und chronologisch sortieren:
num_tourpunkte = int(input_lines.pop(0).strip())
tourpunkte = []
for i in range(num_tourpunkte):
    line = input_lines.pop(0).split(",")
    tourpunkte.append([line[0], line[1], line[2], int(line[3].strip())])
tourpunkte = sorted(tourpunkte, key = lambda k : int(k[1]))

# Kürzeste Route finden, die den Anforderungen entspricht:
before_first_essential = True # Bleibt so lange True, bis der erste essentielle Tourpunkt erreicht wird
possible_routes = [] # Alle möglichen Routen
length = 0
# Über alle Tourpunkte iterieren:
for tourpunkt in tourpunkte:

    ort, jahr = tourpunkt[0], tourpunkt[1]
    essential = tourpunkt[2] == "X"
    distance = tourpunkt[3] - length # Abstand zu vorherigen Tourpunkt berechnen
    length = tourpunkt[3]

    next_possible_routes = []
    if before_first_essential: # Falls der erste essentielle Tourpunkt noch nicht überschritten wurde, kann beim aktuellen Tourpunkt eine Route beginnen
        next_possible_routes.append(Route([ort],[[jahr]],0,geschlossen=True))
    for route in possible_routes: # Über alle möglichen Routen iterieren und versuchen, den aktuellen Tourpunkt hinzuzufügen
        if essential is False:
            next_possible_routes.append(route.oeffnen())
        if route.geschlossen:
            route.add_punkt(ort, jahr, distance)
            next_possible_routes.append(route)
        else:
            if ort == route.orte[-1]:
                route.schliessen(jahr)
                next_possible_routes.append(route)
    if essential:
        if before_first_essential:
            # -> Es handelt sich beim aktuellen Tourpunkt um den ersten essentiellen Tourpunkt
            possible_routes_before_first_essential = deepcopy(next_possible_routes) # Alle mögliche Routen vom Anfang zum ersten essentiellen Tourpunkt werden separat gespeichert, damit später verschiedene Anfangsorte durchprobiert werden können
            next_possible_routes = [Route([ort], [[]], 0, geschlossen=True)]
            before_first_essential = False # Erster essentieller Tourpunkt wurde überschritten, also Variable auf False setzen
        next_possible_routes = [min(next_possible_routes)] # Da es sich beim aktuellen Tourpunkt um einen essentiellen Tourpunkt handelt, müssen alle Routen früher oder später durch diesen Punkt gehen, es muss also nur die bisher kürzeste Route weiter betrachtet werden
    possible_routes = next_possible_routes

# Zusammensetzen aller möglichen Teilrouten von vor / nach dem ersten essentiellen Tourpunkt
possible_routes_complete = []
for route2 in possible_routes:
    for route1 in possible_routes_before_first_essential:
        route = Route(route1.orte[:-1]+route2.orte, route1.jahre[:-1]+[route1.jahre[-1]+route2.jahre[0]]+route2.jahre[1:], route1.length+route2.length, geschlossen=True)
        if route.orte[0] == route.orte[-1]:
            possible_routes_complete.append(route)
# Kürzeste zusammengesetzte Route finden
route = min(possible_routes_complete)

# Gefundene Route ausgeben
print("Die beste Route sieht so aus:")
route.print()
print("\nLänge der Route:", route.length, "LE")
