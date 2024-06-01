import sys

# Textdatei einlesen
# sys.argv[1] ist das erste Kommandozeilenargument
with open(sys.argv[1]) as f:
    input_lines = f.read().split("\n")

class Node:
    """
    Dient zum Speichern einer Position auf dem Weg
    """
    def __init__(self, position : tuple, previous_node, timestamp : int, operation : str):
        self.position = position # Tripel, das die Position angibt und wie folgt aufgebaut ist: (Stockwerk, y-Koordinate, x-Koordinate)
        self.previous_node = previous_node # Die vorherige Position, von der man zur aktuellen Position gelangt ist
        self.timestamp = timestamp # Zeit [s], die man gebraucht hat, um zur aktuellen Position zu gelangen
        self.operation = operation # Operation, die angewendet wurde, um zur aktuellen Position zu gelangen

    def __lt__(self, obj2):
        return self.timestamp < obj2.timestamp

    def successors(self):
        """
        Returns:
            list<Node>: Liste mit allen Positionen, die von der aktuellen Position aus erreicht werden könenn
        """
        floor, y, x = self.position
        successors = []

        if floors[1 if floor == 0 else 0][y][x] != "#":
            new_position = (1 if floor == 0 else 0, y, x)
            if new_position not in visited_fields:
                successors.append(
                    Node(new_position, self, self.timestamp + 3, "!")
                )

        if floors[floor][y][x-1] != "#":
            new_position = (floor, y, x-2)
            if x-2 > 0 and new_position not in visited_fields:
                successors.append(Node(new_position, self, self.timestamp+2, "<")) # Timestamp wird um 2 erhöht, da beim Ändern der Position ein Durchgangsfeld "übersprungen" wird
        if floors[floor][y][x+1] != "#":
            new_position = (floor, y, x+2)
            if x+2 < x_size and new_position not in visited_fields:
                successors.append(Node(new_position, self, self.timestamp+2, ">"))
        if floors[floor][y-1][x] != "#":
            new_position = (floor, y-2, x)
            if y - 2 > 0 and new_position not in visited_fields:
                successors.append(Node(new_position, self, self.timestamp+2, "^"))
        if floors[floor][y+1][x] != "#":
            new_position = (floor, y+2, x)
            if y + 2 < y_size and new_position not in visited_fields:
                successors.append(Node(new_position, self, self.timestamp+2, "v"))

        return successors

def print_path(node):
    """
    Rekonstruiert den zurückgelegten Weg (von der Zielposition aus) und gibt ihn aus
    Args:
        node (Node): Die Zielposition
    """
    while node.previous_node is not None:
        previous_node = node.previous_node
        floors[previous_node.position[0]][previous_node.position[1]][previous_node.position[2]] = node.operation
        if node.operation == "<":
            floors[previous_node.position[0]][previous_node.position[1]][previous_node.position[2]-1] = "<"
        if node.operation == ">":
            floors[previous_node.position[0]][previous_node.position[1]][previous_node.position[2]+1] = ">"
        if node.operation == "^":
            floors[previous_node.position[0]][previous_node.position[1]-1][previous_node.position[2]] = "^"
        if node.operation == "v":
            floors[previous_node.position[0]][previous_node.position[1]+1][previous_node.position[2]] = "v"
        node = previous_node
    for line in floors[1]:
        print("".join(line))
    print("")
    for line in floors[0]:
        print("".join(line))

# Dimensionen der Stockwerke speichern
y_size, x_size = input_lines.pop(0).split(" ")
y_size, x_size = int(y_size), int(x_size)

# Aufbau vom oberen und unteren Stockwerk in Liste floors speichern und dabei das Startfeld und das Endfeld finden
floor1 = []
floor0 = []
for i in range(y_size):
    line = input_lines.pop(0)
    if "A" in line:
        startfeld = (1, i, line.index("A"))
    if "B" in line:
        endfeld = (1, i, line.index("B"))
    floor1.append(list(line))
input_lines.pop(0)
for i in range(y_size):
    line = input_lines.pop(0)
    if "A" in line:
        startfeld = (0, i, line.index("A"))
    if "B" in line:
        endfeld = (1, i, line.index("B"))
    floor0.append(list(line))
floors = [floor0, floor1]

# Set initialisieren, in dem alle besuchten Felder gespeichert werden sollen
visited_fields = set()

to_visit = [Node(startfeld, None, 0, None)]
while len(to_visit) > 0:
    node = min(to_visit)
    to_visit.remove(node)
    if node.position in visited_fields:
        continue # Bereits besuchte Felder nicht erneut besuchen
    if node.position == endfeld:
        print_path(node)
        print(f"\nDer Weg dauert {node.timestamp} Sekunden")
        break
    visited_fields.add(node.position)
    to_visit = to_visit + node.successors()
else:
    # Wenn die Warteschlange leer ist, konnte kein Weg gefunden werden
    print("Konnte keinen Weg finden.")
