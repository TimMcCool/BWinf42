import random

# Parameter festlegen
n = 8 # Größe des Arukone-Gitters
seed = None # Startwert für die Arukone-Generierung. Bei gleichbleibendem Seed wird das Programm immer dasselbe Arukone ausgeben. Soll ein zufälliges Arukone generiert werden, dann muss seed auf None gesetzt werden

def num_free_neighbors(pos):
    """
    Returns:
        int: Die Anzahl an freien Nachbarfeldern, die das Feld am Index pos hat
    """
    num_free_neighbors = 0
    if pos - n >= 0:
        if grid[pos-n] == 0:
            num_free_neighbors += 1
    if pos + n < n*n:
        if grid[pos+n] == 0:
            num_free_neighbors += 1
    if pos % n != 0:
        if grid[pos-1] == 0:
            num_free_neighbors += 1
    if pos % n != n-1:
        if grid[pos+1] == 0:
            num_free_neighbors += 1
    return num_free_neighbors

def get_start_field():
    """
    Returns:
        int: Der Index von dem Feld, das am wenigsten freie Nachbarfelder hat, aber mind. 1 freies Nachbarfeld hat. Wenn es kein solches Feld gibt, wird None zurückgegeben
    """
    feld = None # Bisher "bestes" gefundenes Feld
    best = 5 # Anzahl an freien Nachbarfeldern, die das "beste" gefundene Feld hat
    for _feld in range(n*n):
        if grid[_feld] == 0: # Überprüfen, ob das Feld selbst frei ist
            score = num_free_neighbors(_feld)
            if score != 0: # Sicherstellen, dass das Feld mind. 1 freien Nachbarn hat
                if score < best:
                    best = score
                    feld = _feld
    return feld

def is_allowed_step(old_pos, new_pos, paar):
    """
    Überprüft, ob es möglich ist, eine Linie vom Feld am Index old_pos zum Nachbarfeld am Index new_pos fortsetzen
    """
    if 0 > new_pos or new_pos >= n*n: # Sicherstellen, dass new_pos im Gitter liegt
        return False
    if old_pos % n == 0 and new_pos % n == n-1: # Sicherstellen, dass der Schritt nicht den linken Gitterand überschreitet
        return False
    if old_pos % n == n-1 and new_pos % n == 0: # Sicherstellen, dass der Schritt nicht den rechten Gitterand überschreitet
        return False
    if grid[new_pos] != 0: # Sicherstellen, dass das Feld new_pos frei ist
        return False
    # Sicherstellen, dass keines der Nachbarfelder von new_pos bereits von derselben Linie durchzogen wird
    if new_pos - n >= 0 and new_pos-n != old_pos:
        if grid[new_pos-n] == paar or grid[new_pos-n] == f".{paar}":
            return False
    if new_pos + n < n*n and new_pos+n != old_pos:
        if grid[new_pos+n] == paar or grid[new_pos+n] == f".{paar}":
            return False
    if new_pos % n != 0 and new_pos-1 != old_pos:
        if grid[new_pos-1] == paar or grid[new_pos-1] == f".{paar}":
            return False
    if new_pos % n != n-1 and new_pos+1 != old_pos:
        if grid[new_pos+1] == paar or grid[new_pos+1] == f".{paar}":
            return False
    return True

# Input-Seed verarbeiten
if seed is None:
    input_seed = []
else:
    input_seed = seed.split(";") # Verarbeiteter Input-Seed
output_seed = [] # In dieser Liste werden Daten für den Output-Seed, der ausgegeben wird, gespeichert

grid = [0 for i in range(n*n)] # Initialisieren eines leeren Arukone-Gitter
num_paare = 0 # Initialisieren einer Zählervariable, die die Anzahl der platzierten Paare zählen wird

# Gitter solange füllen, bis kein Platz mehr für neue Paare ist
while True:
    # In jeder Iteration wird ein Zahlenpaar zum Arukone hinzugefügt. Die Schleife wird abgebrochen, sobald mindestens n/2 Paare platziert wurden und es im Gitter keinen Platz für weitere Paare mehr gibt
    if num_paare == 0:
        # Das Anfangsfeld des ersten Zahlenpaars wird anders gewählt als bei späterenm Zahlenpaaren, damit die erstellten Arukone diverser sind
        if input_seed == []:
            # Wenn noch keine Paare platziert und der Input-Seed das Feld nicht vorschreibt, wird für das Startfeld des 1. Zahlenpaars ein zufälliges Feld aus der oberen Hälfte des Arukone gewählt
            start_field = random.randint(0,n*round(n/2))
        else:
            start_field = int(input_seed.pop(0))
            if start_field >= n*n:
                print(f"Ungülter Seed für ein Arukone mit der Gittergröße {n} * {n}")
                exit()
        output_seed.append(str(start_field))
    else:
        start_field = get_start_field() # Ein geeignetes Startfeld wird ermittelt
        if start_field is None:
            break

    num_paare += 1 # Zähler erhöhen
    grid[start_field] = num_paare # Startfeld des Zahlenpaars wird im Gitter markiert

    # Genieren einer Linie, die eine zufällige Länge hat:
    position = start_field
    direction = -n

    # (Maximale) Länge der Linie festlegen. Hierfür wird entweder ein Wert aus dem Seed oder - wenn der Seed keinen Wert enthält - ein zufällig gewählter Wert gewählt
    if input_seed == []:
        iterations = random.randint(n,2*n) # Durch Ausprobieren hat sich herausgestellt, dass (n,2*n) ein geeigneter Zahlenbereich für die Linienlänge ist
    else:
        iterations = int(input_seed.pop(0))
    output_seed.append(str(iterations)) # Festgelegte Länge zu Output-Seed hinzufügen
    for i in range(iterations):
        # Bei jeder Iteration wird versucht, die Linie in die Richtung direction fortzusetzen. Wenn dabei auf ein Hindernis (eine andere Linie, eine Zahl oder den Rand vom Feld) gestoßen wird, wird die Richtung geändert
        directions = [-n, 1, n, -1]
        new_position = position + direction
        # Die while-Schleife wird ausgeführt, wenn die Linienerweiterung nach new_position nicht möglich ist:
        while not is_allowed_step(position, new_position, num_paare):
            # Wenn die Linie in die Richtung direction nicht fortgesetzt werden kann, dann wird solange eine neue Richtung ausprobiert, bis eine Richtung gefunden wurde, in die sie fortgesetzt werden kann (oder alle Richtungen durchprobiert wurden)
            if directions == []:
                # Wenn alle Richtungen durchprobiert wurden und die Linie nicht fortgesetzt werden kann, dann wird dies gespeichert und die Schleife abgebrochen
                found_direction = False
                break
            direction = directions.pop(0)
            new_position = position + direction
        else:
            # Dieser Programmteil wird ausgeführt, wenn eine Richtung gefunden wurde, in die die Linie fortgesetzt werden kann
            found_direction = True

        if found_direction:
            # Linie kann fortgesetzt werden:
            grid[new_position] = "."+str(num_paare) # Neuen zur Linie gehörenden Punkt in Gitter markieren
            position = new_position # Position aktualisieren
            continue
        else:
            # Linie kann nicht fortgesetzt werden -> Linie beenden
            break

    grid[position] = num_paare # Endfeld des Zahlenpaars in Gitter markieren

# Gittergröße, Anzahl platzierte Paare, mit Zahlen gefülltes Gitter, Lösung des Rätsels und Seed ausgeben:
seed = ";".join(output_seed) # Daten für den Seed zu einem zusammenhängenden String zusammenfügen
str_raetsel = ""
str_loesung = ""
for row in range(n):
    output_raetsel = ""
    output_loesung = ""
    for column in range(n):
        field = str(grid.pop(0))
        output_raetsel += str(0 if "." in field else field) + " "
        output_loesung += str("("+field[1:]+")" if "." in field else "["+field+"]") + " "
    str_raetsel += output_raetsel + "\n"
    str_loesung += output_loesung + "\n"

print("Generiertes Rätsel:\n"+str(n)+"\n"+str(num_paare)+"\n"+str_raetsel)
print("Lösung des generierten Rätsels:\n"+str_loesung)
print("Seed zum Reproduzieren:\n"+seed)
