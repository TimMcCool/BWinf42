import numpy as np
import sys

# Textdatei einlesen
# sys.argv[1] ist das erste Kommandozeilenargument
with open("C:/Users/timkr/OneDrive/Projekte/BWInf 24/BWinf 24 Tim Krome Runde 1/A2_DieGoldeneMitte/raetsel1.txt") as f:
    input_lines = f.read().split("\n")

class Kiste:

    """
    Dient zum Speichern einer teilweise befüllten Kiste
    """

    def __init__(self, kiste_dims, inhalt, quaders, num_placed_quaders=0, *, symmetric):
        self.dims = kiste_dims # Die Seitenlängen der Kiste (tuple)
        self.inhalt = inhalt # Der Inhalt der Liste (np.array)
        self.quaders = dict(quaders) # Ein dictionary mit allen Quadern, die noch in die Kiste gefüllt werden müssen
        self.num_placed_quaders = num_placed_quaders # Anzahl an Quadern, die bisher in die Kiste gefüllt wurden
        self.symmetric = symmetric # Gibt an, ob die Befüllung der Kiste punktsymmetrisch zur Mitte (zum goldenen Würfel) sein soll

    def place_quader(self, quader_dims, position, *, mirror=False):
        """
        Platziert einen Quader in der Kiste an der angebenen Position, sofern dies möglich ist

        Args:
            quader_dims (tuple): Die Seitenlängen des Quaders, der platziert werden soll
            position (tuple): Die Koordinaten der Position, an der der Quader platziert werden soll
            mirror (boolean): Gibt an, ob punktsymmetrisch zum goldenen Würfel ein weitere Quader der gleichen Seitenlängen platziert werden soll
        """
        new_inhalt = np.array(self.inhalt)
        range0 = (position[0],position[0]+quader_dims[0])
        range1 = (position[1],position[1]+quader_dims[1])
        range2 = (position[2],position[2]+quader_dims[2])
        if range0[1] > self.dims[0] or range1[1] > self.dims[1] or range2[1] > self.dims[2]:
            return None
        if False in (new_inhalt[ range0[0]:range0[1], range1[0]:range1[1], range2[0]:range2[1]] == 0):
            return None
        new_inhalt[ range0[0]:range0[1], range1[0]:range1[1], range2[0]:range2[1]] = self.num_placed_quaders + 1
        if mirror:
            new_inhalt[ self.dims[0]-range0[1]:self.dims[0]-range0[0], self.dims[1]-range1[1]:self.dims[1]-range1[0], self.dims[2]-range2[1]:self.dims[2]-range2[0]] = self.num_placed_quaders + 2

        return new_inhalt

    def get_corner_index(self):
        """
        Eine beliebige innere Ecke der Kiste ermitteln
        """
        for index0 in range(self.dims[0]):
            for index1 in range(self.dims[1]):
                for index2 in range(self.dims[2]):
                    if self.inhalt[index0, index1, index2] != 0:
                        continue

                    if self.inhalt[index0-1, index1, index2] != 0 or index0-1 < 0:
                        if self.inhalt[index0, index1-1, index2] != 0 or index1-1 < 0:
                            if self.inhalt[index0, index1, index2-1] != 0 or index2-1 < 0:
                                return index0, index1, index2

    def next_kisten(self):
        """
        Ermittelt alle Kisten, die aus dieser Kiste hervorgehen können
        """
        corner_position = self.get_corner_index() # Eckposition festlegen, an der die zu Verfüung stehenden Quader platziert werden sollen
        next_kisten = []
        # Über alle noch nicht platzierten Quader iterieren:
        for quader in list(self.quaders.keys()):
            next_quaders = dict(self.quaders)
            next_quaders[quader] -= 1 + self.symmetric
            if next_quaders[quader] == -1:
                return [] # -> Kiste kann nicht punktsymmetrisch befüllt werden
            if next_quaders[quader] == 0:
                next_quaders.pop(quader)

            # Alle möglichen Rotationen des Quaders ermitteln:
            quader_rotations = []
            for i in range(2):
                for i in range(3):
                    quader = (quader[1], quader[2], quader[0])
                    quader_rotations.append(quader)
                quader = quader[::-1]


            # Über alle möglichen Rotationen des Quaders iterieren:
            for quader in set(quader_rotations): # quader_rotations zu Set konvertieren, um Duplikate zu entfernen
                new_inhalt = self.place_quader(quader, corner_position, mirror=self.symmetric) # Versuchen, den rotierten Quader an der festgelegten Eckposition hinzuzufügen
                if new_inhalt is None:
                    # Platzieren des rotierten Quaders an der Eckposition nicht möglich -> Nächste Rotation ausprobieren
                    continue
                else:
                    # Platzieren des rotierten Quaders erfolgreich -> Resultat speichern
                    next_kisten.append(Kiste(self.dims, new_inhalt, next_quaders, self.num_placed_quaders+self.symmetric+1, symmetric=self.symmetric))

                if self.num_placed_quaders == 0:
                    break

        return next_kisten

    def print(self):
        """
        Gibt den Inhalt der Kiste in der Konsole aus
        """
        for ebene in range(len(self.inhalt)):
            print("\nEbene", ebene+1)
            for row in self.inhalt[ebene]:
                output = ""
                for item in row:
                    item = str(item)
                    if item == "-1":
                        item = "G"
                    while len(item) < len(str(num_quaders)):
                        item = " "+item
                    output += str(item) + "    "
                print(output)


# Größe der Kiste einlesen:
data = input_lines.pop(0).split(" ")
kiste_dims = (int(data[0]), int(data[1]), int(data[2]))
kiste_gesamtvolumen = kiste_dims[0] * kiste_dims[1] * kiste_dims[2]

# Größen der Quader einlesen:
num_quaders = int(input_lines.pop(0))
quaders = {}
quaders_gesamtvolumen = 1
for i in range(num_quaders):
    data = input_lines.pop(0).split(" ")
    data = (int(data[0]), int(data[1]), int(data[2]))
    if data in quaders:
        quaders[data] += 1
    else:
        quaders[data] = 1
    quaders_gesamtvolumen += int(data[0]) * int(data[1]) * int(data[2])

# Überprüfen, ob Gesamtvolumen der Quader dem Volumen der Kiste entspricht (wenn nicht, dann können die Quader gar nicht alle in die Kiste passen):
if kiste_gesamtvolumen != quaders_gesamtvolumen:
    print("Es kann keine Lösung geben, da des Gesamtvolumen der einzufüllenden Quader (einschl. dem goldenen Würfel) nicht dem Volumen der Kiste entspricht.")
    exit()

# Startkiste erzeugen:
inhalt = np.zeros(kiste_dims, dtype=int)
mitte = (int(kiste_dims[0]/2),int(kiste_dims[1]/2),int(kiste_dims[2]/2))
inhalt[mitte] = -1
alle_kisten = [Kiste(kiste_dims, inhalt, quaders, symmetric=True), Kiste(kiste_dims, inhalt, quaders, symmetric=False)]

# Alle Möglichkeiten, innere Ecken zu eliminieren, durchprobieren:

while alle_kisten != []:

    kiste = alle_kisten.pop(0)
    if kiste.num_placed_quaders == num_quaders:

        # Lösung gefunden
        print("Lösung gefunden")
        kiste.print()
        break

    next_kisten = kiste.next_kisten()
    alle_kisten = next_kisten + alle_kisten

else:

    print("Es gibt keine Lösung")
