import sys

# Textdatei einlesen
# sys.argv[1] ist das erste Kommandozeilenargument
with open(sys.argv[1]) as f:
    input_lines = f.read().split("\n")

class Layer:
    """
    Klasse zum Speichern einer Schicht in einer Konstruktion
    """

    def __init__(self, aufbau : list):
        self.aufbau = aufbau

    # Als nächstes kommen Funktionen, die definieren, wie die einzelnen Bausteine bzw. Komponenten mit Signalen umgehen
    # Jede Funktion repräsentiert die Funktionsweise einer Komponente
    # Alle Funktionen funktionieren dabei nach dem gleichen Prinzip:
    # Die ersten beiden Elemente werden aus inputs entnommen bzw. entfernt und als Signale verarbeitet
    # Die verarbeiteten Signale werden dann an outputs angehängt

    def X(self, inputs, outputs):
        outputs.append(False)
        return inputs[1:], outputs

    def WW(self, inputs, outputs):
        sensor1, sensor2 = inputs[:2]
        outputs = outputs + [not (sensor1 and sensor2), not (sensor1 and sensor2)]
        return inputs[2:], outputs

    def BB(self, inputs, outputs):
        sensor1, sensor2 = inputs[:2]
        outputs = outputs + [sensor1, sensor2]
        return inputs[2:], outputs

    def Rr(self, inputs, outputs):
        sensor1, sensor2 = inputs[:2]
        outputs = outputs + [not sensor1, not sensor1]
        return inputs[2:], outputs

    def rR(self, inputs, outputs):
        sensor1, sensor2 = inputs[:2]
        outputs = outputs + [not sensor2, not sensor2]
        return inputs[2:], outputs

    def forward(self, inputs):
        """
        Wenn die Schicht weder Eingabeschicht noch Ausgabeschicht ist, dann kann mit dieser Funktion der Input der vorherigen Schicht eingelesen und verarbeitet werden

        Returns:
            list: Die Ausgaben der Schicht, die von der nächsten Schicht eingelesen werden können
        """
        outputs = []
        for component in self.aufbau:
            if component == "X":
                inputs, outputs = self.X(inputs, outputs)
            if component == "WW":
                inputs, outputs = self.WW(inputs, outputs)
            if component == "BB":
                inputs, outputs = self.BB(inputs, outputs)
            if component == "Rr":
                inputs, outputs = self.Rr(inputs, outputs)
            if component == "rR":
                inputs, outputs = self.rR(inputs, outputs)
        return outputs

    def input_layer(self, inputs):
        """
        Wenn die Schicht Eingabeschicht (die mit den Taschenlampen) ist, dann können mit dieser Funktion die Zustände der Taschenlampen eingelesen werden

        Returns:
            list: Die Zustände der Taschenlampen in einer Form, die von der nächsten Schicht eingelesen werden kann
        """
        outputs = []
        for component in self.aufbau:
            if component == "X":
                outputs.append(False)
            if "Q" in component:
                q_index = int(component[1:])-1
                outputs.append(inputs[q_index])
        return outputs

    def output_layer(self, inputs):
        """
        Wenn die Schicht Ausgabeschicht (die mit den LEDs) ist, dann kann mit dieser Funktion die Ausgabe der vorherigen Schicht eingelesen werden

        Returns:
            list: Die Zustände der LEDs
        """
        outputs = []
        for component in self.aufbau:
            if "L" in component:
                l_index = int(component[1:])-1
                while len(outputs) < l_index + 1:
                    outputs.append(None)
                outputs[l_index] = inputs.pop(0)
            else:
                inputs.pop(0)
        return outputs

# Parameter einlesen:
line = input_lines.pop(0).split(" ")
n = int(line[0])
m = int(line[1])

# Konstruktion einlesen:
konstruktion = []
num_taschenlampen = 0
num_leds = 0
for i in range(m):
    line = input_lines.pop(0).replace("  ", " ").split(" ")[:n]
    aufbau = []
    while line != []:
        component = line.pop(0)
        if "Q" in component:
            num_taschenlampen += 1
        elif "L" in component:
            num_leds += 1
        elif not component == "X":
            component += line.pop(0)
        aufbau.append(component)
    konstruktion.append(Layer(aufbau))

# Alle möglichen Kombinationen der Taschenlampen (ein / aus) ermitteln:
kombinationen = [[]]
for t in range(num_taschenlampen):
    for k in list(kombinationen):
        kombinationen.remove(k)
        kombinationen.append(k+[False])
        kombinationen.append(k+[True])

# Für jede Kombination der Taschenlampen (ein / aus) die Zustände der LEDs (ein / aus) ermitteln:
leds = []
for k in kombinationen:
    output = konstruktion[0].input_layer(k)
    for layer in konstruktion[1:-1]:
        output = layer.forward(output)
    output = konstruktion[-1].output_layer(output)
    leds.append(output)

# Ermittelte Zustände der LEDs in Tabellenform ausgeben:
zustände = {True:"An", False:"Aus"}
print("   | ".join(
    [f"Q{q+1}" for q in range(num_taschenlampen)]+
    [f"L{l+1}" for l in range(num_leds)]
))
for i in range(len(kombinationen)):
    print("| ".join(
        ["{:<5}".format(zustände[kombinationen[i][q]]) for q in range(num_taschenlampen)]+
        ["{:<5}".format(zustände[leds[i][l]]) for l in range(num_leds)]
    ))
