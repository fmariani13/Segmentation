import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# Definizione dati
TMAX = 10
t = np.linspace(0, TMAX, 100)

# Generazione dei segnali più complessi
x1 = np.sin(t) + np.cos(2 * t)  # Combinazione di seno e coseno
x2 = np.exp(-t)  # Funzione esponenziale decrescente
x3 = t**3 / TMAX**3  # Funzione cubica normalizzata
x4 = np.log1p(t) * np.sin(3 * t)  # Logaritmo moltiplicato per seno
x5 = np.tan(t / 2)  # Funzione tangente

# Stack delle feature
X = np.vstack([x1, x2, x3, x4, x5]).T  # Shape (100, 5)

# Variabile target (una combinazione più complessa di funzioni)
y_target = (t / TMAX).reshape(-1, 1)

def power2(x):
    return x**2
# Funzione per aggiungere la possibilità di elevare le variabili a potenza
power2_function = make_function(function=power2, name="power2", arity=1)

# Modello di regressione simbolica con altre funzioni
model = SymbolicRegressor(population_size=2000, generations=100, 
                          stopping_criteria=0.001, p_crossover=0.7, 
                          p_subtree_mutation=0.1, p_hoist_mutation=0.05, 
                          p_point_mutation=0.1, max_samples=0.9, 
                          verbose=1, random_state=42, 
                          function_set=['add', 'sub', 'mul', 'div', 
                                        'sin', 'cos', 'tan', 
                                        power2_function])  # Aggiungi qui le funzioni

# Addestramento
model.fit(X, y_target)

# Stampa della formula trovata
print("Formula trovata:", model._program)

# Previsione sulla variabile target
y_pred = model.predict(X)

# Creazione del grafico
plt.figure(figsize=(8, 6))
plt.plot(t, y_target, label='Target curve', color='blue', linewidth=2)
plt.plot(t, y_pred, label='Fitted curve', color='red', linestyle='--', linewidth=2)
plt.xlabel('Time (t)')
plt.ylabel('y')
plt.title('Target Curve vs Fitted Curve with Complex Signals')
plt.legend()
plt.grid(True)
plt.show()
