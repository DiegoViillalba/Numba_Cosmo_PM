import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
# FILENAME = "build/data/snap_0000.bin"   
FILENAME = "resultados/snap_0240.bin"

# 1. Leer el archivo ignorando el primer valor problemático por ahora
# Sabemos que el total de datos es 14,680,064
raw_data = np.fromfile(FILENAME, dtype=np.float64)

# 2. Si el tamaño total es 14,680,065, el primero es el header.
# Si es 14,680,064, significa que NO hay header y son puras partículas.
if len(raw_data) % 7 == 0:
    print("Detectado: Archivo sin header (solo partículas).")
    particles_data = raw_data.reshape(-1, 7)
else:
    print(f"Detectado: Archivo con header. Valor del header: {raw_data[0]}")
    # Forzamos el reshape con los datos restantes
    particles_data = raw_data[1:].reshape(-1, 7)

# 3. Extraer Posiciones (x, y, z son las primeras 3 columnas)
pos = particles_data[:, 0:3]

print(f"Total de partículas procesadas: {len(particles_data)}")

# 4. Histograma de Densidad 2D con escala logarítmica
plt.figure(figsize=(10, 10), facecolor='black')

# Usamos LogNorm para que el colormap 'magma' represente el log de la densidad.
# El parámetro 'bins=256' es ideal si Ng=256 o Ng=128.
h = plt.hist2d(pos[:, 0], pos[:, 1], bins=256, cmap='magma')

plt.title("Distribución de Materia (Log-Density) - pm_cosmo", color='white', pad=20)
plt.axis('off')

# Añadir una barra de color suele ayudar a interpretar la densidad logarítmica
cbar = plt.colorbar(h[3], ax=plt.gca(), fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors='white')

plt.savefig('resultado_log.png', dpi=300, bbox_inches='tight', facecolor='black')
plt.show()