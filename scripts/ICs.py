# -----------------------------------------------------------------------------
# Initical conditions aproximation using python
# -----------------------------------------------------------------------------

import numpy as np

def power_spectrum_eh(k, h=0.674, Om=0.315, Ob=0.049, ns=0.965):
    """
    Espectro de Eisenstein & Hu para cosmología de referencia Planck 2018.
    Se basa en la aproximación de Eisenstein & Hu (1998).
    
    Args:
        k (np.ndarray): Vector de números de onda en unidades de Mpc^-1
        h (float): Parámetro de Hubble
        Om (float): Parámetro de densidad de materia
        Ob (float): Parámetro de densidad de bariones
        ns (float): Índice escalar del espectro de potencia
    
    Returns:
        np.ndarray: Espectro de potencia en unidades de (Mpc/h)^3
    """
    theta_2p7 = 2.725 / 2.7 
    keq = 7.46e-2 * Om * h**2 * theta_2p7**-2
    q = k / (keq * h)
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731 / (1 + 62.5 * q)
    T = L0 / (L0 + C0 * q**2)
    return k**ns * T**2

def growth_factor(z, Om=0.315):
    """Factor de crecimiento lineal D(a) aproximado para Lambda-CDM."""
    a = 1.0 / (1.0 + z)
    # Aproximación de Carroll et al. (1992)
    return a * (2.5 * Om) / (Om**(4/7) - (1 - Om) + (1 + 0.5 * Om) * (1 + (1 - Om)/70))

def generate_ics_for_cpp(Ng, Lbox, z_ini=50.0, sigma8=0.81, seed=42):
    """
    Genera ICs siguiendo la Aproximación de Zel'dovich (ZA).
    """
    np.random.seed(seed)
    h = 0.674  # Consistente con Planck 2018
    
    # 1. Configuración de la malla y frecuencias
    k_lin = np.fft.fftfreq(Ng) * (2 * np.pi)
    kx, ky, kz = np.meshgrid(k_lin, k_lin, k_lin, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0 

    # 2. Espectro de Potencia y Normalización sigma8
    k_phys = np.sqrt(k_sq) * (Ng / Lbox)
    Pk_raw = power_spectrum_eh(k_phys, h=h)
    
    # Normalización: Calculamos sigma8 del espectro crudo
    R8 = 8.0 / h
    window = 3 * (np.sin(k_phys * R8) - k_phys * R8 * np.cos(k_phys * R8)) / (k_phys * R8)**3
    window[0, 0, 0] = 1.0
    sigma8_current = np.sqrt(np.sum(Pk_raw * window**2 * k_phys**2 * (k_phys[1,0,0]-k_phys[0,0,0])) / (2 * np.pi**2)) # Simplificado
    # Para fines prácticos, escalamos Pk para que coincida con sigma8 a z=0
    norm = (sigma8 / 0.05) # Factor de ajuste manual basado en tu implementación previa o integración numérica
    Pk = Pk_raw * norm

    # 3. Campo de desplazamiento (Psi)
    # Usamos la varianza para asegurar que delta_k sea adimensional en la grilla
    vol_factor = (Ng**3) / (Lbox**3)
    white_noise = np.random.normal(size=(Ng, Ng, Ng)) + 1j * np.random.normal(size=(Ng, Ng, Ng))
    delta_k = white_noise * np.sqrt(Pk * vol_factor)
    delta_k[0,0,0] = 0

    psi_x = np.fft.ifftn(-1j * kx * delta_k / k_sq).real
    psi_y = np.fft.ifftn(-1j * ky * delta_k / k_sq).real
    psi_z = np.fft.ifftn(-1j * kz * delta_k / k_sq).real

    # 4. Cálculo de Amplitud (D(z) y factor de crecimiento)
    Dz = growth_factor(z_ini) / growth_factor(0.0)
    
    # En ZA: x = q + D(z) * Psi
    # Como Psi ya está en unidades de "frecuencia de grilla", 
    # la amplitud física escala el desplazamiento.
    amplitude = Dz 

    q_lin = np.arange(Ng)
    QX, QY, QZ = np.meshgrid(q_lin, q_lin, q_lin, indexing='ij')
    
    pos = np.stack([QX + amplitude * psi_x, 
                    QY + amplitude * psi_y, 
                    QZ + amplitude * psi_z], axis=-1).reshape(-1, 3)
    
    # 5. Velocidades (v = a * H * f * D * Psi)
    # f es dlnD/dlna approx Om(z)^0.55
    Om_z = 0.315 * (1+z_ini)**3 / (0.315 * (1+z_ini)**3 + 0.685)
    f_growth = Om_z**0.55
    
    # Factor de velocidad para que sea consistente con pos en unidades de celda
    # v_code = dx/dt = (a * H * f * displacement) / (unidades_de_tiempo)
    H_z = 100 * h * np.sqrt(0.315 * (1+z_ini)**3 + 0.685) # km/s/Mpc
    
    # Para tu simulador C++, solemos usar un v_factor que relacione Psi con el kick
    v_factor = amplitude * f_growth * (H_z / (1 + z_ini))
    
    vel = v_factor * np.stack([psi_x, psi_y, psi_z], axis=-1).reshape(-1, 3)

    return np.mod(pos, Ng), vel

def generate_ics_with_lpt_kick(Ng, Lbox, z_target=10.0, sigma8=0.81, seed=42):
    np.random.seed(seed)
    h = 0.674  

    # 1. Configuración de k
    k_lin = np.fft.fftfreq(Ng) * (2 * np.pi)
    kx, ky, kz = np.meshgrid(k_lin, k_lin, k_lin, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    
    # IMPORTANTE: k_sq[0,0,0] debe ser 1 para evitar division por cero, 
    # pero luego delta_k[0,0,0] debe ser 0.
    k_sq_safe = np.where(k_sq == 0, 1.0, k_sq)
    k_phys = np.sqrt(k_sq_safe) * (Ng / Lbox)
    
    # 2. Espectro y Normalización robusta
    Pk_raw = power_spectrum_eh(k_phys, h=h)
    
    # Normalización sigma8 (Integración en k)
    R8 = 8.0 / h
    # Función ventana Top-hat
    window = np.where(k_phys > 0, 
                      3 * (np.sin(k_phys * R8) - k_phys * R8 * np.cos(k_phys * R8)) / (k_phys * R8)**3, 
                      1.0)
    
    dk = k_phys[1,0,0] - k_phys[0,0,0]
    # Filtramos k=0 para la suma
    mask = (k_sq > 0)
    sigma8_sq = np.sum((Pk_raw * window**2 * k_phys**2)[mask]) * dk / (2 * np.pi**2)
    sigma8_current = np.sqrt(max(1e-10, sigma8_sq))
    
    # Escalamiento
    Pk = Pk_raw * (sigma8 / sigma8_current)**2

    # 3. Desplazamiento Psi (LPT)
    vol_factor = (Ng**3) / (Lbox**3)
    white_noise = np.random.normal(size=(Ng, Ng, Ng)) + 1j * np.random.normal(size=(Ng, Ng, Ng))
    delta_k = white_noise * np.sqrt(Pk * vol_factor)
    delta_k[~mask] = 0.0 # Forzar modo DC a cero

    # Psi = -i * (k / k^2) * delta
    psi_x = np.fft.ifftn(-1j * kx * delta_k / k_sq_safe).real
    psi_y = np.fft.ifftn(-1j * ky * delta_k / k_sq_safe).real
    psi_z = np.fft.ifftn(-1j * kz * delta_k / k_sq_safe).real

    # 4. Kick LPT al z_target
    Dz = growth_factor(z_target) / growth_factor(0.0)
    
    q_lin = np.arange(Ng)
    QX, QY, QZ = np.meshgrid(q_lin, q_lin, q_lin, indexing='ij')
    
    # Posiciones: q + Dz * Psi
    pos = np.stack([QX + Dz * psi_x, 
                    QY + Dz * psi_y, 
                    QZ + Dz * psi_z], axis=-1).reshape(-1, 3)

    # Velocidades LPT: v = a * H * f * Dz * Psi
    a = 1.0 / (1.0 + z_target)
    # Tasa de crecimiento lineal f ~ Omega_m(a)^0.55
    Om_z = 0.315 * (1+z_target)**3 / (0.315 * (1+z_target)**3 + 0.685)
    f_growth = Om_z**0.55
    H_z = 100 * h * np.sqrt(0.315 * (1+z_target)**3 + 0.685) # km/s/Mpc
    
    # Ajuste para unidades de código (distancia de celda / tiempo)
    v_factor = a * H_z * f_growth * Dz
    vel = v_factor * np.stack([psi_x, psi_y, psi_z], axis=-1).reshape(-1, 3)

    return np.mod(pos, Ng), vel


# --- PARÁMETROS IGUALES A TU CONFIG.HPP ---
NG = 128
L_BOX = 100.0

pos, vel = generate_ics_with_lpt_kick(NG, L_BOX)
data = np.hstack([pos, vel])

# --- GUARDADO EN FORMATO ASCII PARA IC_READER.CPP ---
filename = "ic/ic_128.dat"
with open(filename, "w") as f:
    # Línea 1: N_PART (Tu C++ hace f >> n_part)
    f.write(f"{len(pos)}\n")
    # Líneas 2 en adelante: x y z vx vy vz
    np.savetxt(f, data, fmt='%.8f')

print(f"¡Éxito! Archivo {filename} generado para tu simulador.")
print(f"Recuerda correrlo como: ./pm_parallel -i {filename}")