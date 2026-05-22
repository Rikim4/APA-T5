"""
estereo.py

Autor: Ricard Mula Cañameras

Módulo para el manejo de los canales de una señal estéreo almacenada en ficheros
WAVE PCM de 16 bits, y su codificación/decodificación para compatibilidad con
sistemas monofónicos.

Solo se usa la biblioteca estándar `struct` para la lectura/escritura de datos binarios.

Funciones:
    estereo2mono  -- Extrae un canal (o semisuma/semidiferencia) de un WAVE estéreo.
    mono2estereo  -- Construye un WAVE estéreo a partir de dos WAVE monofónicos.
    codEstereo    -- Codifica un WAVE estéreo en 32 bits (semisuma + semidiferencia).
    decEstereo    -- Decodifica un WAVE de 32 bits y recupera el estéreo de 16 bits.
"""

import struct


# ---------------------------------------------------------------------------
# Helpers: cabecera WAVE
# ---------------------------------------------------------------------------

def _lee_cabecera(f):
    """
    Lee y valida la cabecera de un fichero WAVE PCM.

    Devuelve un diccionario con los campos de la cabecera y la posición del
    primer byte de datos (justo después del subcacho 'data').

    Eleva ValueError si el fichero no tiene el formato WAVE PCM esperado.
    """
    # --- Cacho RIFF ---
    riff_id = f.read(4)
    if riff_id != b'RIFF':
        raise ValueError("El fichero no empieza por 'RIFF'.")
    (riff_size,) = struct.unpack('<I', f.read(4))
    wave_id = f.read(4)
    if wave_id != b'WAVE':
        raise ValueError("El fichero no contiene la marca 'WAVE'.")

    # --- Subcacho fmt  ---
    fmt_id = f.read(4)
    if fmt_id != b'fmt ':
        raise ValueError("No se encuentra el subcacho 'fmt '.")
    (fmt_size,) = struct.unpack('<I', f.read(4))
    fmt_data = f.read(fmt_size)

    if fmt_size < 16:
        raise ValueError("Subcacho 'fmt ' demasiado pequeño.")

    (audio_format, num_channels, sample_rate,
     byte_rate, block_align, bits_per_sample) = struct.unpack('<HHIIHH', fmt_data[:16])

    if audio_format != 1:
        raise ValueError("Solo se admite PCM lineal (AudioFormat=1).")

    # --- Subcacho data ---
    # Puede haber subcachos intermedios; los saltamos hasta encontrar 'data'.
    while True:
        chunk_id = f.read(4)
        if len(chunk_id) < 4:
            raise ValueError("No se encuentra el subcacho 'data'.")
        (chunk_size,) = struct.unpack('<I', f.read(4))
        if chunk_id == b'data':
            data_size = chunk_size
            break
        f.read(chunk_size)   # Saltamos subcachos desconocidos

    return {
        'riff_size':      riff_size,
        'num_channels':   num_channels,
        'sample_rate':    sample_rate,
        'byte_rate':      byte_rate,
        'block_align':    block_align,
        'bits_per_sample': bits_per_sample,
        'data_size':      data_size,
    }


def _escribe_cabecera(f, num_channels, sample_rate, bits_per_sample, num_samples):
    """
    Escribe una cabecera WAVE PCM estándar en el fichero `f`.
    """
    byte_rate   = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size   = num_samples * num_channels * bits_per_sample // 8
    riff_size   = 36 + data_size          # 4 (WAVE) + 24 (fmt ) + 8 (data hdr) + data

    f.write(b'RIFF')
    f.write(struct.pack('<I', riff_size))
    f.write(b'WAVE')

    f.write(b'fmt ')
    f.write(struct.pack('<I', 16))        # Tamaño del subcacho fmt
    f.write(struct.pack('<H', 1))         # AudioFormat = PCM
    f.write(struct.pack('<H', num_channels))
    f.write(struct.pack('<I', sample_rate))
    f.write(struct.pack('<I', byte_rate))
    f.write(struct.pack('<H', block_align))
    f.write(struct.pack('<H', bits_per_sample))

    f.write(b'data')
    f.write(struct.pack('<I', data_size))


# ---------------------------------------------------------------------------
# Función 1: estereo2mono
# ---------------------------------------------------------------------------

def estereo2mono(ficEste, ficMono, canal=2):
    """
    Lee el fichero WAVE estéreo `ficEste` y escribe `ficMono` con una señal
    monofónica según el valor de `canal`:

        0 -> canal izquierdo  L
        1 -> canal derecho    R
        2 -> semisuma         (L + R) / 2   [por defecto]
        3 -> semidiferencia   (L - R) / 2

    Eleva ValueError si el fichero de entrada no es estéreo de 16 bits PCM,
    o si `canal` no está entre 0 y 3.
    """
    if canal not in (0, 1, 2, 3):
        raise ValueError(f"canal debe ser 0, 1, 2 o 3 (recibido: {canal}).")

    with open(ficEste, 'rb') as f:
        cab = _lee_cabecera(f)

        if cab['num_channels'] != 2:
            raise ValueError("El fichero de entrada no es estéreo.")
        if cab['bits_per_sample'] != 16:
            raise ValueError("Solo se admiten muestras de 16 bits.")

        num_muestras = cab['data_size'] // 4   # 2 canales × 2 bytes
        datos_raw = f.read(cab['data_size'])

    # Desempaquetamos todas las muestras (interleaved L, R, L, R, ...)
    muestras = struct.unpack(f'<{num_muestras * 2}h', datos_raw)
    L = muestras[0::2]
    R = muestras[1::2]

    if canal == 0:
        mono = L
    elif canal == 1:
        mono = R
    elif canal == 2:
        mono = tuple((l + r) // 2 for l, r in zip(L, R))
    else:  # canal == 3
        mono = tuple((l - r) // 2 for l, r in zip(L, R))

    with open(ficMono, 'wb') as f:
        _escribe_cabecera(f, num_channels=1,
                          sample_rate=cab['sample_rate'],
                          bits_per_sample=16,
                          num_samples=num_muestras)
        f.write(struct.pack(f'<{num_muestras}h', *mono))


# ---------------------------------------------------------------------------
# Función 2: mono2estereo
# ---------------------------------------------------------------------------

def mono2estereo(ficIzq, ficDer, ficEste):
    """
    Lee los ficheros WAVE monofónicos `ficIzq` (canal L) y `ficDer` (canal R)
    y construye el fichero WAVE estéreo `ficEste` intercalando sus muestras.

    Eleva ValueError si alguno de los ficheros no es monofónico PCM de 16 bits,
    o si tienen distinta frecuencia de muestreo o distinto número de muestras.
    """
    with open(ficIzq, 'rb') as f:
        cab_L = _lee_cabecera(f)
        if cab_L['num_channels'] != 1:
            raise ValueError("ficIzq no es monofónico.")
        if cab_L['bits_per_sample'] != 16:
            raise ValueError("ficIzq no tiene muestras de 16 bits.")
        n_L = cab_L['data_size'] // 2
        datos_L = struct.unpack(f'<{n_L}h', f.read(cab_L['data_size']))

    with open(ficDer, 'rb') as f:
        cab_R = _lee_cabecera(f)
        if cab_R['num_channels'] != 1:
            raise ValueError("ficDer no es monofónico.")
        if cab_R['bits_per_sample'] != 16:
            raise ValueError("ficDer no tiene muestras de 16 bits.")
        n_R = cab_R['data_size'] // 2
        datos_R = struct.unpack(f'<{n_R}h', f.read(cab_R['data_size']))

    if cab_L['sample_rate'] != cab_R['sample_rate']:
        raise ValueError("Los ficheros tienen distinta frecuencia de muestreo.")
    if n_L != n_R:
        raise ValueError("Los ficheros tienen distinto número de muestras.")

    # Intercalamos L y R: L[0], R[0], L[1], R[1], ...
    interleaved = tuple(v for par in zip(datos_L, datos_R) for v in par)

    with open(ficEste, 'wb') as f:
        _escribe_cabecera(f, num_channels=2,
                          sample_rate=cab_L['sample_rate'],
                          bits_per_sample=16,
                          num_samples=n_L)
        f.write(struct.pack(f'<{len(interleaved)}h', *interleaved))


# ---------------------------------------------------------------------------
# Función 3: codEstereo
# ---------------------------------------------------------------------------

def codEstereo(ficEste, ficCod):
    """
    Lee el fichero WAVE estéreo PCM de 16 bits `ficEste` y escribe `ficCod`,
    un fichero WAVE monofónico de 32 bits donde:

        bits 31-16  →  semisuma     (L + R) / 2
        bits 15-0   →  semidiferencia (L - R) / 2

    Los reproductores monofónicos de 32 bits sólo percibirán la semisuma;
    la semidiferencia aparece como ruido ~90 dB por debajo.
    """
    with open(ficEste, 'rb') as f:
        cab = _lee_cabecera(f)
        if cab['num_channels'] != 2:
            raise ValueError("El fichero de entrada no es estéreo.")
        if cab['bits_per_sample'] != 16:
            raise ValueError("Solo se admiten muestras de 16 bits.")
        num_muestras = cab['data_size'] // 4
        datos_raw = f.read(cab['data_size'])

    muestras = struct.unpack(f'<{num_muestras * 2}h', datos_raw)
    L = muestras[0::2]
    R = muestras[1::2]

    # Semisuma en los 16 bits altos, semidiferencia en los 16 bits bajos.
    # Empaquetamos como enteros de 32 bits con signo.
    codificadas = tuple(
        (((l + r) // 2) << 16) | (((l - r) // 2) & 0xFFFF)
        for l, r in zip(L, R)
    )

    with open(ficCod, 'wb') as f:
        _escribe_cabecera(f, num_channels=1,
                          sample_rate=cab['sample_rate'],
                          bits_per_sample=32,
                          num_samples=num_muestras)
        f.write(struct.pack(f'<{num_muestras}i', *codificadas))


# ---------------------------------------------------------------------------
# Función 4: decEstereo
# ---------------------------------------------------------------------------

def decEstereo(ficCod, ficEste):
    """
    Lee el fichero WAVE monofónico de 32 bits `ficCod` (generado por
    `codEstereo`) y escribe el fichero WAVE estéreo de 16 bits `ficEste`
    recuperando los canales L y R.

        semisuma     = bits 31-16  →  S = muestra >> 16
        semidiferencia = bits 15-0  →  D = (muestra & 0xFFFF) como entero con signo
        L = S + D
        R = S - D

    Eleva ValueError si el fichero no es monofónico PCM de 32 bits.
    """
    with open(ficCod, 'rb') as f:
        cab = _lee_cabecera(f)
        if cab['num_channels'] != 1:
            raise ValueError("El fichero codificado no es monofónico.")
        if cab['bits_per_sample'] != 32:
            raise ValueError("El fichero codificado no tiene muestras de 32 bits.")
        num_muestras = cab['data_size'] // 4
        datos_raw = f.read(cab['data_size'])

    codificadas = struct.unpack(f'<{num_muestras}i', datos_raw)

    # Extraemos semisuma (16 bits altos) y semidiferencia (16 bits bajos con signo)
    S = tuple(c >> 16 for c in codificadas)
    D = tuple(struct.unpack('<h', struct.pack('<H', c & 0xFFFF))[0]
              for c in codificadas)

    L = tuple(s + d for s, d in zip(S, D))
    R = tuple(s - d for s, d in zip(S, D))

    # Saturamos a rango [-32768, 32767] por seguridad
    L = tuple(max(-32768, min(32767, v)) for v in L)
    R = tuple(max(-32768, min(32767, v)) for v in R)

    interleaved = tuple(v for par in zip(L, R) for v in par)

    with open(ficEste, 'wb') as f:
        _escribe_cabecera(f, num_channels=2,
                          sample_rate=cab['sample_rate'],
                          bits_per_sample=16,
                          num_samples=num_muestras)
        f.write(struct.pack(f'<{len(interleaved)}h', *interleaved))
