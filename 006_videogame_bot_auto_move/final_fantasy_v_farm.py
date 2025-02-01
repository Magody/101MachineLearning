import time
import pyautogui
import pygetwindow as gw
import cv2
import numpy as np
import win32gui
import win32con
import win32api
import random

path_references = './references_images'

# --- Cargamos las imágenes de referencia en memoria (en vez de hacerlo en cada detección) ---
exploracion_tpl = cv2.imread(f"{path_references}/exploracion.png", cv2.IMREAD_UNCHANGED)
combate_tpl = cv2.imread(f"{path_references}/combate.png", cv2.IMREAD_UNCHANGED)
fin_combate_tpl = cv2.imread(f"{path_references}/fin_combate.png", cv2.IMREAD_UNCHANGED)
objeto_recibido_tpl = cv2.imread(f"{path_references}/objeto_recibido.png", cv2.IMREAD_UNCHANGED)

exploracion_tpl_left = cv2.imread(f"{path_references}/exploracion-left.png", cv2.IMREAD_UNCHANGED)
exploracion_tpl_right = cv2.imread(f"{path_references}/exploracion-right.png", cv2.IMREAD_UNCHANGED)

# Función para normalizar imágenes (quitar canal alfa si existe)
def normalizar_imagen(imagen):
    if imagen is None:
        return None
    if imagen.shape[-1] == 4:
        return cv2.cvtColor(imagen, cv2.COLOR_BGRA2BGR)
    return imagen

# Normalizamos todas las referencias para evitar discrepancias en los canales
exploracion_tpl = normalizar_imagen(exploracion_tpl)
combate_tpl = normalizar_imagen(combate_tpl)
fin_combate_tpl = normalizar_imagen(fin_combate_tpl)
objeto_recibido_tpl = normalizar_imagen(objeto_recibido_tpl)
exploracion_tpl_left = normalizar_imagen(exploracion_tpl_left)
exploracion_tpl_right = normalizar_imagen(exploracion_tpl_right)

# Obtén el manejador (handle) de la ventana del juego
def obtener_handle_ventana(nombre_ventana):
    ventana = gw.getWindowsWithTitle(nombre_ventana)
    if ventana:
        return win32gui.FindWindow(None, ventana[0].title)
    else:
        raise Exception(f"No se encontró la ventana con el título: {nombre_ventana}")

# Envía una tecla a la ventana del juego
def enviar_tecla(handle, tecla, press_time=0.1):
    scan_code = win32api.MapVirtualKey(tecla, 0)
    # Enviar evento de tecla presionada
    win32api.PostMessage(handle, win32con.WM_KEYDOWN, tecla, (scan_code << 16) | 1)
    time.sleep(press_time)  # Pausa breve
    # Enviar evento de tecla liberada
    win32api.PostMessage(handle, win32con.WM_KEYUP, tecla, (scan_code << 16) | 1)

# Captura la pantalla de la ventana del juego
def capturar_ventana(coordenadas):
    x, y, w, h = coordenadas
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# Detectar escena en una captura dada (ya no recargamos la plantilla cada vez)
def detectar_escena(pantalla, plantilla, umbral=0.8, nombre_imagen=""):
    if plantilla is None:
        raise Exception("No se pudo cargar la imagen de referencia")

    # Asegurar que la captura no tenga canal alfa
    if pantalla.shape[-1] == 4:
        pantalla = cv2.cvtColor(pantalla, cv2.COLOR_BGRA2BGR)

    resultado = cv2.matchTemplate(pantalla, plantilla, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(resultado)
    print(f"Analizando: {nombre_imagen}. Umbral: {max_val >= umbral}. Similitud: {round(max_val, 2)}")  # Puedes descomentar para depurar
    return max_val >= umbral

# Mover al personaje en patrón cuadrado
def move(handle, time_press_left=0.12, time_press_right=0.12):
    
    for _ in range(5):
        # enviar_tecla(handle, win32con.VK_UP)
        enviar_tecla(handle, win32con.VK_RIGHT, press_time=time_press_right)
        time.sleep(0.01)
        # enviar_tecla(handle, win32con.VK_DOWN)
        enviar_tecla(handle, win32con.VK_LEFT, press_time=time_press_left)
        time.sleep(0.01)
        
# Bot principal
def bot_farmeo(nombre_ventana):
    handle = obtener_handle_ventana(nombre_ventana)
    ventana = gw.getWindowsWithTitle(nombre_ventana)[0]
    
    monitor_principal_ancho = 1920  

    x, y = ventana.left, ventana.top
    w, h = ventana.width, ventana.height

    # Si x es negativo, lo compensamos sumándole 1920
    # if x < 0:
    #     x_compensado = x + monitor_principal_ancho
    # else:
    x_compensado = x

    coordenadas = (x_compensado, y, w, h)
    
    print(f"Coordenadas de la ventana: {coordenadas}")

    umbral_fin_combate = 0.8
    umbral_combatiendo = 0.79
    
    umbral_explorando = 0.69
    umbral_explorando_left = 0.6
    umbral_explorando_right = 0.6
    
    fins = 0
    
    while fins < 100:
        time.sleep(0.1)  # Pausa breve para no saturar CPU
        try:
            # Capturamos solo UNA VEZ por iteración
            screenshot = capturar_ventana(coordenadas)

            # Fase 1 / Fase 4: Detección de fin de combate u objeto recibido
            if detectar_escena(screenshot, fin_combate_tpl, umbral_fin_combate, "fin_combate"):
                print("Fase 4: Fin del combate detectado. Presionando 'A'.")
                enviar_tecla(handle, ord('A'))  # Simula tecla 'A'
                time.sleep(1)  # Esperar por pantalla negra
                print("Volviendo a Fase 1.")
                fins += 1
            # Fase 3: Detección de combate
            elif detectar_escena(screenshot, combate_tpl, umbral_combatiendo, "combatiendo"):
                while True:
                    print("Fase 3: Combate detectado. Esperando fin del combate.")
                    time.sleep(1)  # Esperamos antes de verificar de nuevo
                    screenshot_combate = capturar_ventana(coordenadas)
                    if detectar_escena(screenshot_combate, fin_combate_tpl, umbral_fin_combate, "fin_combate"):
                        enviar_tecla(handle, ord('A'))
                        time.sleep(1)
                        fins += 1
                        print("FINS", fins)
                        break

            # Fase 1: Exploración
            elif detectar_escena(screenshot, exploracion_tpl, umbral_explorando, "explorando"):
                print("Fase 1: Exploración detectada. Moviendo.")
                move(handle, time_press_left=0.102, time_press_right=0.101)
                
            elif detectar_escena(screenshot, exploracion_tpl_left, umbral_explorando_left, "panel-izquierda"):
                print("Fase 1: Exploración detectada en offset izquierdo. Moviendo.")
                move(handle, time_press_left=0.10, time_press_right=0.2)
                
            elif detectar_escena(screenshot, exploracion_tpl_right, umbral_explorando_right, "panel-derecho"):
                print("Fase 1: Exploración detectada en offset derecho. Moviendo.")
                move(handle, time_press_left=0.2, time_press_right=0.10)
                
            elif detectar_escena(screenshot, objeto_recibido_tpl, 0.8, "objeto_recibido"):
                print("Fase 5: Recolectar objeto. Presionando 'A'.")
                enviar_tecla(handle, ord('A'))  # Simula tecla 'A'
                time.sleep(1)  # Esperar por pantalla negra
                print("Volviendo a Fase 1.")

        except Exception as e:
            print(f"Error durante la ejecución: {e}")

# Ejecutar el bot
if __name__ == "__main__":
    ventana_juego = "FINAL FANTASY V"  # Título de la ventana del juego
    try:
        bot_farmeo(ventana_juego)
    except KeyboardInterrupt:
        print("Bot detenido.")
    except Exception as e:
        print(f"Error: {e}")
