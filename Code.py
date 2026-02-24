import cv2
import numpy as np

print(" INICIANDO DETECCIÓN EN TIEMPO REAL CON HAAR CASCADE")
print(" Activando cámara web...")

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # 0 = cámara por defecto

# Verificar que la cámara se abrió correctamente
if not cap.isOpened():
    print(" Error: No se pudo acceder a la cámara")
    print("   Posibles soluciones:")
    print("   1. Verifica que la cámara esté conectada")
    print("   2. Asegúrate de que no esté siendo usada por otra aplicación")
    print("   3. Prueba con cv2.VideoCapture(1) si tienes múltiples cámaras")
    exit()

print(" Cámara web activada correctamente")

# Cargar el clasificador Haar Cascade para cuerpo completo
cascade_path = 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

if cascade.empty():
    print(f" Error: No se pudo cargar el clasificador {cascade_path}")
    print("   Asegúrate de que el archivo XML esté en la carpeta correcta")
    cap.release()
    exit()

print(" Clasificador Haar Cascade cargado correctamente")

# Configuración de parámetros para la detección
scale_factor = 1.1
min_neighbors = 5
min_size = (50, 50)  # Tamaño mínimo para cuerpos completos

print("\n CONFIGURACIÓN ACTUAL:")
print(f"   • Scale Factor: {scale_factor}")
print(f"   • Min Neighbors: {min_neighbors}")
print(f"   • Min Size: {min_size}")

print("\n🎮 CONTROLES:")
print("   • Presiona 'Q' para salir")
print("   • Presiona 'S' para tomar una captura de pantalla")
print("   • Presiona '+' para aumentar sensibilidad")
print("   • Presiona '-' para disminuir sensibilidad")

# Contadores y estadísticas
frame_count = 0
detection_count = 0
total_detections = 0

print("\n Iniciando detección en tiempo real...")

while True:
    # Leer frame de la cámara
    ret, frame = cap.read()
    
    if not ret:
        print(" Error: No se pudo leer el frame de la cámara")
        break
    
    frame_count += 1
    
    # Voltear el frame horizontalmente (espejo) para mejor experiencia
    frame = cv2.flip(frame, 1)
    
    # Convertir a escala de grises (requerido por Haar Cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar detección de cuerpos completos
    bodies = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Dibujar rectángulos alrededor de las detecciones
    detection_count = len(bodies)
    total_detections += detection_count
    
    for (x, y, w, h) in bodies:
        # Rectángulo verde para cuerpos detectados
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Etiqueta con información
        cv2.putText(frame, 'Cara', (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Punto central
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
        
        # Información de tamaño
        cv2.putText(frame, f'{w}x{h}', (x, y + h + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Agregar información en pantalla
    cv2.putText(frame, f'Detecciones: {detection_count}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(frame, f'Scale: {scale_factor}', (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, f'Neighbors: {min_neighbors}', (10, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, f'Frames: {frame_count}', (10, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Agregar leyenda de controles
    cv2.putText(frame, "Q: Salir  S: Captura  +/-: Sensibilidad", 
               (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
               0.4, (255, 255, 255), 1)
    
    # Mostrar el frame
    cv2.imshow('Detector de Cuerpos - Haar Cascade en Tiempo Real', frame)
    
    # Manejar entrada del teclado
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        print("\n Cerrando aplicación...")
        break
    elif key == ord('s') or key == ord('S'):
        # Tomar captura de pantalla
        filename = f"captura_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f" Captura guardada como: {filename}")
    elif key == ord('+'):
        # Aumentar sensibilidad (disminuir scale factor)
        scale_factor = max(1.05, scale_factor - 0.05)
        print(f" Aumentando sensibilidad - Scale Factor: {scale_factor:.2f}")
    elif key == ord('-'):
        # Disminuir sensibilidad (aumentar scale factor)
        scale_factor = min(2.0, scale_factor + 0.05)
        print(f" Disminuyendo sensibilidad - Scale Factor: {scale_factor:.2f}")
    elif key == ord('n'):
        # Aumentar minNeighbors (menos detecciones, más precisas)
        min_neighbors = min(20, min_neighbors + 1)
        print(f" Aumentando Min Neighbors: {min_neighbors}")
    elif key == ord('m'):
        # Disminuir minNeighbors (más detecciones, menos precisas)
        min_neighbors = max(1, min_neighbors - 1)
        print(f" Disminuyendo Min Neighbors: {min_neighbors}")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

# Mostrar estadísticas finales
print("\n ESTADÍSTICAS FINALES:")
print(f"   • Total de frames procesados: {frame_count}")
print(f"   • Total de detecciones: {total_detections}")
if frame_count > 0:
    print(f"   • Promedio de detecciones por frame: {total_detections/frame_count:.2f}")
print(" Aplicación finalizada correctamente")

