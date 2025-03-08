import cv2
import numpy as np
import os


def cargar_imagen(ruta):
    """Carga la imagen desde la ruta especificada."""
    imagen = cv2.imread(ruta)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en {ruta}")
    return imagen

def recortar_area_principal(imagen):
    """Recorta el área donde aparecen los números principales."""
    x, y, w, h = 100, 155, 165, 250  # Coordenadas aproximadas
    recorte = imagen[y:y + h, x:x + w]
    cv2.imshow("Recorte Principal", recorte)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return recorte

def generar_regilla(imagen):
    """Dibuja una rejilla separando cada número."""
    img_copy = imagen.copy()
    alto, ancho, _ = img_copy.shape
    separaciones = [0, ancho // 3, 2 * ancho // 3, ancho]

    for x in separaciones:
        cv2.line(img_copy, (x, 0), (x, alto), (0, 255, 0), 2)

    cv2.imshow("Imagen con Rejilla", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_copy


def recortar_numeros(individualmente, ruta_salida):
    """Recorta cada número por separado y los guarda en la carpeta de origen."""
    alto, ancho, _ = individualmente.shape
    segmentos = [(0, ancho // 3), (ancho // 3, 2 * ancho // 3), (2 * ancho // 3, ancho)]
    nombres = ["numero1.jpg", "numero2.jpg", "numero3.jpg"]
    import cv2
    import numpy as np
    import os

    def cargar_imagen(ruta):
        """Carga la imagen desde la ruta especificada."""
        imagen = cv2.imread(ruta)
        if imagen is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen en {ruta}")
        return imagen

    def recortar_area_principal(imagen):
        """Recorta el área donde aparecen los números principales."""
        x, y, w, h = 80, 50, 250, 200  # Coordenadas aproximadas
        recorte = imagen[y:y + h, x:x + w]
        cv2.imshow("Recorte Principal", recorte)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return recorte

    def generar_regilla(imagen):
        """Dibuja una rejilla separando cada número."""
        img_copy = imagen.copy()
        alto, ancho, _ = img_copy.shape
        separaciones = [0, ancho // 3, 2 * ancho // 3, ancho]

        for x in separaciones:
            cv2.line(img_copy, (x, 0), (x, alto), (0, 255, 0), 2)

        cv2.imshow("Imagen con Rejilla", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img_copy

    def recortar_numeros(individualmente, ruta_salida):
        """Recorta cada número por separado y los guarda en la carpeta de origen."""
        alto, ancho, _ = individualmente.shape
        segmentos = [(0, ancho // 3), (ancho // 3, 2 * ancho // 3), (2 * ancho // 3, ancho)]
        nombres = ["numero1.jpg", "numero2.jpg", "numero3.jpg"]

        for i, (x1, x2) in enumerate(segmentos):
            numero = individualmente[:, x1:x2]
            ruta_guardado = os.path.join(ruta_salida, nombres[i])
            cv2.imwrite(ruta_guardado, numero)
            cv2.imshow(f"Numero {i + 1}", numero)
            cv2.waitKey(500)

        cv2.destroyAllWindows()

    for i, (x1, x2) in enumerate(segmentos):
        numero = individualmente[:, x1:x2]
        ruta_guardado = os.path.join(ruta_salida, nombres[i])
        cv2.imwrite(ruta_guardado, numero)
        cv2.imshow(f"Numero {i + 1}", numero)
        cv2.waitKey(500)

    cv2.destroyAllWindows()
