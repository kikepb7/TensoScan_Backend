from app.utils.image_processing import ImageProcessor # Importamos la clase

# Creamos una instancia de la clase
processor = ImageProcessor()

# Llamamos a los métodos pasando sus argumentos
image = processor.load_image("C:/Users/garci/OneDrive/Documentos/_IABD/_proyectos/tensoscan/TensoScan_Images/tensiometro.jpg")  # Cargar imagen
processed_image = processor.process_image(image, threshold=150)  # Procesar imagen
display_area = processor.extract_display_area(image, (10, 10, 2875, 4100))  # Extraer zona de interés
digit_positions = processor.detect_digit_positions(display_area)  # Detectar posiciones de los dígitos