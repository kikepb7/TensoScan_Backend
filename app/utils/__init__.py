from image_processing import *
from label_scrip_automatized import *
imageP = ImageProcessor()

#Cargar la imagen del tensiómetro
image = cv2.imread(r'C:\Users\garci\OneDrive\Documentos\_IABD\_proyectos\tensoscan\TensoScan_Images\200.jpg')
image2 = Image.open(r'C:\Users\garci\OneDrive\Documentos\_IABD\_proyectos\tensoscan\TensoScan_Images\200.jpg')
image2 = np.array(image2)

print(type(image2))

#Detectar el display
display_area = imageP.extract_display_area(image)
display_area = cv2.resize(display_area, (1109, 1431))
print(display_area.shape)

#Mostrar el área detectada (si existe)
if display_area is not None:
    cv2.imshow("Display Area", display_area)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se detectó el display.")
digit_positions = imageP.crop_digit_areas(display_area)

autolabel()