# Usamos la imagen base de Python
FROM python:3.9-slim

# Instalar dependencias del sistema para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar todo el proyecto al contenedor
COPY . .

# Instalar las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Configurar PYTHONPATH para que Python reconozca la carpeta 'app'
ENV PYTHONPATH=/app

# Exponer el puerto 8000
EXPOSE 8000

# Comando para arrancar la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
