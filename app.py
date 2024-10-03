import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from PIL import Image as PILImage
import io
import numpy as np
import cv2

class AnalisisEnfermedadCebada(toga.App):
    def startup(self):
        self.main_window = toga.MainWindow(title="Análisis de Enfermedad en Cebada", size=(600, 800))
        
        self.cargar_imagen_button = toga.Button("Cargar Imagen", on_press=self.cargar_imagen)
        self.guardar_imagen_button = toga.Button("Guardar Imagen Procesada", on_press=self.guardar_imagen)
        
        self.image_view = toga.ImageView(style=Pack(width=500, height=500))
        
        self.slider_umbral_b = toga.Slider(on_change=self.actualizar_umbral_b, min=0, max=255, value=180)
        self.label_umbral_b = toga.Label('Umbral B: 180', style=Pack(padding=(0, 5)))
        
        # Reintroducir el slider para la clasificación
        self.slider_umbral_i = toga.Slider(on_change=self.actualizar_umbral_indice, min=-1, max=1, value=0)
        self.label_umbral_i = toga.Label('Umbral Índice: 0.000', style=Pack(padding=(0, 5)))
        
        self.label_severidad = toga.Label('Severidad: 0%', style=Pack(padding=(0, 5)))
        
        self.mascara_hojas = None
        self.nombre_archivo = ""
        self.severidad = 0
        self.imagen_procesada = None
        
        box = toga.Box(
            children=[self.cargar_imagen_button, self.guardar_imagen_button, 
                      self.slider_umbral_b, self.label_umbral_b, 
                      self.slider_umbral_i, self.label_umbral_i,
                      self.label_severidad, self.image_view],
            style=Pack(direction=COLUMN, padding=10)
        )
        
        self.main_window.content = box
        self.main_window.show()

    async def cargar_imagen(self, widget):
        try:
            file_path = await self.main_window.dialog(
                toga.OpenFileDialog(title="Seleccionar una imagen", multiple_select=False, file_types=['png', 'jpg', 'jpeg', 'bmp'])
            )
            
            if file_path:
                self.nombre_archivo = str(file_path)
                self.mostrar_imagen(file_path)
            else:
                print("No se seleccionó ningún archivo.")
        except Exception as e:
            self.main_window.info_dialog("Error", f"No se pudo cargar la imagen: {str(e)}")

    def mostrar_imagen(self, file_path):
        try:
            self.pil_image = PILImage.open(file_path)
            self.pil_image.thumbnail((500, 500))  # Redimensionar
            self.original_array = np.array(self.pil_image)
            
            self.actualizar_umbral_b(self.slider_umbral_b)
            
        except Exception as e:
            self.main_window.info_dialog("Error", f"No se pudo mostrar la imagen: {str(e)}")

    def actualizar_umbral_b(self, widget):
        ub = int(widget.value)
        self.label_umbral_b.text = f'Umbral B: {ub}'

        # Segmentación usando canal B
        b = self.original_array[:,:,2]  # Canal B en RGB
        self.mascara_hojas = b <= ub

        # Actualizar la imagen con la nueva segmentación
        self.actualizar_umbral_indice(self.slider_umbral_i)

    def actualizar_umbral_indice(self, widget):
        if self.mascara_hojas is not None:
            ui = float(widget.value)
            self.label_umbral_i.text = f'Umbral Índice: {ui:.3f}'

            # Calcular el índice NGRDI
            r, g, _ = cv2.split(self.original_array)
            ngrdi = np.divide(g.astype(float) - r.astype(float), g.astype(float) + r.astype(float), 
                              out=np.zeros_like(g, dtype=float), where=(g + r) != 0)

            # Clasificar hojas sanas y enfermas
            mascara_enferma = (ngrdi <= ui) & self.mascara_hojas
            mascara_sana = (ngrdi > ui) & self.mascara_hojas

            # Crear imagen resultado
            img_resultado = self.original_array.copy()
            img_resultado[mascara_enferma] = [255, 0, 0]  # Rojo para enfermo
            img_resultado[mascara_sana] = [0, 255, 0]  # Verde para sano
            img_resultado[~self.mascara_hojas] = [0, 0, 0]  # Negro para el fondo

            # Calcular severidad
            area_total_hojas = np.sum(self.mascara_hojas)
            area_enferma = np.sum(mascara_enferma)
            self.severidad = (area_enferma / area_total_hojas) * 100 if area_total_hojas > 0 else 0
            self.label_severidad.text = f'Severidad: {self.severidad:.2f}%'

            self.mostrar_imagen_procesada(img_resultado)

    def mostrar_imagen_procesada(self, imagen):
        imagen_pil = PILImage.fromarray(imagen)
        with io.BytesIO() as f:
            imagen_pil.save(f, format='PNG')
            self.image_view.image = toga.Image(f.getvalue())
        self.imagen_procesada = imagen_pil

    async def guardar_imagen(self, widget):
        if self.imagen_procesada:
            save_path = await self.main_window.dialog(toga.SaveFileDialog(title="Guardar imagen procesada", suggested_filename="imagen_procesada.png"))
            if save_path:
                self.imagen_procesada.save(save_path)
                self.main_window.info_dialog("Éxito", "Imagen guardada correctamente")

def main():
    return AnalisisEnfermedadCebada(formal_name="Análisis de Enfermedad en Cebada", app_id="org.example.leafseverity")

if __name__ == '__main__':
    main().main_loop()