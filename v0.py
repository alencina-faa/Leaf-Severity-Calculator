import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import cv2
import numpy as np
from PIL import Image
import io
import asyncio
from pathlib import Path

class LeafSeverityCalculator(toga.App):
    def startup(self):
        self.img = None
        self.img_original = None
        self.mascara_hojas = None
        self.nombre_archivo = ""
        self.ui_inicial = -0.030792934
        self.ub_inicial = 180
        self.severidad = 0

        # Main box
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=20))

        # Top box
        top_box = toga.Box(style=Pack(direction=ROW, padding=(0, 0, 10, 0)))
        self.btn_cargar = toga.Button('Cargar Imagen', on_press=self.cargar_imagen)
        self.lbl_severidad = toga.Label('Severidad: ', style=Pack(font_size=16, font_weight='bold'))
        top_box.add(self.btn_cargar)
        top_box.add(self.lbl_severidad)

        # Image box
        self.image_box = toga.Box(style=Pack(direction=ROW, padding=(0, 10)))
        self.img_view_original = toga.ImageView(style=Pack(flex=1, width=300, height=300))
        self.img_view_procesada = toga.ImageView(style=Pack(flex=1, width=300, height=300))
        self.image_box.add(self.img_view_original)
        self.image_box.add(self.img_view_procesada)

        # Control box
        control_box = toga.Box(style=Pack(direction=COLUMN, padding=(10, 0)))
        self.slider_background = self.crear_control_deslizador("Fondo", 0, 255, self.actualizar_umbral_b, self.ub_inicial)
        self.slider_disease = self.crear_control_deslizador("Enfermedad", -1, 1, self.actualizar_umbral_indice, self.ui_inicial)
        control_box.add(self.slider_background)
        control_box.add(self.slider_disease)

        main_box.add(top_box)
        main_box.add(self.image_box)
        main_box.add(control_box)

        self.main_window = toga.MainWindow(title="Calculadora de Severidad de Hojas")
        self.main_window.content = main_box
        self.main_window.show()

    def crear_control_deslizador(self, label, min_val, max_val, on_change, valor_inicial):
        box = toga.Box(style=Pack(direction=ROW, padding=5))
        lbl = toga.Label(label, style=Pack(width=100))
        slider = toga.Slider(range=(min_val, max_val), value=valor_inicial, on_change=on_change)
        entry = toga.TextInput(value=str(valor_inicial), style=Pack(width=100))
        btn = toga.Button("OK", on_press=lambda widget: self.finalizar_segmentacion(label))
        
        box.add(lbl)
        box.add(slider)
        box.add(entry)
        box.add(btn)
        
        setattr(self, f"entry_{label.lower()}", entry)
        return box

    async def cargar_imagen(self, widget):
        try:
            file_dialog_result = await self.main_window.open_file_dialog(
                title="Seleccionar una imagen",
                multiselect=False,
                file_types=['jpg', 'png']
            )
            if file_dialog_result:
                file_path = file_dialog_result[0] if isinstance(file_dialog_result, (list, tuple)) else file_dialog_result
                file_path = Path(file_path)
                
                self.nombre_archivo = file_path.name
                self.img_original = cv2.imread(str(file_path))
                if self.img_original is None:
                    raise ValueError(f"No se pudo cargar la imagen: {file_path}")
                self.img = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
                
                self.mostrar_imagen(self.img_original, self.img_view_original)
                
                self.slider_background.value = self.ub_inicial
                self.actualizar_umbral_b(None, self.ub_inicial)
                self.finalizar_segmentacion("Fondo")
                
                self.slider_disease.value = self.ui_inicial
                self.actualizar_umbral_indice(None, self.ui_inicial)
                self.finalizar_segmentacion("Enfermedad")
            else:
                print("No se seleccionó ningún archivo")
        except Exception as e:
            await self.main_window.info_dialog('Error', str(e))

    def mostrar_imagen(self, img, img_view):
        img = cv2.medianBlur(img, 5)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        # Resize the image to fit the ImageView
        pil_image.thumbnail((300, 300))
        
        with io.BytesIO() as buffer:
            pil_image.save(buffer, format="PNG")
            img_data = buffer.getvalue()
        
        img_view.image = toga.Image(data=img_data)

    def actualizar_umbral_b(self, widget, value):
        if self.img is not None:
            ub = int(float(value))
            self.entry_fondo.value = str(ub)
            b, _, _ = cv2.split(self.img_original)
            mascara = b > ub
            img_copia = self.img.copy()
            img_copia[mascara] = [255, 0, 0]  # Azul para visualización
            self.mostrar_imagen(img_copia, self.img_view_procesada)

    def actualizar_umbral_indice(self, widget, value):
        if self.mascara_hojas is not None:
            ui = float(value)
            self.entry_enfermedad.value = str(ui)
            _, g, r = cv2.split(self.img_original)
            indice = np.divide(g.astype(float) - r.astype(float), g.astype(float) + r.astype(float), out=np.zeros_like(g, dtype=float), where=(g+r)!=0)
            mascara_enferma = (indice <= ui) & self.mascara_hojas
            mascara_sana = (indice > ui) & self.mascara_hojas
            
            img_resultado = self.img.copy()
            img_resultado[mascara_enferma] = [0, 0, 255]  # Rojo para enfermo
            img_resultado[mascara_sana] = [0, 255, 0]  # Verde para sano
            
            self.severidad = np.sum(mascara_enferma) / np.sum(self.mascara_hojas)
            self.lbl_severidad.text = f"Severidad: {self.severidad:.1%}"
            
            self.mostrar_imagen(img_resultado, self.img_view_procesada)

    def finalizar_segmentacion(self, label):
        if label == "Fondo":
            if self.img is not None:
                ub = self.slider_background.value
                b, g, r = cv2.split(self.img_original)
                self.mascara_hojas = b <= ub
                img_segmentada = self.img_original.copy()
                img_segmentada[~self.mascara_hojas] = [0, 0, 0]  # Negro para el fondo
                self.img = cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2RGB)
                self.mostrar_imagen(self.img, self.img_view_procesada)
                
                # Inicializar la visualización del Umbral Indice
                self.actualizar_umbral_indice(None, self.ui_inicial)
        elif label == "Enfermedad":
            print(f"Análisis completado para la imagen: {self.nombre_archivo}")
            print(f"Severidad: {self.severidad:.1%}")

def main():
    return LeafSeverityCalculator('Calculadora de Severidad de Hojas', 'org.example.leafseveritycalculator')

if __name__ == '__main__':
    main().main_loop()