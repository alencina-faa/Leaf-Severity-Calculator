import cv2
import numpy as np
import flet as ft
from PIL import Image
from io import BytesIO
import base64

class AnalisisEnfermedadCebada:
    def __init__(self):
        self.img = None
        self.img_original = None
        self.mascara_hojas = np.ones((1, 1), dtype=bool)  # Inicializa como máscara de hojas válida
        self.nombre_archivo = ""
        self.ui_inicial = -0.030792934
        self.ub_inicial = 180
        
        # Crear controles
        self.btn_cargar = ft.ElevatedButton(text="Cargar Imagen", on_click=self.cargar_imagen)
        self.file_picker = ft.FilePicker(on_result=self.procesar_imagen)
        self.slider_background = ft.Slider(min=0, max=255, value=self.ub_inicial, label="Umbral de Fondo", on_change=self.actualizar_umbral_b)
        self.slider_disease = ft.Slider(min=-1, max=1, value=self.ui_inicial, label="Umbral de Enfermedad", on_change=self.actualizar_umbral_indice)
        self.img_original_view = ft.Image(width=400, height=400)
        self.img_procesada_view = ft.Image(width=400, height=400)
        self.lbl_severidad = ft.Text(value="Severidad: 0%", size=16, weight=ft.FontWeight.BOLD)

    def build(self):
        return ft.Column(
            controls=[
                self.btn_cargar,
                self.file_picker,  # Asegúrate de que el FilePicker esté aquí
                self.slider_background,
                self.slider_disease,
                self.lbl_severidad,
                ft.Row(controls=[self.img_original_view, self.img_procesada_view]),
            ],
            expand=True
        )

    def cargar_imagen(self, e):
        # Abre el FilePicker cuando se hace clic en el botón
        self.file_picker.pick_files(allow_multiple=False)

    def procesar_imagen(self, e: ft.FilePickerResultEvent):
        if e.files:
            file_path = e.files[0].path
            self.img_original = cv2.imread(file_path)
            if self.img_original is None:
                print("Error: La imagen no se pudo cargar.")
                return
            self.img = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)

            # Mostrar imagen original
            self.mostrar_imagen(self.img_original, self.img_original_view)

            # Inicializar sliders
            self.slider_background.value = self.ub_inicial
            self.actualizar_umbral_b(None)
            self.actualizar_umbral_indice(None)

    def mostrar_imagen(self, img, image_view):
        img_pil = Image.fromarray(img)
        buffer = BytesIO()
        img_pil.save(buffer, format="PNG")
        img_data = buffer.getvalue()
        image_view.src_base64 = base64.b64encode(img_data).decode("utf-8")  # Asigna src_base64
        image_view.update()

    def actualizar_umbral_b(self, e):
        if self.img is not None:
            ub = int(self.slider_background.value)
            b, _, _ = cv2.split(self.img_original)
            mascara = b > ub
            img_copia = self.img.copy()
            img_copia[mascara] = [255, 0, 0]  # Azul para visualización
            self.mostrar_imagen(img_copia, self.img_procesada_view)

    def actualizar_umbral_indice(self, e):
        if self.mascara_hojas is not None and self.img is not None:
            ui = float(self.slider_disease.value)
            _, g, r = cv2.split(self.img_original)
            indice = np.divide(g.astype(float) - r.astype(float), g.astype(float) + r.astype(float), out=np.zeros_like(g, dtype=float), where=(g + r) != 0)
            mascara_enferma = (indice <= ui) & self.mascara_hojas
            mascara_sana = (indice > ui) & self.mascara_hojas

            img_resultado = self.img.copy()
            img_resultado[mascara_enferma] = [0, 0, 255]  # Rojo para enfermo
            img_resultado[mascara_sana] = [0, 255, 0]  # Verde para sano

            severidad = np.sum(mascara_enferma) / np.sum(self.mascara_hojas)
            self.lbl_severidad.value = f"Severidad: {severidad:.1%}"
            self.lbl_severidad.update()

            self.mostrar_imagen(img_resultado, self.img_procesada_view)
