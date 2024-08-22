# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 2024

@author: Alberto Lencina, Luisa Cabezas y Claude IA

Contacto: alencina@azul.faa.unicen.edu.ar

Objetivo: 
Identificar las manchas de enfermedades fúngicas en hojas de cebada que fueron cortadas de la planta, pegadas sobre un papel blanco y fotografiadas.

Procedimiento de análisis:
* Se segmenta el papel (fondo) de las hojas (muestra) seleccionando un nivel digital en el canal azul (B) de la imagen.
* Se segmenta la muestra separando la parte verde (sana) de la enferma mediante la selección de un umbral para el índice (G-R)/(G+R) siendo G el canal verde de la imagen y R el rojo.
* Se calcula la severidad de la enfermedad como el cociente entre el recuento de píxeles enfermos respecto del total de píxeles de la muestra.

Algoritmo:
1 Abrir una imagen seleccionándola desde el cuadro de diálogo de windows.

2 Segmentación del fondo:
    2.a Mostrar la imagen.
    2.b Junto con la imagen incluir un deslizador que seleccione un umbral (Ub) para el canal azul (B) de la imagen para segmentar el fondo.
    2.c Los píxeles de la imagen cuyo valor en el canal azul sea mayor que Ub se mostrarán dinámicamente en una capa azul superpuesta sobre la imagen mostrada en el paso 2.a, a medida que se modifica el valor en el deslizador.
    2.d Debajo del deslizador incluir un botón que diga “Ok” para finalizar la segmentación del fondo, una vez que el usuario esté conforme con el valor Ub seleccionado.
    2.e Aquellos píxeles que satisfacen la condición 2.c serán modificados con el valor NaN en los tres canales: R, G y B.

3 Segmentación de la muestra:
    3.a Mostrar la imagen resultante del paso 2.e.
    3.b Junto con la imagen incluir un deslizador que seleccione un umbral (Ui) para el índice (G-R)/(G+R) siendo G el valor del píxel en el canal verde de la imagen y R en el rojo.
    3.c Los píxeles de la imagen cuyo valor de (G-R)/(G+R) sea mayor que Ui (enfermos) se mostrarán dinámicamente en una capa roja superpuesta sobre la imagen mostrada en el paso 2.a, mientras que aquellos valores menores o iguales que Ui (samos) se mostrarán dinámicamente en una capa verde. Todo esto a medida que se modifica el valor en el deslizador.
    3.d Dinámicamente, en el borde superior izquierdo de la imágen se muestra el valor de la severidad, que se calcula como el cociente entre el recuento de píxeles enfermos respecto del total de píxeles distintos de NaN de la imagen mostrada en el paso 3.a.
    3.e Debajo del deslizador incluir un botón que diga “Ok” para finalizar la segmentación de la muestra, una vez que el usuario esté conforme con el valor Ui seleccionado.

4 Imprimir en pantalla en nombre del archivo de la imagen seleccionada en el paso 1 y el valor de severidad calculada en el paso 3.d.


Changelog:
2024-08-22: Se colocaron los valores iniciales del modelo entrenado->Paremeters.xlsx
            Se comentaron los siguientes pasos en def mostrar_imagen(self, img, canvas):
                # Sugerir umbrales iniciales
                # Establecer y actualizar umbral B
                # Actualizar el valor del Umbral Indice antes de desactivarlo
                # Asegurarse de que el Umbral Indice esté desactivado
                Se tradujo al inglés el texto visible

            


"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from sklearn.cluster import MiniBatchKMeans

class AnalisisEnfermedadCebada:
    def __init__(self, master):
        self.master = master
        self.master.title("Leaf Severity Calculator")
        self.master.state('zoomed')  # Maximizar la ventana
        
        self.img = None
        self.img_original = None
        self.mascara_hojas = None
        self.nombre_archivo = ""
        self.ui_inicial = -0.030792934
        self.ub_inicial = 180
        
        # Frame principal
        main_frame = tk.Frame(master)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Frame superior
        self.frame_superior = tk.Frame(main_frame)
        self.frame_superior.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_cargar = tk.Button(self.frame_superior, text="Load Image", command=self.cargar_imagen)
        self.btn_cargar.pack(side=tk.LEFT, padx=(0, 1020))
        
        self.lbl_severidad = tk.Label(self.frame_superior, text="Severity: ", font=("Arial", 16, "bold"))
        self.lbl_severidad.pack(side=tk.LEFT)
        
        # Frame de imágenes
        self.frame_imagenes = tk.Frame(main_frame)
        self.frame_imagenes.pack(expand=True, fill=tk.BOTH, pady=10)
        
        # Fijar el tamaño de los canvas una sola vez
        canvas_width = self.master.winfo_width() // 2
        canvas_height = int(self.master.winfo_height() * 0.7)
        
        self.canvas_original = tk.Canvas(self.frame_imagenes, bg='lightgray')
        self.canvas_original.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.canvas_procesada = tk.Canvas(self.frame_imagenes, bg='lightgray')
        self.canvas_procesada.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Frame de controles
        self.frame_controles = tk.Frame(main_frame)
        self.frame_controles.pack(fill=tk.X, pady=(10, 0))
        
        self.crear_control_deslizador("Background", 0, 255, self.actualizar_umbral_b, resolution=1, valor_inicial=self.ub_inicial)
        self.crear_control_deslizador("Disease", -1, 1, self.actualizar_umbral_indice, resolution=0.001, valor_inicial=self.ui_inicial)
        
        
    def crear_control_deslizador(self, label, from_, to, command, resolution=1, valor_inicial=None):
        frame = tk.Frame(self.frame_controles)
        frame.pack(fill=tk.X, pady=5)
    
        lbl = tk.Label(frame, text=label, width=15, anchor='e')
        lbl.pack(side=tk.LEFT, padx=(0, 10))
    
        slider = tk.Scale(frame, from_=from_, to=to, resolution=resolution, orient=tk.HORIZONTAL, length=500, command=command)
        if valor_inicial is not None:
            slider.set(valor_inicial)
            slider.pack(side=tk.LEFT, expand=True, fill=tk.X)
    
        entry = tk.Entry(frame, width=10)
        entry.pack(side=tk.LEFT, padx=(10, 0))
        entry.bind('<Return>', lambda e, s=slider: s.set(entry.get()))
    
        btn = tk.Button(frame, text="OK", command=lambda: self.finalizar_segmentacion(label))
        btn.pack(side=tk.LEFT, padx=(10, 0))
    
        setattr(self, f"slider_{label.replace(' ', '_').lower()}", slider)
        setattr(self, f"entry_{label.replace(' ', '_').lower()}", entry)
        setattr(self, f"btn_{label.replace(' ', '_').lower()}", btn)
    
        # Actualizar el entry con el valor inicial
        if valor_inicial is not None:
            entry.delete(0, tk.END)
            entry.insert(0, str(valor_inicial))

    def sugerir_umbrales(self, imagen_path):
        # Cargar la imagen
        img = cv2.imread(imagen_path)
        
        # Separar canales (BGR en OpenCV)
        b, g, r = cv2.split(img)
        
        # 1. Segmentación del fondo (umbral B)
        ub, _ = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # 2. Segmentación de la muestra (umbral del índice NGRDI)
        # Calcular NGRDI
        ngrdi = np.divide(g.astype(float) - r.astype(float), g.astype(float) + r.astype(float), out=np.zeros_like(g, dtype=float), where=(g+r)!=0)
        
        # Crear máscara de hojas
        mascara_hojas = b <= ub
        
        # Aplicar K-means solo a los píxeles de las hojas
        ngrdi_hojas = ngrdi[mascara_hojas].reshape(-1, 1)
        kmeans = MiniBatchKMeans(n_clusters=4, random_state=0, init="k-means++", n_init=100, max_iter=300).fit(ngrdi_hojas)
        
        # El umbral será el punto medio entre los dos centroides #menores
        clu_cent = kmeans.cluster_centers_[:,0]
        clu_cent.sort()
        ui = np.mean(clu_cent[:2]) #np.mean(kmeans.cluster_centers_)
        
        return ub, ui

    def cargar_imagen(self):
        path = filedialog.askopenfilename()
        if path:
            self.nombre_archivo = path.split("/")[-1]
            self.img_original = cv2.imread(path)
            self.img = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
            
            # Sugerir umbrales iniciales
            #self.ub_inicial, self.ui_inicial = self.sugerir_umbrales(path)
            
            self.mostrar_imagen(self.img_original, self.canvas_original)
            
            # Establecer y actualizar umbral B
            #self.slider_umbral_b.set(self.ub_inicial)
            #self.actualizar_umbral_b(self.ub_inicial)
            
            # Actualizar el valor del Umbral Indice antes de desactivarlo
            #self.slider_umbral_indice.set(self.ui_inicial)
            #self.entry_umbral_indice.delete(0, tk.END)
            #self.entry_umbral_indice.insert(0, str(self.ui_inicial))
            
            # Asegurarse de que el Umbral Indice esté desactivado
            #self.slider_umbral_indice.config(state='disabled')
            #self.entry_umbral_indice.config(state='disabled')
            #self.btn_umbral_indice.config(state='disabled')
            
    def mostrar_imagen(self, img, canvas):
        img = cv2.medianBlur(img, 5)
        
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        if canvas_width / canvas_height > aspect_ratio:
            new_height = canvas_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = canvas_width
            new_height = int(new_width / aspect_ratio)
        
        img_resized = cv2.resize(img, (new_width, new_height))
        
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_resized))
        
        # Calcula la posición para centrar la imagen
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        # Borra el contenido anterior del canvas
        canvas.delete("all")
        
        # Crea la imagen centrada en el canvas
        canvas.create_image(x, y, image=photo, anchor=tk.NW)
        canvas.image = photo

    def actualizar_umbral_b(self, ub):
        if self.img is not None:
            ub = int(float(ub))
            self.entry_umbral_b.delete(0, tk.END)
            self.entry_umbral_b.insert(0, str(ub))
            b, _, _ = cv2.split(self.img_original)
            mascara = b > ub
            img_copia = self.img.copy()
            img_copia[mascara] = [255, 0, 0]  # Azul para visualización
            self.mostrar_imagen(img_copia, self.canvas_procesada)
            
            # No actualizar la máscara de hojas ni llamar a actualizar_umbral_i aquí
            
    def actualizar_umbral_indice(self, ui):
        if self.mascara_hojas is not None and self.slider_disease['state'] == 'normal':
            ui = float(ui)
            self.entry_disease.delete(0, tk.END)
            self.entry_disease.insert(0, str(ui))
            b, g, r = cv2.split(self.img_original)
            indice = np.divide(g.astype(float) - r.astype(float), g.astype(float) + r.astype(float), out=np.zeros_like(g, dtype=float), where=(g+r)!=0)
            mascara_enferma = (indice <= ui) & self.mascara_hojas
            mascara_sana = (indice > ui) & self.mascara_hojas
            
            img_resultado = self.img.copy()
            img_resultado[mascara_enferma] = [0, 0, 255]  # Rojo para enfermo
            img_resultado[mascara_sana] = [0, 255, 0]  # Verde para sano
            
            severidad = np.sum(mascara_enferma) / np.sum(self.mascara_hojas)
            self.lbl_severidad.config(text=f"Severity: {severidad:.1%}")
            
            self.mostrar_imagen(img_resultado, self.canvas_procesada)
            self.severidad = severidad
            
    def finalizar_segmentacion(self, label):
        if label == "Background":
            if self.img is not None:
                ub = self.slider_background.get()
                b, g, r = cv2.split(self.img_original)
                self.mascara_hojas = b <= ub
                img_segmentada = self.img_original.copy()
                img_segmentada[~self.mascara_hojas] = [0, 0, 0]  # Negro para el fondo
                self.img = cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2RGB)
                self.mostrar_imagen(self.img, self.canvas_procesada)
                
                # Activar el control de Umbral Indice
                self.slider_disease.config(state='normal')
                self.entry_disease.config(state='normal')
                self.btn_disease.config(state='normal')
                
                # Inicializar la visualización del Umbral Indice
                self.actualizar_umbral_indice(self.ui_inicial)
        elif label == "Disease":
            print(f"Analysis completed for image: {self.nombre_archivo}")
            print(f"Severity: {self.severidad:.1%}")
            #self.master.destroy()


root = tk.Tk()
app = AnalisisEnfermedadCebada(root)
root.mainloop()