import flet as ft
from analisis import AnalisisEnfermedadCebada

def main(page: ft.Page):
    page.adaptive = True
    page.title = "Calculadora de Severidad de Hojas"
    analisis = AnalisisEnfermedadCebada()
    page.add(analisis.build())

ft.app(target=main)
