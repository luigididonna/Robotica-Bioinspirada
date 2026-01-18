# Robotica-Bioinspirada
Estrategias de locomoción ondulatoria en fluidos viscosos mediante Aprendizaje Profundo por Refuerzo (PPO). Utilizando el entorno MuJoCo Swimmer-v5, este proyecto demuestra la capacidad de un agente artificial para aprender patrones de nado eficientes en condiciones de Tabula Rasa, sin modelado cinemático previo.

# Validación Experimental de Robótica Bioinspirada mediante PPO

Este repositorio contiene el código fuente utilizado para la validación experimental de estrategias de locomoción bioinspirada en el entorno `Swimmer-v5` (MuJoCo), utilizando el algoritmo *Proximal Policy Optimization* (PPO).

## Descripción del Proyecto
El objetivo del experimento es demostrar la capacidad de un agente de Aprendizaje por Refuerzo para desarrollar estrategias de locomoción ondulatoria eficientes en un fluido viscoso simulado, sin conocimiento previo de la dinámica del sistema (*Tabula Rasa*).

## Estructura del Repositorio
* `train.py`: Script principal para el entrenamiento del agente.
* `evaluate.py`: Script para la evaluación del modelo y generación de vídeo.
* `requirements.txt`: Lista de dependencias necesarias.

## Instalación y Requisitos

El código ha sido desarrollado en Python 3.10. Se requiere el motor físico MuJoCo.

### 1. Clonar el repositorio
```bash
git clone [https://github.com/TUO_USERNAME/swimmer-rl-project.git](https://github.com/TUO_USERNAME/swimmer-rl-project.git)
cd swimmer-rl-project
