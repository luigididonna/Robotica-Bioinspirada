import os
import gymnasium as gym
from stable_baselines3 import PPO
import imageio
import numpy as np

# Configuración del nombre del modelo a cargar
MODEL_PATH = "swimmer_ppo_model"
OUTPUT_VIDEO = "simulation_result.mp4"

def setup_rendering():
    """
    Configura el backend de renderizado. 
    Necesario para entornos sin monitor (headless) como servidores o Colab.
    """
    if "COLAB_GPU" in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

def evaluate_model():
    """
    Carga el modelo entrenado y genera un vídeo de la simulación.
    La evaluación se realiza en modo determinista.
    """
    
    # Comprobación de la existencia del archivo
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        raise FileNotFoundError(f"El modelo {MODEL_PATH}.zip no se encuentra. Ejecute train.py primero.")

    print("[INFO] Cargando modelo y generando simulación...")

    # Carga del modelo
    model = PPO.load(MODEL_PATH)

    # Creación del entorno de evaluación
    env = gym.make("Swimmer-v5", render_mode="rgb_array")
    obs, _ = env.reset()

    frames = []
    total_reward = 0.0
    done = False
    
    # Ciclo de inferencia (máximo 1000 pasos)
    for _ in range(1000):
        # Predicción determinista de la acción
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        # Captura del frame
        frames.append(env.render())

        if terminated or truncated:
            break

    env.close()

    # Guardado del vídeo
    imageio.mimsave(OUTPUT_VIDEO, frames, fps=30)
    
    print(f"[INFO] Simulación finalizada.")
    print(f"[RESULT] Recompensa Total Acumulada: {total_reward:.2f}")
    print(f"[OUTPUT] Vídeo guardado en: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    setup_rendering()
    evaluate_model()
