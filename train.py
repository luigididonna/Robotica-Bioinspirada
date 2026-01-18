import os
import gymnasium as gym
from stable_baselines3 import PPO

# Configuración de los hiperparámetros
TIMESTEPS = 1_000_000
MODEL_NAME = "swimmer_ppo_model"
LOG_DIR = "./logs/"

def train_model():
    """
    Ejecuta el entrenamiento del agente PPO en el entorno Swimmer-v5.
    
    El modelo se entrena durante un número definido de pasos temporales
    y se guarda en el disco al finalizar. Los registros de tensorboard
    se almacenan en el directorio especificado.
    """
    
    # Creación del directorio de logs si no existe
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"[INFO] Iniciando el entrenamiento del agente ({TIMESTEPS} pasos)...")

    # Inicialización del entorno
    env = gym.make("Swimmer-v5", render_mode="rgb_array")

    # Configuración del modelo PPO con política MLP
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64
    )

    # Ejecución del aprendizaje
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    
    # Guardado del modelo
    model.save(MODEL_NAME)
    print(f"[INFO] Entrenamiento completado. Modelo guardado como: {MODEL_NAME}.zip")

if __name__ == "__main__":
    train_model()
