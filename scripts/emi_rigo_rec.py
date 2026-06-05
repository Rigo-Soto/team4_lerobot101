from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

import logging

# ── Configuración ──────────────────────────────────────────────────────────────
NUM_EPISODES = 50
FPS = 30
EPISODE_TIME_SEC = 27

# CAMBIAR entre sesiones de grabación:
#   Sesión 1 (pilar rojo):   "Grab red pillar and place in box"
#   Sesión 2 (pilar blanco): "Grab white pillar and place in box"
TASK_DESCRIPTION = "Grab white pillar and place in box"

PORT_LEADER = "/dev/ttyACM0"
PORT_FOLLOW = "/dev/ttyACM1"
CAM_INDEX = 2
# ──────────────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # ── Robot y teleoperador (IDs originales para usar calibración existente) ──
    robot_config = SO101FollowerConfig(
        id="follower_dlr",
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=CAM_INDEX,
                width=640,
                height=480,
                fps=FPS,
            )
        },
        port=PORT_FOLLOW,
    )

    teleop_config = SO101LeaderConfig(
        id="lider_de_la_rosa",
        port=PORT_LEADER,
    )

    robot = SO101Follower(robot_config)
    teleop = SO101Leader(teleop_config)

    # ── Dataset ────────────────────────────────────────────────────────────────
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id="emiliano-ng/so101-pilares001",
        root="../dataset/so101-pilares-blanco/angel01",
	 fps=FPS,
	features=dataset_features,
	robot_type=robot.name,
	use_videos=True,
	image_writer_threads=4,
    )

    # ── Inicialización ─────────────────────────────────────────────────────────
    _, events = init_keyboard_listener()
    init_rerun(session_name="recording")

    robot.connect()
    teleop.connect()

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    # ── Loop de grabación ──────────────────────────────────────────────────────
    print(f"\n  Tarea: {TASK_DESCRIPTION}")
    print(f"  Total de episodios: {NUM_EPISODES}")
    print(f"  Duración por episodio: {EPISODE_TIME_SEC}s")
    print(f"  Controles: ENTER = guardar | r = descartar y repetir\n")

    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:

        # Esperar confirmación manual antes de cada episodio
        input(
            f"\n  ── Episodio {episode_idx + 1}/{NUM_EPISODES} ──\n"
            f"  Posiciona el pilar y la caja.\n"
            f"  Presiona ENTER cuando estés listo para grabar..."
        )

        log_say(f"Grabando episodio {episode_idx + 1} de {NUM_EPISODES}")

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

        # ── Guardar o repetir ──────────────────────────────────────────────────
        decision = input(
            f"\n  ¿Cómo salió el episodio?\n"
            f"  ENTER → guardar y continuar\n"
            f"  r     → descartar y repetir\n"
            f"  > "
        ).strip().lower()

        if decision == "r":
            dataset.clear_episode_buffer()
            log_say("Episodio descartado")
            print("  ✗ Episodio descartado. Se repetirá.")
            continue
        # ──────────────────────────────────────────────────────────────────────

        dataset.save_episode()
        log_say(f"Episodio {episode_idx + 1} guardado")
        print(f"  ✓ Episodio {episode_idx + 1}/{NUM_EPISODES} guardado.")
        episode_idx += 1

    # ── Fin ────────────────────────────────────────────────────────────────────
    log_say("Grabación terminada")
    print(f"\n  Grabación terminada. {episode_idx} episodios guardados.")
    print(f"  Dataset: emiliano-ng/so101-pilares\n")

    robot.disconnect()
    teleop.disconnect()


if __name__ == "__main__":
    main()
