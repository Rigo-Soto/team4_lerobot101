# In your data collection script (e.g. record_episodes.py)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.so_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so_leader.so100_leader import SO100Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

from yolo_extract import YOLODetector, extract_feature_vector

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "LEGO deconstructor"

YOLO_WEIGHTS = "backup/yolov4-tiny-custom_final.weights"
YOLO_CONFIG = "custom_cfg/yolov4-tiny-custom.cfg"

YOLO_CONF_THRESHOLD = 0.45
YOLO_NMS_THRESHOLD  = 0.4
YOLO_CLASS_NAMES = ["base", "column"]               # must match your training order
YOLO_CAMERA_KEY  = "front"                          # which camera to run YOLO on
YOLO_FEATURE_DIM = 16                               # fixed vector length — do not change
 
# Set True to draw bounding boxes on the camera feed (useful during recording)
YOLO_DEBUG_OVERLAY = True

class YOLOObservationProcessor:
    """
    Wraps the standard LeRobot robot_observation_processor and appends a YOLO
    feature vector to every observation dict before it is handed to record_loop.
 
    Usage:
        base_processor = make_default_processors()[2]   # robot_observation_processor
        yolo_processor = YOLOObservationProcessor(base_processor, detector)
        # Pass yolo_processor as robot_observation_processor to record_loop
    """
 
    def __init__(
        self,
        base_processor: Any,
        detector: YOLODetector,
        debug_overlay: bool = False,
    ):
        self.base_processor  = base_processor
        self.detector        = detector
        # LeRobot stores images under this key pattern in the observation dict
        self.image_obs_key   = f"observation.images"
        self.debug_overlay   = debug_overlay
        self._last_detections: dict = {}   # expose for external logging
 
    def __call__(self, observation: dict) -> dict:
        """
        1. Run the standard processor (handles image resizing, normalisation, etc.)
        2. Grab the processed camera frame and run YOLO inference.
        3. Inject `observation.yolo_features` into the output dict.
        """
        # Step 1 — standard processing
        processed = self.base_processor(observation)
 
        # Step 2 — YOLO inference on the designated camera frame
        frame = processed.get(self.image_obs_key)
        if frame is not None:
            # frame may be a torch.Tensor (C, H, W) or np.ndarray (H, W, C)
            if hasattr(frame, "numpy"):
                # Convert CHW float tensor → HWC uint8 for OpenCV
                np_frame = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                np_frame = frame
 
            detections = self.detector.detect(np_frame)
            self._last_detections = detections
 
            if self.debug_overlay:
                overlay = self.detector.draw_overlay(np_frame, detections)
                # Display in a separate window; press 'q' in the main rerun view to close
                cv2.imshow("YOLO detections", overlay)
                cv2.waitKey(1)
        else:
            logging.warning(
                f"[YOLO] Key '{self.image_obs_key}' not found in observation. "
                "Check that YOLO_CAMERA_KEY matches your camera config."
            )
            detections = {}
            self._last_detections = {}
 
        # Step 3 — append feature vector
        feat_vec = extract_yolo_feature_vector(detections, np_frame.shape if frame is not None else (480, 640))
        processed["observation.yolo_features"] = feat_vec
 
        return processed
 
 
def make_yolo_dataset_feature() -> dict:
    """
    Returns a LeRobot-compatible feature descriptor for the YOLO vector.
 
    LeRobot uses the same format as HuggingFace datasets under the hood.
    A 1-D float32 sequence is described with dtype, shape, and optional names.
    """
    return {
        "observation.yolo_features": {
            "dtype": "float32",
            "shape": (YOLO_FEATURE_DIM,),
            "names": [
                # base (7)
                "base_cx", "base_cy", "base_bw", "base_bh",
                "base_conf", "base_area", "base_valid",
                # column (7)
                "col_cx", "col_cy", "col_bw", "col_bh",
                "col_conf", "col_area", "col_valid",
                # relative pose (2)
                "delta_x", "delta_y",
            ],
        }
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 
    detector = YOLODetector("weights/tiny_yolov4.weights", "cfg/tiny_yolov4.cfg",use_gpu=False)

    # Create robot configuration
    robot_config = SO101FollowerConfig(
        id="follower_dlr",
        cameras={
            "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS) 
        },
        port="/dev/tty.usbmodem58760434471",
    )

    teleop_config = SO101LeaderConfig(
        id="lider_de_la_rosa",
        port="/dev/tty.usbmodem585A0077581",
    )

    # Initialize the robot and teleoperator
    robot = SO100Follower(robot_config)
    teleop = SO100Leader(teleop_config)

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    yolo_features = make_yolo_dataset_feature()
    dataset_features = {**action_features, **obs_features}

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id="legoDataset",
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Initialize the keyboard listener and rerun visualization
    _, events = init_keyboard_listener()
    init_rerun(session_name="recording")

    # Connect the robot and teleoperator
    robot.connect()
    teleop.connect()

    # Create the required processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Wrap the standard observation processor with the YOLO layer
    yolo_observation_processor = YOLOObservationProcessor(
        base_processor=robot_observation_processor,
        detector=detector,
        debug_overlay=YOLO_DEBUG_OVERLAY,
    )

    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=yolo_observation_processor,
            teleop=teleop,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

        # Reset the environment if not stopping or re-recording
        if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        episode_idx += 1

    # Clean up
    log_say("Stop recording")
    robot.disconnect()
    teleop.disconnect()

if __name__ == '__main__':
    main()
