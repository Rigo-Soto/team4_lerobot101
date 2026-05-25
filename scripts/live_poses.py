from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration
import json
import time

# Cargar calibración guardada
calib_path = "/Users/rigo/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/team4_leader_arm.json"
with open(calib_path) as f:
    calib_data = json.load(f)

bus = FeetechMotorsBus(
    port="/dev/tty.usbmodem5B141111761",
    motors={
        "shoulder_pan":   Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
        "shoulder_lift":  Motor(id=2, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
        "elbow_flex":     Motor(id=3, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
        "wrist_flex":     Motor(id=4, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
        "wrist_roll":     Motor(id=5, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
        "gripper":        Motor(id=6, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    }
)

bus.connect()

# Registrar calibración
calibration = {
    name: MotorCalibration(**vals) for name, vals in calib_data.items()
}
bus.write_calibration(calibration)

print("Leader conectado. Mueve el brazo y observa los valores:\n")

try:
    while True:
        positions = {}
        for name in bus.motors:
            positions[name] = bus.read("Present_Position", name)
        print(positions, end="\r")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nDetenido.")
    bus.disconnect()