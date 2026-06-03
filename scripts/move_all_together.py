from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration
import json
import time

def main():
    calib_path = "./callibration/lider_de_la_rosa.json"
    with open(calib_path) as f:
        calib_data = json.load(f)

    bus = FeetechMotorsBus(
        port="/dev/ttyACM0",
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
    calibration = {name: MotorCalibration(**vals) for name, vals in calib_data.items()}
    bus.write_calibration(calibration)
    bus.enable_torque()

    START = {
        "shoulder_pan":  47.15,
        "shoulder_lift": 49.99,
        "elbow_flex":    49.43,
        "wrist_flex":    79.89,
        "wrist_roll":    49.08,
        "gripper":       95.84,
    }

    END = {
        "shoulder_pan":  49.43,
        "shoulder_lift": 77.46,
        "elbow_flex":    26.47,
        "wrist_flex":    9.04,
        "wrist_roll":    49.82,
        "gripper":       95.84,
    }

    def move_all(targets, steps=1000, delay=0.01):
        current = {}
        for name in targets:
            current[name] = bus.read("Present_Position", name)

        print("  Moviendo todos a la vez...")
        for step in range(1, steps + 1):
            t = step / steps
            for name, target_pct in targets.items():
                interp = current[name] + (target_pct - current[name]) * t
                try:
                    bus.write("Goal_Position", name, interp, num_retry=3)
                except Exception as e:
                    print(f"  ⚠️  Error en {name}: {e}")
            time.sleep(delay)

    print("Moviendo todos los motores a la vez.\n")
    input("Presiona ENTER para comenzar...\n")

    try:
        move_all(END, steps=100, delay=0.1)
        print("\n✅ Listo.")

    except KeyboardInterrupt:
        print("\n⛔ Detenido.")

    finally:
        bus.disable_torque()
        bus.disconnect()
        print("Desconectado.")

if __name__ == '__main__':
    main()
