from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

dataset_rojo = LeRobotDataset(
    repo_id="emiliano-ng/so101-pilares",
    root="/home/angel/.cache/huggingface/lerobot/emiliano-ng/so101-pilares",
)

dataset_blanco = LeRobotDataset(
    repo_id="emiliano-ng/so101-pilares-blanco",
    root="/tmp/so101-pilares-blanco",
)

merged = merge_datasets(
    datasets=[dataset_rojo, dataset_blanco],
    output_repo_id="emiliano-ng/so101-pilares-merged",
    output_dir="/home/angel/.cache/huggingface/lerobot/emiliano-ng/so101-pilares-merged",
)

print(f"Total episodios: {merged.num_episodes}")
print(f"Tareas: {merged.meta.tasks})")

