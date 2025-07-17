import asyncio
from viam.module.module import Module
try:
    from models.yolov8n_pose import Yolov8nPose
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.yolov8n_pose import Yolov8nPose


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
