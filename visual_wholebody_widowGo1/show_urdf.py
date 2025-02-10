import pybullet as p
import pybullet_data
import time

# 连接到 PyBullet 仿真环境
physics_client = p.connect(p.GUI)  # 或者使用 p.DIRECT 如果不需要GUI

# 设置搜索路径以便找到标准模型
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载平面和 URDF 模型
plane_id = p.loadURDF("plane.urdf")  # 加载平面
robot_id = p.loadURDF(
    # "/home/yuhao/Documents/USDA_Gripper/urdf/USDA_Gripper.urdf",
    # "/home/yuhao/ManiSkill-WidowX250S/USDA_Gripper.urdf",
    # "/home/yuhao/ManiSkill-WidowX250S/wx250s_USDA.urdf",
    "/home/yuhao/pybullet_ur5_gripper/robots/urdf/ur5e_USDA.urdf",
    [0, 0, 1],  # 初始位置
    useFixedBase=True  # 固定底座
)

# 设置重力
p.setGravity(0, 0, -9.81)

# 创建滑动条控制关节
num_joints = p.getNumJoints(robot_id)  # 获取机器人关节数量
joint_sliders = []  # 用于保存滑动条的参数ID
debug_lines = []  # 用于保存关节坐标轴线的ID

for joint_index in range(num_joints):
    joint_info = p.getJointInfo(robot_id, joint_index)
    joint_name = joint_info[1].decode("utf-8")
    joint_lower_limit = joint_info[8]  # 关节下限
    joint_upper_limit = joint_info[9]  # 关节上限

    # 如果关节是转动或滑动关节，创建滑动条
    if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        slider_id = p.addUserDebugParameter(
            joint_name, 
            joint_lower_limit, 
            joint_upper_limit, 
            0.0  # 初始值为0
        )
        joint_sliders.append((joint_index, slider_id))

    # 创建初始关节坐标轴的线条ID
    debug_lines.append({
        "x": p.addUserDebugLine([0, 0, 0], [0, 0, 0], [1, 0, 0], lineWidth=2),
        "y": p.addUserDebugLine([0, 0, 0], [0, 0, 0], [0, 1, 0], lineWidth=2),
        "z": p.addUserDebugLine([0, 0, 0], [0, 0, 0], [0, 0, 1], lineWidth=2)
    })

# 模拟并使用滑动条控制关节
while True:
    # 读取每个滑动条的值，并更新机器人关节的角度
    for i, (joint_index, slider_id) in enumerate(joint_sliders):
        target_value = p.readUserDebugParameter(slider_id)  # 获取滑动条值
        p.setJointMotorControl2(
            robot_id,
            joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_value
        )

        # 更新关节坐标轴的线条
        joint_state = p.getLinkState(robot_id, joint_index)
        joint_position = joint_state[4]  # 世界坐标中的关节位置
        x_end = [joint_position[0] + 0.1, joint_position[1], joint_position[2]]
        y_end = [joint_position[0], joint_position[1] + 0.1, joint_position[2]]
        z_end = [joint_position[0], joint_position[1], joint_position[2] + 0.1]

        # 更新坐标轴线
        p.addUserDebugLine(joint_position, x_end, [1, 0, 0], replaceItemUniqueId=debug_lines[i]["x"])
        p.addUserDebugLine(joint_position, y_end, [0, 1, 0], replaceItemUniqueId=debug_lines[i]["y"])
        p.addUserDebugLine(joint_position, z_end, [0, 0, 1], replaceItemUniqueId=debug_lines[i]["z"])

    p.stepSimulation()  # 单步仿真
    time.sleep(1.0 / 240.0)  # 仿真步长为240 Hz

# 断开连接
p.disconnect()