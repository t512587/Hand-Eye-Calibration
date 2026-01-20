import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import json
import os
from pymycobot.elephantrobot import ElephantRobot

# === 棋盤格設定 ===
CHESSBOARD_SIZE = (9, 6)  # 內角點數量 (列, 行)
SQUARE_SIZE = 0.025  # 每個方格的實際大小，單位: 公尺 (25mm)

# === 生成棋盤格 3D 物件點 ===
def create_chessboard_points():
    """
    生成棋盤格的 3D 世界座標點
    原點在棋盤格左上角，Z=0 平面
    """
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp

# === 初始化連線 ===
elephant_client = ElephantRobot("192.168.50.123", 5001)
elephant_client.start_client()
print("ElephantRobot目前座標：", elephant_client.get_coords())

# === 初始化 RealSense 相機 ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# === 取得內參 ===
profile = pipeline.get_active_profile()
video_stream_profile = profile.get_stream(rs.stream.color)
intr = video_stream_profile.as_video_stream_profile().get_intrinsics()

camera_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
])
dist_coeffs = np.array(intr.coeffs[:5]).reshape(5, 1) if len(intr.coeffs) >= 5 else np.zeros((5, 1))

print("\n=== 相機內參 ===")
print(f"焦距: fx={intr.fx:.3f}, fy={intr.fy:.3f}")
print(f"主點: cx={intr.ppx:.3f}, cy={intr.ppy:.3f}")
print(f"畸變係數: {intr.coeffs}")

# === 棋盤格 3D 點 ===
objp = create_chessboard_points()

# === 資料儲存 ===
output_data = []
save_dir = "handeye_records"
os.makedirs(save_dir, exist_ok=True)

print(f"\n=== 手眼標定資料記錄系統 (棋盤格 {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]}) ===")
print(f"方格大小: {SQUARE_SIZE * 1000:.1f} mm")
print("s - 記錄棋盤格 + 機械手初始姿態")
print("m - 記錄該點移動後的姿態")
print("v - 查看已記錄資料")
print("r - 重置資料")
print("q - 離開並儲存")
print("=" * 50)

# 角點優化參數
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

try:
    current_index = 0
    last_rvec = None
    last_tvec = None
    chessboard_found = False

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 檢測棋盤格角點
        ret, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )

        chessboard_found = ret

        if ret:
            # 優化角點位置
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 繪製棋盤格角點
            cv2.drawChessboardCorners(color_image, CHESSBOARD_SIZE, corners_refined, ret)

            # 使用 solvePnP 計算姿態
            success, rvec, tvec = cv2.solvePnP(
                objp, corners_refined, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                last_rvec = rvec
                last_tvec = tvec

                # 繪製座標軸
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, SQUARE_SIZE * 3)

                # 顯示位置資訊
                tvec_mm = tvec.flatten() * 1000
                cv2.putText(color_image, f"Pos: ({tvec_mm[0]:.0f}, {tvec_mm[1]:.0f}, {tvec_mm[2]:.0f}) mm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 顯示棋盤格檢測成功
                cv2.putText(color_image, "Chessboard DETECTED", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(color_image, "Chessboard NOT found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 顯示已記錄數量
        cv2.putText(color_image, f"Recorded: {len(output_data)} poses", (10, color_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Hand-Eye Calibration (Chessboard)", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s') and chessboard_found and last_rvec is not None:
            # 記錄棋盤格姿態和機器人姿態
            rvec_list = last_rvec.flatten().tolist()
            tvec_mm = (last_tvec.flatten() * 1000).tolist()  # 轉換為 mm
            robot_pose = elephant_client.get_coords()

            entry = {
                "marker_id": 0,  # 棋盤格統一使用 ID 0
                "aruco_tvec": tvec_mm,  # 保持相容性，使用相同的欄位名稱
                "aruco_rvec": rvec_list,
                "robot_pose_at_detect": robot_pose,
                "robot_pose_after_move": None,
                "type": "chessboard",
                "chessboard_size": list(CHESSBOARD_SIZE),
                "square_size_mm": SQUARE_SIZE * 1000
            }

            output_data.append(entry)
            current_index = len(output_data) - 1

            print(f"\n✅ 已記錄第 {current_index + 1} 筆")
            print(f"棋盤格位置 (mm): {tvec_mm}")
            print(f"旋轉向量: {rvec_list}")
            print(f"手臂姿態: {robot_pose}")

        elif key == ord('m') and output_data and output_data[current_index]["robot_pose_after_move"] is None:
            moved_pose = elephant_client.get_coords()
            output_data[current_index]["robot_pose_after_move"] = moved_pose
            print(f"🔁 已補上移動後手臂姿態：{moved_pose}")

        elif key == ord('v'):
            print(f"\n📋 已記錄 {len(output_data)} 筆資料：")
            for i, d in enumerate(output_data):
                moved = "✅" if d["robot_pose_after_move"] else "⏳"
                print(f"  第 {i+1} 筆 {moved}")

        elif key == ord('r'):
            output_data = []
            print("🔄 已清空所有記錄")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    if output_data:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"handeye_chessboard_{timestamp}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 已儲存 {len(output_data)} 筆資料至 {filename}")
    print("📌 程式結束")