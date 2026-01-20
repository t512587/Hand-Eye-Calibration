import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
from pymycobot.elephantrobot import ElephantRobot
import json
import os

# === 自動取得 RealSense Color 相機內參 ===
def get_color_camera_intrinsics(pipeline, config):
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
    pipeline.stop()

    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ])
    dist_coeffs = np.zeros((5, 1))
    return camera_matrix, dist_coeffs

# 建立 ElephantRobot 連線
elephant_client = ElephantRobot("192.168.50.123", 5001)
elephant_client.start_client()
print("ElephantRobot目前座標：", elephant_client.get_coords())

# 設定 RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 自動取得相機內參
camera_matrix, dist_coeffs = get_color_camera_intrinsics(pipeline, config)
pipeline.start(config)

# ArUco 設定
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
marker_length = 0.04  # 單位：公尺

# 紀錄點位的資料結構
recorded_points = []
target_points = 15
current_count = 0

# 創建儲存資料夾
if not os.path.exists('aruco_records'):
    os.makedirs('aruco_records')

print(f"=== ArUco 15點記錄系統 ===")
print(f"目標記錄點數: {target_points}")
print("操作說明:")
print("- 按 's' 鍵記錄當前檢測到的 ArUco 標記")
print("- 按 'r' 鍵重置記錄")
print("- 按 'q' 鍵退出程式")
print("- 按 'v' 鍵查看已記錄的點位")
print("=" * 40)

try:
    while current_count < target_points:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                cv2.aruco.drawDetectedMarkers(color_image, corners)
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                corner = corners[i][0]
                cv2.putText(color_image, f"ID: {ids[i][0]}", 
                           (int(corner[0][0]), int(corner[0][1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                info_text = f"Recorded: {current_count}/{target_points} points"
                cv2.putText(color_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if ids is not None:
                    detection_text = f"Detected {len(ids)} ArUco markers"
                    cv2.putText(color_image, detection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                cv2.imshow('ARUCO Detection - 15 Point Recording System', color_image)
                key = cv2.waitKey(1) & 0xFF


                if key == ord('q'):
                    break
                elif key == ord('s') and ids is not None:
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    robot_coords = elephant_client.get_coords()
                    record_data = {
                        'record_id': current_count + 1,
                        'timestamp': timestamp,
                        'robot_coords': robot_coords,
                        'aruco_markers': []
                    }

                    for i in range(len(ids)):
                        tvec_mm = tvecs[i][0] * 1000
                        marker_data = {
                            'id': int(ids[i][0]),
                            'translation_mm': tvec_mm.tolist(),
                            'rotation_vector': rvecs[i][0].tolist(),
                            'corners': corners[i].tolist()
                        }
                        record_data['aruco_markers'].append(marker_data)

                    recorded_points.append(record_data)
                    current_count += 1

                    image_filename = f"aruco_records/point_{current_count:02d}_{timestamp}.png"
                    cv2.imwrite(image_filename, color_image)

                    print(f"\n=== 記錄點位 {current_count}/{target_points} ===")
                    print(f"時間戳記: {timestamp}")
                    print(f"影像檔案: {image_filename}")
                    print(f"機械臂座標: {robot_coords}")
                    print(f"檢測到 {len(ids)} 個 ArUco 標記:")

                    for i, marker in enumerate(record_data['aruco_markers']):
                        print(f"  標記 {i+1} - ID: {marker['id']}")
                        print(f"    位置 (X Y Z) mm: {marker['translation_mm']}")
                        print(f"    旋轉向量: {marker['rotation_vector']}")

                    print(f"剩餘記錄點數: {target_points - current_count}")
                    print("-" * 40)

                elif key == ord('r'):
                    recorded_points = []
                    current_count = 0
                    print("\n=== 記錄已重置 ===")

                elif key == ord('v'):
                    print(f"\n=== 已記錄點位總覽 ({current_count}/{target_points}) ===")
                    for i, point in enumerate(recorded_points):
                        print(f"點位 {i+1}: {point['timestamp']} - {len(point['aruco_markers'])} 個標記")
                    print("-" * 40)

            if current_count >= target_points:
                print(f"\n🎉 已完成所有 {target_points} 個點位的記錄！")
                json_filename = f"aruco_records/complete_record_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(recorded_points, f, ensure_ascii=False, indent=2)

                print(f"完整記錄已儲存至: {json_filename}")

                total_markers = sum(len(point['aruco_markers']) for point in recorded_points)
                print(f"\n=== 記錄統計 ===")
                print(f"總記錄點數: {len(recorded_points)}")
                print(f"總 ArUco 標記數: {total_markers}")
                print(f"平均每點標記數: {total_markers/len(recorded_points):.1f}")

                print(f"\n=== 詳細記錄 ===")
                for i, point in enumerate(recorded_points):
                    print(f"\n點位 {i+1} ({point['timestamp']}):")
                    print(f"  機械臂座標: {point['robot_coords']}")
                    print(f"  ArUco 標記數: {len(point['aruco_markers'])}")
                    for j, marker in enumerate(point['aruco_markers']):
                        print(f"    標記 {j+1} - ID: {marker['id']}, 位置: {marker['translation_mm']}")

finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n程式結束")