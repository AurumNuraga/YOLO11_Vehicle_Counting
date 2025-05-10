from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import shutil
import os
import csv
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="uploads"), name="static")

model = YOLO('yolo11l.pt')
UPLOAD_FOLDER = "uploads"
CSV_FOLDER = "csv_data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CSV_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    output_video_path, csv_path = process_video(file_path)
    return {
        "message": "Video processed", 
        "video_url": f"{os.path.basename(output_video_path)}",
        "csv_url": f"{os.path.basename(csv_path)}"
    }

@app.get("/download-csv/{filename}")
async def download_csv(filename: str):
    csv_path = os.path.join(CSV_FOLDER, filename)
    if os.path.exists(csv_path):
        return FileResponse(
            path=csv_path, 
            media_type="text/csv", 
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    return {"error": "File not found"}

def process_video(video_path):
    class_list = model.names
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #avc1 diganti mp4v
    out_path = "uploads/output_video.mp4"
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    #line_y_red = 1000
    #line_y_red = 700
    line_y_red = 425
    class_counts = defaultdict(int)
    crossed_ids = set()
    vehicle_data = []  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, classes=[1, 2, 3, 5, 7])

        if results[0].boxes.data is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [0] * len(boxes)
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu()

            #cv2.line(frame, (40, line_y_red), (1650, line_y_red), (0, 0, 255), 3)
            #cv2.line(frame, (1020, line_y_red), (1600, line_y_red), (0, 0, 255), 3)
            cv2.line(frame, (500, line_y_red), (1450, line_y_red), (0, 0, 255), 3)

            for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                class_name = class_list[class_idx]

                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if line_y_red - 10 <= cy <= line_y_red + 10 and 500 <= cx <= 1450 and track_id not in crossed_ids:
                    crossed_ids.add(track_id)
                    class_counts[class_name] += 1
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    vehicle_data.append({
                        "timestamp": timestamp,
                        "vehicle_id": track_id,
                        "vehicle_type": class_name,
                        "confidence": round(float(conf), 2)
                    })

            y_offset = 30
            for class_name, count in class_counts.items():
                cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 30

        out.write(frame)

    cap.release()
    out.release()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"vehicle_counts_{timestamp}.csv"
    csv_path = os.path.join(CSV_FOLDER, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Vehicle Type', 'Count'])
        for class_name, count in class_counts.items():
            writer.writerow([class_name, count])
        
        writer.writerow([])  
        
        writer.writerow(['Timestamp', 'Vehicle ID', 'Vehicle Type', 'Confidence'])
        for vehicle in vehicle_data:
            writer.writerow([
                vehicle['timestamp'],
                vehicle['vehicle_id'],
                vehicle['vehicle_type'],
                vehicle['confidence']
            ])

    fixed_output_path = os.path.join(UPLOAD_FOLDER, "output_video_convert.mp4") #video yang akan diputar di UI 
    os.system(f"ffmpeg -i {out_path} -vcodec libx264 -pix_fmt yuv420p {fixed_output_path}") #dikonversi agar bisa pemutaran kompatibel dengan browser
    
    return fixed_output_path, csv_path

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.2", port=3158)