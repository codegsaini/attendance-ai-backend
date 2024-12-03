import asyncio
import cv2
import numpy as np
import socketio
import uvicorn
from fastapi import FastAPI, Response
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

sio=socketio.AsyncServer(cors_allowed_origins=[],async_mode="asgi")
socket_app=socketio.ASGIApp(sio)

@app.get("/")
def read_root():
    return Response("AttendanceAi server is running")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

activeSources=set(())

async def get_camera_stream(source):
    if isinstance(source,str):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)

    if not cap.isOpened():
        await sio.emit('video_frame', {'success': False, 'error': 0, 'data': "Camera not found", 'source': source})
        return

    activeSources.add(source)
    while source in activeSources:
        ret, frame = await asyncio.get_running_loop().run_in_executor(None, cap.read)
        if not ret:
            await sio.emit('video_frame', {'success': False, 'error': 1, 'data': "No video", 'source': source})
            activeSources.remove(source)
            continue

        # Check if the frame is valid (not all zeros, etc.)
        if frame is None or frame.size == 0 or np.all(frame == 0):
            await sio.emit('video_frame', {'success': False, 'error': 2, 'data': "Corrupt frame", 'source': source})
            continue

        faces = detect_bounding_box(
            frame
        )
        # Encode the frame in JPEG format
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            await sio.emit('video_frame', {'success': False, 'error': 3, 'data': "Frame encoding failed", 'source': source})
            continue

        # frame_bytes = base64.b64encode(buffer).decode('utf-8')

        # Send the frame via WebSocket
        await sio.emit('video_frame', {'success': True, 'data': buffer.tobytes(), 'source': source})
        await asyncio.sleep(0.1)

    cap.release()
    return


@sio.on("start_stream")
async def get_camera_feed(sid, camera_type, source):
    if camera_type == 'usb':
        source = int(source)  # Convert to integer for USB camera index

    if not source in activeSources:
        await get_camera_stream(source)


@sio.on("stop_stream")
def stop_camera_feed(sid, camera_type, source):
    if camera_type == 'usb':
        source = int(source)  # Convert to integer for USB camera index

    if source in activeSources:
        activeSources.remove(source)
    return True

@sio.on("stop_all_stream")
def stop_all_camera_feed(sid):
    activeSources.clear()
    return True

app.mount("/", socket_app)


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=7000, lifespan="on", reload=True)