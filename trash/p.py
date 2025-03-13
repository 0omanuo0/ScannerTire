import asyncio
import cv2
import numpy as np
from aiortc import VideoStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import TcpSocketSignaling

class VideoTrack(VideoStreamTrack):
    async def recv(self):
        # Simular un frame de ruido
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        # Crear y devolver el frame como video
        return VideoFrame.from_ndarray(frame, format="bgr24")

async def run():
    pc = RTCPeerConnection()
    pc.addTrack(VideoTrack())

    signaling = TcpSocketSignaling("localhost", 9999)
    await signaling.connect()
    offer = await signaling.receive()
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await signaling.send(pc.localDescription)

if __name__ == "__main__":
    asyncio.run(run())
