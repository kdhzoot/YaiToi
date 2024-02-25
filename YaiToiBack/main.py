from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import io
import base64
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
from model import get_model
from fastapi.middleware.cors import CORSMiddleware
from torchvision.ops import nms

app = FastAPI()

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 모델 로드 (사전 훈련된 모델 또는 자체 훈련된 모델의 경로를 제공하세요)
model = get_model()  # 정의된 model load

# 저장된 state_dict 로드
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))

# 모델을 평가 모드로 설정 (추론을 위한 경우)
model.eval()


# 이미지 데이터를 위한 Pydantic 모델 정의
class ImageData(BaseModel):
    image: str


# 이미지를 모델 입력 형태로 변환하는 함수
def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')

    my_transforms = transforms.Compose([
        # 이미지 크기 조정이 필요하다면 유지, 그렇지 않으면 제거
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
    ])

    image = my_transforms(image).unsqueeze(0)
    return image


@app.post("/predict/image")
async def predict_api(item: ImageData):
    # Base64 인코딩된 이미지 데이터를 디코딩
    image_data = item.image.split(",")[1]  # Base64 부분만 추출
    image_bytes = base64.b64decode(image_data)
    image = transform_image(image_bytes=image_bytes)

    # 모델 예측 수행
    with torch.no_grad():
        # print(model.roi_heads.score_thresh)
        model.roi_heads.score_thresh = 0
        predictions = model(image)

    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']  # 신뢰도 점수

    # NMS 적용
    nms_indices = nms(boxes, scores, iou_threshold=0.1)

    nms_scores = scores[nms_indices]
    nms_boxes = boxes[nms_indices]
    nms_labels = labels[nms_indices]

    # 예측 결과를 JSON 형태로 변환
    output = [
        {
            "labels": nms_labels.cpu().numpy().tolist(),
            "boxes": nms_boxes.cpu().numpy().tolist(),
            "scores": nms_scores.cpu().numpy().tolist(),
        }
    ]

    return output
