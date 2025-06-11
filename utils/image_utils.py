import base64
import io
from PIL import Image

def encode_image_to_base64(image: Image.Image) -> str:
    """
    PIL Image 객체를 받아 PNG 형식으로 바이트 버퍼에 저장한 뒤
    Base64로 인코딩하여 data URI 문자열로 반환합니다.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_str}"
