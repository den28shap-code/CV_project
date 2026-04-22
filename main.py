import json
from reader import analyze_instrument

IMAGE_PATH = r"images/IMG_5868.JPG"

result = analyze_instrument(IMAGE_PATH)
print(json.dumps(result, ensure_ascii=False, indent=2))

