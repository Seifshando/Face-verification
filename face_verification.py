from deepface import DeepFace

img1_path = "./data_set/Eng_Ahmed.jpeg"
img2_path = "./data_set/Eng_Ahmed2.jpeg"

try:
    result = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name="VGG-Face",
        distance_metric="cosine",
        detector_backend="retinaface"
    )

    print(f"هل هما نفس الشخص؟ {result['verified']}")
    print(f"نسبة المسافة (Distance): {result['distance']}")

except Exception as e:
    print(f"حدث خطأ: {e}")