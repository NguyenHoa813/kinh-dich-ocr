
# from src.image_loader import load_image
# from pathlib import Path

# img = load_image(r'F:\kinh-dich-ocr\page1.jpg')
# print('Load thành công')
# print('Kích thước:', img.size)
# print('Mode:', img.mode)

# from paddleocr import PaddleOCR
# import os

# os.environ["PADDLE_PDX_LOCAL_FONT_FILE_PATH"] = r"F:\kinh-dich-ocr\fonts\times.ttf"   

# ocr = PaddleOCR(
#     lang="vi",                    
#     ocr_version="PP-OCRv5",
#     use_textline_orientation=True,
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
# )

# img_path = "test.png"
# result = ocr.predict(img_path)

# for res in result:
#     res.print()
#     res.save_to_img("output")
#     res.save_to_json("output")

