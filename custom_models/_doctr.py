from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# model = ocr_predictor(pretrained=True, detect_language=True)
# # PDF
# # doc = DocumentFile.from_pdf(r"C:\Users\annch\OneDrive\Desktop\projects\private\achikashua.pdf")
# # Images
# multi_img_doc = DocumentFile.from_images(["p1.jpg"])
# # Analyze
# result = model(multi_img_doc)
# print(result)

from doctr.datasets import FUNSD
train_set = FUNSD(train=True, download=True)
img, target = train_set[0]