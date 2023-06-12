# from doctr import DocumentFile
from custom_models._doctr import ocr_predictor

model = ocr_predictor(pretrained=True)
# PDF
# doc = DocumentFile.from_pdf("path/to/your/doc.pdf")
# Analyze
# result = model(doc)