# Georgian Language OCR
This project aims to develop an Optical Character Recognition (OCR) system specifically designed for the Georgian language. The OCR system is capable of extracting text from images or scanned documents written in the Georgian script and converting it into editable and searchable text.

## Features
- Accurate and robust text extraction from Georgian language documents.
- Support for various image formats such as JPEG, PNG, and TIFF.
- Preprocessing techniques for image enhancement and noise reduction.
- Character segmentation to isolate individual characters for recognition.
- Training and fine-tuning of machine learning models for improved accuracy.
- Post-processing methods for language-specific text correction and normalization.
- Integration with external applications or libraries for further processing or analysis.

<br></br>
- Huggingface Dataset does not working
- <del> load_model returns utf 8 encoding error </del>
- Should build better convolution neural net
https://data-flair.training/blogs/handwritten-character-recognition-neural-network/
https://pylessons.com/handwriting-recognition-pytorch
- try how works Resnet and vgg models
- improve augmentation
https://towardsdatascience.com/effective-data-augmentation-for-ocr-8013080aa9fa
- improve text split system
https://github.com/KadenMc/PreprocessingHTR/blob/main/doc/0img.jpg
https://stackoverflow.com/questions/56698714/how-to-segment-characters-and-words-from-images-into-contours
- build web app (idea is still unclear)
- try to use kubernets, ci/cd, docker or similar mlops tools