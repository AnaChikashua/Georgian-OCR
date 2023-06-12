from mmocr.mmocr.apis import MMOCRInferencer
infer = MMOCRInferencer(rec='svtr-small')
result = infer('test.JPG', save_vis=True, return_vis=True)
print(result['predictions'])
