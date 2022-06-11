
import numpy as np
import onnx
import onnxruntime as rt
import cv2


if __name__ == "__main__":
    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
    
    #create input data
    input_data = cv2.imread("../43.jpg")
    input_data = cv2.resize(input_data, (1024, 512))
    input_data = input_data[None, :, :, ::-1].astype(np.float32).copy()
    #create runtime session
    sess = rt.InferenceSession("bisenetv2_a2d2_512x1024_withpre.onnx", 
                               providers=[
                                #    'TensorrtExecutionProvider', 
                                   'CUDAExecutionProvider', 
                                   'CPUExecutionProvider'])
    # get output name
    input_name = sess.get_inputs()[0].name
    print("input name", input_name)
    output_name= sess.get_outputs()[0].name
    print("output name", output_name)
    output_shape = sess.get_outputs()[0].shape
    print("output shape", output_shape)
    #forward model
    res = sess.run([output_name], {input_name: input_data})
    
    out = np.array(res)[0, 0]
    print(out.shape)

    pred = palette[out]
    pred = cv2.resize(pred, (512, 512))
    cv2.imshow('pred', pred)
    cv2.waitKey(0)
