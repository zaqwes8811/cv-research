
You don't "export to .ckpt" with Ultralytics; rather, .pt (PyTorch checkpoint) files


```
wget -O yolov8n.hef https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8n.hef

wget -O test_image.jpg https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg

wget -O test_image.jpg https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01440764_tench.JPEG

wget -O test_image.jpg https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg

More objects
wget -O test_image.jpg http://images.cocodataset.org/train2017/000000252038.jpg

wget -O test_image.jpg  http://images.cocodataset.org/train2017/000000581921.jpg

scp infer/*.py yolov8n.hef *.jpg root@192.168.1.100:


/opt/hailo-libs/ld-linux-aarch64.so.1 --library-path /opt/hailo-libs /usr/bin/python3 yolov8n_example.py
```

wget -O wine_glass_ubc.jpg https://democart.phas.ubc.ca/lib/exe/fetch.php?media=demonstrations:3_oscillations_and_waves:wine_glass_resonance:img_2057.jpg



convert i.webp output.jpg

convert output.jpg -resize 640x640! output_640x640.jpg