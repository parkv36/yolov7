# YOLOv7 for EdgeFirst

This is the Au-Zone branch of YOLOv7 for EdgeFirst processing.  We are focused on embedded deployments such as the Maivin AI Vision platform.  The tweaks and workflows aim to improve accuracy when running quantized models on accelerators such as the i.MX 8M Plus NPU and the Hailo accelerator.

Currently an update to the yolov7-tiny backbone is provided named yolov7-edgefirst.  The backbone has been updated to use ReLU6 activations which demonstrate superior accuracy when the model is quantized.  The workflows for DeepViewRT and HailoRT are documented our support site at https://support.deepviewml.com.

# Ready to Use Models

Ready to use models are available under releases.  These models have been pre-trained on the COCO dataset and optimized for DeepViewRT and HailoRT and are ready to use.

**DEMOS COMING SOON**

