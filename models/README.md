# Model files

This project does not commit third-party model weights to Git.

Run:

```bash
python scripts/download_models.py
```

It downloads the OpenCV Zoo YuNet face detector and SFace face recognizer into
this directory.

Expected files:

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

The downloaded files are third-party artifacts and remain subject to their
upstream licenses. The repository MIT license does not relicense those model
weights.
