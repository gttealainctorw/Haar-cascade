
# Real-Time Vision — Haar Cascade

Lightweight real-time face detection powered by classical computer vision.

Built with Python and OpenCV, this project implements the **Haar Cascade** detection algorithm originally proposed by Paul Viola and Michael Jones.

No deep learning.
No heavy models.
Just fast, efficient detection.

---

## What it does

* 🎥 Captures live webcam feed
* 🧠 Detects faces using `haarcascade_frontalface_default.xml`
* 🟩 Draws real-time bounding boxes
* 📊 Displays live detection metrics
* ⚙️ Allows dynamic sensitivity tuning
* 📸 Saves screenshots on demand

---

## Controls

| Key | Action                            |
| --- | --------------------------------- |
| Q   | Exit application                  |
| S   | Save screenshot                   |
| +   | Increase sensitivity              |
| -   | Decrease sensitivity              |
| N   | Increase precision (minNeighbors) |
| M   | Decrease precision                |

---

## Tech Stack

```
Python
OpenCV
NumPy
```

---

## Why Haar Cascade?

While modern systems rely on deep neural networks, Haar Cascade remains:

* Extremely fast
* Lightweight
* CPU-friendly
* Ideal for embedded or academic projects

A classic algorithm that still delivers.
