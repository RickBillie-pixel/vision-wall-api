# 🎯 Vision API v2 - Layer Support & YOLO Detection

## 🚀 Features

- **PDF Layer Detection:** Automatic layer detection via OCG analysis
- **Per-Layer Processing:** YOLO detection per layer separately  
- **Intelligent Fallback:** Standard batching when no layers detected
- **Memory Optimized:** Efficient processing of 7k×9k drawings
- **Coordinate Mapping:** Results in PDF coordinates (mm)

## ⚠️ REQUIRED: YOLO Model

**This API requires a trained YOLO model to work!**

### Quick Setup (For Testing):
```bash
# Download a basic YOLO model (won't detect buildings well):
mkdir models
wget https://github.com/ultralytics/yolov8/releases/download/v8.0.0/yolov8n.pt
mv yolov8n.pt models/ves_model.pt
```

### Production Setup (Recommended):
1. Go to [Roboflow.com](https://roboflow.com)
2. Create project: "Building Drawings"
3. Upload building drawings (7k×9k images)
4. Annotate objects: walls, doors, windows, stairs, dimensions, text
5. Train YOLOv8 model (100-200 epochs)
6. Export as PyTorch (.pt) format
7. Download and rename to: `ves_model.pt`
8. Place in `models/ves_model.pt`

## 🔧 API Endpoints

### `/analyze-pdf-with-layers/`
**Main endpoint with layer support**

```bash
curl -X POST "https://your-api.onrender.com/analyze-pdf-with-layers/" \
  -F "file=@drawing.pdf" \
  -F "tekeningtype=plattegrond" \
  -F "confidence=0.25" \
  -F "dpi=200"
```

**Parameters:**
- `file`: PDF file to analyze
- `tekeningtype`: `plattegrond|doorsnede|gevel|bestektekening|detailtekening`
- `confidence`: YOLO detection threshold (0.1-1.0)
- `dpi`: PDF to PNG conversion quality (150-600)

**Response with Layers:**
```json
{
  "has_layers": true,
  "pages": [{
    "has_layers": true,
    "layers": [{
      "layer_name": "Walls",
      "layer_visible": true,
      "total_detections": 25,
      "detections": [{
        "label": "wall",
        "confidence": 0.94,
        "x": 123.4,
        "y": 456.7,
        "width": 30.5,
        "height": 5.2
      }]
    }]
  }]
}
```

**Response without Layers:**
```json
{
  "has_layers": false,
  "pages": [{
    "has_layers": false,
    "total_detections": 50,
    "detections": [...]
  }]
}
```

### `/health/`
**Health check endpoint**

```bash
curl https://your-api.onrender.com/health/
```

## 📊 Performance

- **With Layers:** 5-15 minutes per layer
- **Without Layers:** 3-8 minutes total
- **Memory Usage:** 1GB - 2GB (YOLO model)
- **Recommended Instance:** Starter+ or higher

## 🐛 Troubleshooting

### "YOLO model not loaded"
```bash
# Check if model exists:
ls -la models/ves_model.pt

# Check health endpoint:
curl https://your-api.onrender.com/health/
```

### High memory usage
```bash
# Upgrade Render instance type
# Or reduce DPI in config (CONFIG["dpi"] = 150)
```

### Layer detection fails
```bash
# Check if PDF actually has layers:
# - Open in Adobe Acrobat
# - Look for "Layers" panel
# - Test with known layered PDF
```

## 🔄 Integration

This API is designed to work with:
- **Master API v5:** Complete orchestration
- **Vector API v2:** Layer-aware vector extraction
- **Filter API v7:** Layer-tagged filtering
- **Scale API v7:** Layer-aware calculations

## 📝 Version

- **API Version:** 2.0.0
- **Layer Support:** ✅ Full
- **Backward Compatible:** ✅ Yes
- **YOLO Framework:** Ultralytics YOLOv8

vision-api-v2/
├── main.py                 # ✅ (Vision API v2 code)
├── requirements.txt        # ✅ (YOLO dependencies)
├── gunicorn_config.py      # ✅ (Server config)
├── runtime.txt             # ✅ (python-3.11.6)
├── render.yaml             # ⚠️ NIEUW - Render configuration
├── .gitignore              # ⚠️ NIEUW - Git ignore rules
├── README.md               # ⚠️ NIEUW - Documentation
└── models/
    ├── .gitkeep            # ⚠️ NIEUW - Keeps directory in git
    └── ves_model.pt        # ⚠️ MOET JE TOEVOEGEN - YOLO model
