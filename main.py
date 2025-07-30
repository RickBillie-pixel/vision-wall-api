import os
import io
import json
import time
import logging
import asyncio
import gc
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import math

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# PDF & Image Processing
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import numpy as np

# YOLO & ML
from ultralytics import YOLO
import torch

# Memory management
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VES Detection API v2 - Layer Support",
    description="High-Performance PDF → YOLO → JSON Pipeline with PDF Layer Support",
    version="2.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global model and config
YOLO_MODEL = None
MODEL_CLASSES = []

# Optimized configuration for large drawings
CONFIG = {
    "dpi": 200,  # Lower DPI for memory efficiency
    "tile_size": 640,
    "tile_overlap": 0.1,
    "batch_size": 8,  # Process 8 tiles at once
    "max_tiles_per_page": 500,  # Safety limit
    "confidence_threshold": 0.25,
    "model_path": "models/ves_model.pt",  # ⚠️ YOU NEED TO PROVIDE THIS MODEL
    "memory_threshold": 80,  # Stop if memory > 80%
    "enable_memory_cleanup": True
}

# ==================== PYDANTIC MODELS ====================

class Detection(BaseModel):
    label: str
    confidence: float
    x: float  # Center X in mm
    y: float  # Center Y in mm  
    width: float  # Width in mm
    height: float  # Height in mm

class TileStats(BaseModel):
    total_tiles: int
    processed_tiles: int
    skipped_tiles: int
    batch_count: int
    memory_cleanups: int

class LayerResult(BaseModel):
    layer_name: str
    layer_visible: bool
    resolution: str  # "7000x9000"
    tile_stats: TileStats
    total_detections: int
    processing_time_seconds: float
    detections: List[Detection]

class PageResult(BaseModel):
    page: int
    page_width_mm: float
    page_height_mm: float
    has_layers: bool
    total_layers: int
    total_processing_time: float
    # Either layers (if has_layers=True) or fallback data (if has_layers=False)
    layers: Optional[List[LayerResult]] = None
    # Fallback fields (when no layers)
    original_resolution: Optional[str] = None
    tile_stats: Optional[TileStats] = None
    total_detections: Optional[int] = None
    processing_time_seconds: Optional[float] = None
    detections: Optional[List[Detection]] = None

class VESResult(BaseModel):
    status: str
    filename: str
    total_pages: int
    model_used: str
    has_layers: bool
    total_processing_time: float
    pages: List[PageResult]

# ==================== LAYER DETECTION FUNCTIONS ====================

def extract_comprehensive_layers(pdf_doc: fitz.Document) -> Dict[str, Any]:
    """Extract complete layer information from PDF document (same as Vector API)"""
    layers = []
    ocg_info = {}
    layer_configurations = []
    pages_with_layers = []
    layer_names = set()
    layer_usage_stats = {}
    
    # Get PDF metadata
    metadata = pdf_doc.metadata
    pdf_version = getattr(pdf_doc, 'pdf_version', 'Unknown')
    
    try:
        # Method 1: Extract OCG information from document catalog
        ocg_info = extract_ocg_catalog_info(pdf_doc)
        
        # Method 2: Extract from xref table
        xref_layers = extract_layers_from_xref(pdf_doc)
        
        # Combine all layer information
        all_layers = {}
        
        # Process OCG catalog info
        if ocg_info.get('ocgs'):
            for ocg_ref, ocg_data in ocg_info['ocgs'].items():
                layer_name = ocg_data.get('name', f'Layer_{ocg_ref}')
                all_layers[layer_name] = {
                    'name': layer_name,
                    'ocg_ref': ocg_ref,
                    'visible': ocg_data.get('visible', True),
                    'locked': ocg_data.get('locked', False),
                    'intent': ocg_data.get('intent', []),
                    'usage': ocg_data.get('usage', {}),
                    'creator_info': ocg_data.get('creator_info', ''),
                    'source': 'OCG_catalog'
                }
                layer_names.add(layer_name)
        
        # Process xref layers
        for layer_data in xref_layers:
            layer_name = layer_data['name']
            if layer_name in all_layers:
                all_layers[layer_name].update({k: v for k, v in layer_data.items() if v})
            else:
                all_layers[layer_name] = layer_data
                all_layers[layer_name]['source'] = 'xref_table'
            layer_names.add(layer_name)
        
        # Convert to list format
        layers = list(all_layers.values())
        
    except Exception as e:
        logger.error(f"Error in comprehensive layer extraction: {str(e)}")
        return {
            "has_layers": False,
            "layer_count": 0,
            "layers": [],
            "ocg_info": {},
            "layer_configurations": [],
            "pages_with_layers": [],
            "layer_usage_analysis": {}
        }
    
    return {
        "has_layers": len(layer_names) > 0,
        "layer_count": len(layer_names),
        "layers": layers,
        "ocg_info": ocg_info,
        "layer_configurations": layer_configurations,
        "pages_with_layers": pages_with_layers,
        "layer_usage_analysis": layer_usage_stats
    }

def extract_ocg_catalog_info(pdf_doc: fitz.Document) -> Dict[str, Any]:
    """Extract OCG information from document catalog"""
    ocg_info = {'ocgs': {}, 'default_config': {}, 'alternate_configs': []}
    
    try:
        catalog = pdf_doc.pdf_catalog()
        if 'OCProperties' in catalog:
            oc_props = catalog['OCProperties']
            if 'OCGs' in oc_props:
                ocgs = oc_props['OCGs']
                for i, ocg_ref in enumerate(ocgs):
                    try:
                        ocg_obj = pdf_doc.xref_get_object(ocg_ref)
                        ocg_data = parse_ocg_object(ocg_obj)
                        ocg_info['ocgs'][str(ocg_ref)] = ocg_data
                    except:
                        continue
    except Exception as e:
        logger.debug(f"Could not extract OCG catalog info: {str(e)}")
    
    return ocg_info

def parse_ocg_object(ocg_obj: str) -> Dict[str, Any]:
    """Parse OCG object string to extract layer information"""
    import re
    ocg_data = {
        'name': 'Unknown Layer',
        'visible': True,
        'locked': False,
        'intent': [],
        'usage': {},
        'creator_info': ''
    }
    
    try:
        # Extract name
        name_match = re.search(r'/Name\s*\((.*?)\)', ocg_obj)
        if name_match:
            ocg_data['name'] = name_match.group(1)
    except Exception as e:
        logger.debug(f"Error parsing OCG object: {str(e)}")
    
    return ocg_data

def extract_layers_from_xref(pdf_doc: fitz.Document) -> List[Dict[str, Any]]:
    """Extract layer information from xref table"""
    layers = []
    
    try:
        xref_count = pdf_doc.xref_length()
        
        for xref in range(xref_count):
            try:
                obj_type = pdf_doc.xref_get_key(xref, "Type")
                if obj_type and "OCG" in str(obj_type):
                    name_obj = pdf_doc.xref_get_key(xref, "Name")
                    layer_name = str(name_obj).strip('()/"') if name_obj else f"Layer_{xref}"
                    
                    layer_data = {
                        "name": layer_name,
                        "visible": True,
                        "locked": False,
                        "ocg_ref": str(xref),
                        "intent": [],
                        "usage": {},
                        "xref_number": xref
                    }
                    
                    layers.append(layer_data)
                    
            except Exception as e:
                logger.debug(f"Error processing xref {xref}: {str(e)}")
                continue
                
    except Exception as e:
        logger.debug(f"Error in xref layer extraction: {str(e)}")
    
    return layers

# ==================== PDF TO PNG CONVERSION WITH LAYERS ====================

def convert_pdf_page_to_png_per_layer(pdf_doc: fitz.Document, page_num: int, layers: List[Dict], dpi: int = 200) -> List[Tuple[Image.Image, Dict]]:
    """
    Convert PDF page to PNG images per layer
    
    Returns:
        List of (PIL.Image, metadata) tuples, one per layer
    """
    logger.info(f"Converting page {page_num + 1} to PNG per layer at {dpi} DPI")
    images_with_metadata = []
    
    try:
        page = pdf_doc[page_num]
        page_rect = page.rect
        
        # Calculate zoom factor
        zoom_factor = dpi / 72.0
        matrix = fitz.Matrix(zoom_factor, zoom_factor)
        
        # Get base page dimensions
        pdf_width_pt = page_rect.width
        pdf_height_pt = page_rect.height
        pdf_width_mm = pdf_width_pt * 0.352778
        pdf_height_mm = pdf_height_pt * 0.352778
        
        # Process each layer
        for layer in layers:
            layer_name = layer['name']
            layer_visible = layer.get('visible', True)
            ocg_ref = layer.get('ocg_ref')
            
            logger.info(f"Processing layer: {layer_name} (visible: {layer_visible})")
            
            try:
                # Create a copy of the page for this layer
                layer_page = pdf_doc[page_num]
                
                # TODO: Enable/disable specific layer for rendering
                # This is a complex operation that depends on PDF structure
                # For now, we render the full page (all layers visible)
                # In a production environment, you'd need to:
                # 1. Parse OCG states
                # 2. Modify layer visibility
                # 3. Render with specific layer settings
                
                # Render page to PNG
                pixmap = layer_page.get_pixmap(matrix=matrix)
                png_data = pixmap.tobytes("png")
                
                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(png_data))
                
                # Calculate actual PNG dimensions
                png_width_px = pil_image.width
                png_height_px = pil_image.height
                
                # Calculate conversion factors
                pixels_per_point_x = png_width_px / pdf_width_pt
                pixels_per_point_y = png_height_px / pdf_height_pt
                pixels_per_mm_x = png_width_px / pdf_width_mm
                pixels_per_mm_y = png_height_px / pdf_height_mm
                
                # Complete metadata for coordinate conversion
                metadata = {
                    "page_num": page_num + 1,
                    "layer_name": layer_name,
                    "layer_visible": layer_visible,
                    "layer_ocg_ref": ocg_ref,
                    
                    # Original PDF dimensions
                    "pdf_width_pt": pdf_width_pt,
                    "pdf_height_pt": pdf_height_pt,
                    "pdf_width_mm": pdf_width_mm,
                    "pdf_height_mm": pdf_height_mm,
                    
                    # PNG dimensions
                    "png_width_px": png_width_px,
                    "png_height_px": png_height_px,
                    
                    # Conversion factors
                    "pixels_per_point_x": pixels_per_point_x,
                    "pixels_per_point_y": pixels_per_point_y,
                    "pixels_per_mm_x": pixels_per_mm_x,
                    "pixels_per_mm_y": pixels_per_mm_y,
                    
                    # DPI info
                    "dpi": dpi,
                    "zoom_factor": zoom_factor
                }
                
                images_with_metadata.append((pil_image, metadata))
                
                logger.info(f"✅ Layer {layer_name}: {png_width_px}×{png_height_px}px")
                
            except Exception as e:
                logger.error(f"❌ Failed to process layer {layer_name}: {e}")
                continue
        
        logger.info(f"✅ Page {page_num + 1}: Generated {len(images_with_metadata)} layer images")
        return images_with_metadata
        
    except Exception as e:
        logger.error(f"❌ PDF to PNG per layer conversion failed: {e}")
        return []

def convert_pdf_page_to_png_fallback(pdf_doc: fitz.Document, page_num: int, dpi: int = 200) -> Tuple[Image.Image, Dict]:
    """
    Fallback: Convert PDF page to single PNG (no layers)
    """
    logger.info(f"Converting page {page_num + 1} to PNG (fallback mode) at {dpi} DPI")
    
    try:
        page = pdf_doc[page_num]
        page_rect = page.rect
        
        # Calculate zoom factor
        zoom_factor = dpi / 72.0
        matrix = fitz.Matrix(zoom_factor, zoom_factor)
        
        # Get base page dimensions
        pdf_width_pt = page_rect.width
        pdf_height_pt = page_rect.height
        pdf_width_mm = pdf_width_pt * 0.352778
        pdf_height_mm = pdf_height_pt * 0.352778
        
        # Render page to PNG
        pixmap = page.get_pixmap(matrix=matrix)
        png_data = pixmap.tobytes("png")
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(png_data))
        
        # Calculate actual PNG dimensions
        png_width_px = pil_image.width
        png_height_px = pil_image.height
        
        # Calculate conversion factors
        pixels_per_point_x = png_width_px / pdf_width_pt
        pixels_per_point_y = png_height_px / pdf_height_pt
        pixels_per_mm_x = png_width_px / pdf_width_mm
        pixels_per_mm_y = png_height_px / pdf_height_mm
        
        # Complete metadata for coordinate conversion
        metadata = {
            "page_num": page_num + 1,
            "layer_name": None,
            "layer_visible": True,
            "layer_ocg_ref": None,
            
            # Original PDF dimensions
            "pdf_width_pt": pdf_width_pt,
            "pdf_height_pt": pdf_height_pt,
            "pdf_width_mm": pdf_width_mm,
            "pdf_height_mm": pdf_height_mm,
            
            # PNG dimensions
            "png_width_px": png_width_px,
            "png_height_px": png_height_px,
            
            # Conversion factors
            "pixels_per_point_x": pixels_per_point_x,
            "pixels_per_point_y": pixels_per_point_y,
            "pixels_per_mm_x": pixels_per_mm_x,
            "pixels_per_mm_y": pixels_per_mm_y,
            
            # DPI info
            "dpi": dpi,
            "zoom_factor": zoom_factor
        }
        
        logger.info(f"✅ Page {page_num + 1} (fallback): {png_width_px}×{png_height_px}px")
        return pil_image, metadata
        
    except Exception as e:
        logger.error(f"❌ PDF to PNG fallback conversion failed: {e}")
        raise

# ==================== YOLO DETECTION FUNCTIONS ====================

@app.on_event("startup")
async def load_model():
    """Load YOLO model with memory optimization"""
    global YOLO_MODEL, MODEL_CLASSES
    
    logger.info("🚀 Loading optimized VES YOLO model...")
    
    try:
        model_path = CONFIG["model_path"]
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            logger.error("⚠️ YOU NEED TO PROVIDE A TRAINED YOLO MODEL AT: models/ves_model.pt")
            logger.error("   1. Train your model on Roboflow")
            logger.error("   2. Export as YOLOv8 PyTorch (.pt) format")
            logger.error("   3. Place it at models/ves_model.pt")
            return
        
        # Load model with CPU optimization
        YOLO_MODEL = YOLO(model_path)
        YOLO_MODEL.to('cpu')  # Ensure CPU usage
        
        MODEL_CLASSES = list(YOLO_MODEL.names.values())
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"📋 Classes: {MODEL_CLASSES}")
        
        # Log memory usage
        memory = psutil.virtual_memory()
        logger.info(f"💾 Memory usage after model load: {memory.percent:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

def check_memory_usage() -> float:
    """Check current memory usage percentage"""
    return psutil.virtual_memory().percent

def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def calculate_optimal_batching(total_tiles: int, available_memory_mb: int) -> int:
    """Calculate optimal batch size based on available memory"""
    
    # Estimate memory per tile (rough calculation)
    memory_per_tile_mb = 50  # ~50MB per 640x640 tile processing
    
    # Calculate safe batch size
    safe_batch_size = max(1, min(
        CONFIG["batch_size"],
        available_memory_mb // (memory_per_tile_mb * 2),  # 2x safety margin
        total_tiles  # Don't exceed total tiles
    ))
    
    logger.info(f"📊 Optimal batch size: {safe_batch_size} (from {total_tiles} total tiles)")
    return safe_batch_size

def create_tiles_optimized(image: Image.Image, tile_size: int = 640, overlap: float = 0.1) -> List[Tuple[Image.Image, Dict]]:
    """Optimized tiling for large images with memory management"""
    img_width, img_height = image.size
    step_size = int(tile_size * (1 - overlap))
    
    logger.info(f"🔲 Creating tiles for {img_width}x{img_height}px image")
    logger.info(f"📐 Tile size: {tile_size}px, Overlap: {overlap*100}%, Step: {step_size}px")
    
    # Calculate total tiles
    x_tiles = math.ceil((img_width - tile_size) / step_size) + 1 if img_width > tile_size else 1
    y_tiles = math.ceil((img_height - tile_size) / step_size) + 1 if img_height > tile_size else 1
    total_tiles = x_tiles * y_tiles
    
    logger.info(f"📊 Total tiles to create: {x_tiles} x {y_tiles} = {total_tiles}")
    
    # Safety check
    if total_tiles > CONFIG["max_tiles_per_page"]:
        logger.warning(f"⚠️ Too many tiles ({total_tiles}), reducing image size")
        # Reduce image size to stay within limits
        scale_factor = math.sqrt(CONFIG["max_tiles_per_page"] / total_tiles)
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_width, img_height = image.size
        
        # Recalculate
        x_tiles = math.ceil((img_width - tile_size) / step_size) + 1 if img_width > tile_size else 1
        y_tiles = math.ceil((img_height - tile_size) / step_size) + 1 if img_height > tile_size else 1
        total_tiles = x_tiles * y_tiles
        logger.info(f"📏 Resized to {img_width}x{img_height}px, {total_tiles} tiles")
    
    tiles = []
    tile_count = 0
    
    for y in range(0, img_height, step_size):
        for x in range(0, img_width, step_size):
            # Calculate tile bounds
            x1 = x
            y1 = y
            x2 = min(x + tile_size, img_width)
            y2 = min(y + tile_size, img_height)
            
            # Skip tiny tiles
            if (x2 - x1) < tile_size * 0.5 or (y2 - y1) < tile_size * 0.5:
                continue
            
            # Extract tile (use crop which is memory efficient)
            tile = image.crop((x1, y1, x2, y2))
            
            # Pad if necessary
            if tile.size != (tile_size, tile_size):
                padded_tile = Image.new('RGB', (tile_size, tile_size), color='white')
                padded_tile.paste(tile, (0, 0))
                tile = padded_tile
            
            # Tile metadata
            tile_metadata = {
                "tile_id": tile_count,
                "x_offset": x1,
                "y_offset": y1,
                "original_width": x2 - x1,
                "original_height": y2 - y1,
                "tile_size": tile_size
            }
            
            tiles.append((tile, tile_metadata))
            tile_count += 1
            
            # Memory check every 50 tiles
            if tile_count % 50 == 0:
                memory_usage = check_memory_usage()
                if memory_usage > CONFIG["memory_threshold"]:
                    logger.warning(f"⚠️ High memory usage: {memory_usage:.1f}%")
                    cleanup_memory()
    
    logger.info(f"✅ Created {len(tiles)} tiles successfully")
    return tiles

async def run_yolo_batch(tiles_batch: List[Tuple[Image.Image, Dict]], confidence: float = 0.25) -> List[Tuple[List[Dict], Dict]]:
    """Run YOLO on a batch of tiles efficiently"""
    if YOLO_MODEL is None:
        logger.error("❌ YOLO model not loaded")
        return []
    
    batch_results = []
    
    try:
        # Convert all tiles to numpy arrays
        tile_arrays = []
        tile_metas = []
        
        for tile_img, tile_meta in tiles_batch:
            tile_arrays.append(np.array(tile_img))
            tile_metas.append(tile_meta)
        
        # Run batch prediction (much faster than individual predictions)
        logger.debug(f"🔍 Running YOLO on batch of {len(tile_arrays)} tiles")
        
        batch_predictions = YOLO_MODEL.predict(
            tile_arrays,
            conf=confidence,
            verbose=False,
            device='cpu',
            batch=len(tile_arrays)  # Process as true batch
        )
        
        # Process results
        for i, (result, tile_meta) in enumerate(zip(batch_predictions, tile_metas)):
            detections = []
            
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence_score = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    class_name = MODEL_CLASSES[class_id] if class_id < len(MODEL_CLASSES) else f"class_{class_id}"
                    
                    detection = {
                        "label": class_name,
                        "confidence": confidence_score,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "center_x": (x1 + x2) / 2,
                        "center_y": (y1 + y2) / 2,
                        "width": x2 - x1,
                        "height": y2 - y1
                    }
                    detections.append(detection)
            
            batch_results.append((detections, tile_meta))
        
        return batch_results
        
    except Exception as e:
        logger.error(f"❌ Batch YOLO prediction failed: {e}")
        return [([], meta) for _, meta in tiles_batch]

def convert_png_pixels_to_pdf_coordinates(png_detections: List[Dict], page_metadata: Dict) -> List[Detection]:
    """Convert PNG pixel coordinates to PDF coordinates (in mm)"""
    pdf_detections = []
    
    # Get conversion factors from metadata
    pixels_per_mm_x = page_metadata["pixels_per_mm_x"]
    pixels_per_mm_y = page_metadata["pixels_per_mm_y"]
    
    logger.info(f"📏 Using conversion factors: {pixels_per_mm_x:.2f}px/mm (X), {pixels_per_mm_y:.2f}px/mm (Y)")
    
    for detection in png_detections:
        # Convert PNG pixels to PDF millimeters
        pdf_center_x_mm = detection["png_center_x"] / pixels_per_mm_x
        pdf_center_y_mm = detection["png_center_y"] / pixels_per_mm_y
        pdf_width_mm = detection["png_width"] / pixels_per_mm_x
        pdf_height_mm = detection["png_height"] / pixels_per_mm_y
        
        # Create Detection object with PDF coordinates
        pdf_detection = Detection(
            label=detection["label"],
            confidence=round(detection["confidence"], 3),
            x=round(pdf_center_x_mm, 2),      # Center X in mm
            y=round(pdf_center_y_mm, 2),      # Center Y in mm
            width=round(pdf_width_mm, 2),     # Width in mm
            height=round(pdf_height_mm, 2)    # Height in mm
        )
        
        pdf_detections.append(pdf_detection)
    
    logger.info(f"📏 Converted {len(pdf_detections)} detections to PDF coordinates (mm)")
    return pdf_detections

def convert_tile_coordinates_to_png_pixels(detections_with_meta: List[Tuple[List[Dict], Dict]]) -> List[Dict]:
    """Convert tile-relative coordinates to full PNG pixel coordinates"""
    png_pixel_detections = []
    
    for tile_detections, tile_meta in detections_with_meta:
        tile_x_offset = tile_meta["x_offset"]
        tile_y_offset = tile_meta["y_offset"]
        
        for detection in tile_detections:
            # Convert tile coordinates to full PNG coordinates
            png_detection = {
                "label": detection["label"],
                "confidence": detection["confidence"],
                
                # Bounding box in full PNG pixel coordinates
                "png_x1": detection["x1"] + tile_x_offset,
                "png_y1": detection["y1"] + tile_y_offset,
                "png_x2": detection["x2"] + tile_x_offset,
                "png_y2": detection["y2"] + tile_y_offset,
                
                # Center and dimensions in PNG pixels
                "png_center_x": detection["center_x"] + tile_x_offset,
                "png_center_y": detection["center_y"] + tile_y_offset,
                "png_width": detection["width"],
                "png_height": detection["height"]
            }
            
            png_pixel_detections.append(png_detection)
    
    logger.info(f"📐 Converted {len(png_pixel_detections)} detections to PNG pixel coordinates")
    return png_pixel_detections

def merge_overlapping_detections_fast(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """Fast overlap merging optimized for large detection lists"""
    if not detections:
        return []
    
    # Group by label first (only merge same-label detections)
    label_groups = {}
    for det in detections:
        if det.label not in label_groups:
            label_groups[det.label] = []
        label_groups[det.label].append(det)
    
    merged_all = []
    
    for label, label_detections in label_groups.items():
        if len(label_detections) == 1:
            merged_all.extend(label_detections)
            continue
        
        # Sort by confidence (highest first)
        label_detections.sort(key=lambda x: x.confidence, reverse=True)
        
        merged_label = []
        used = set()
        
        for i, det1 in enumerate(label_detections):
            if i in used:
                continue
                
            merged_label.append(det1)
            used.add(i)
            
            # Check overlaps (optimized - only check remaining items)
            for j in range(i + 1, len(label_detections)):
                if j in used:
                    continue
                    
                det2 = label_detections[j]
                
                # Quick distance check before expensive IoU
                dx = abs(det1.x - det2.x)
                dy = abs(det1.y - det2.y)
                if dx > (det1.width + det2.width) / 2 or dy > (det1.height + det2.height) / 2:
                    continue
                
                # Calculate IoU
                iou = calculate_iou_fast(det1, det2)
                if iou > iou_threshold:
                    used.add(j)
        
        merged_all.extend(merged_label)
    
    logger.info(f"📊 Merged {len(detections)} → {len(merged_all)} detections")
    return merged_all

def calculate_iou_fast(det1: Detection, det2: Detection) -> float:
    """Fast IoU calculation"""
    # Convert center+size to corners
    x1_1 = det1.x - det1.width / 2
    y1_1 = det1.y - det1.height / 2
    x2_1 = det1.x + det1.width / 2
    y2_1 = det1.y + det1.height / 2
    
    x1_2 = det2.x - det2.width / 2
    y1_2 = det2.y - det2.height / 2
    x2_2 = det2.x + det2.width / 2
    y2_2 = det2.y + det2.height / 2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = det1.width * det1.height
    area2 = det2.width * det2.height
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

# ==================== MAIN PROCESSING FUNCTIONS ====================

async def process_layer_image(image: Image.Image, layer_metadata: Dict, confidence: float) -> LayerResult:
    """Process a single layer image with YOLO detection"""
    start_time = time.time()
    layer_name = layer_metadata["layer_name"]
    
    logger.info(f"🔍 Processing layer: {layer_name}")
    
    # Step 1: Create tiles
    tiles = create_tiles_optimized(image, CONFIG["tile_size"], CONFIG["tile_overlap"])
    total_tiles = len(tiles)
    
    # Step 2: Calculate optimal batching
    available_memory = psutil.virtual_memory().available // (1024 * 1024)  # MB
    batch_size = calculate_optimal_batching(total_tiles, available_memory)
    
    # Step 3: Process tiles in batches
    all_detections_with_meta = []
    processed_tiles = 0
    skipped_tiles = 0
    memory_cleanups = 0
    batch_count = 0
    
    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i + batch_size]
        batch_count += 1
        
        logger.info(f"📦 Processing batch {batch_count}: tiles {i+1}-{min(i+batch_size, total_tiles)} of {total_tiles}")
        
        # Memory check
        memory_usage = check_memory_usage()
        if memory_usage > CONFIG["memory_threshold"]:
            logger.warning(f"⚠️ High memory usage ({memory_usage:.1f}%), cleaning up...")
            cleanup_memory()
            memory_cleanups += 1
        
        # Process batch
        try:
            batch_results = await run_yolo_batch(batch, confidence)
            all_detections_with_meta.extend(batch_results)
            processed_tiles += len(batch)
            
        except Exception as e:
            logger.error(f"❌ Batch {batch_count} failed: {e}")
            skipped_tiles += len(batch)
            # Add empty results for failed batch
            all_detections_with_meta.extend([([], meta) for _, meta in batch])
        
        # Progress logging
        progress = (processed_tiles + skipped_tiles) / total_tiles * 100
        logger.info(f"⏳ Progress: {progress:.1f}% ({processed_tiles + skipped_tiles}/{total_tiles} tiles)")
    
    # Step 4: Convert coordinates
    logger.info("📐 Converting coordinates...")
    png_pixel_detections = convert_tile_coordinates_to_png_pixels(all_detections_with_meta)
    pdf_detections = convert_png_pixels_to_pdf_coordinates(png_pixel_detections, layer_metadata)
    
    # Step 5: Merge overlapping detections
    logger.info("🔗 Merging overlapping detections...")
    merged_detections = merge_overlapping_detections_fast(pdf_detections)
    
    # Final cleanup
    if CONFIG["enable_memory_cleanup"]:
        cleanup_memory()
        memory_cleanups += 1
    
    processing_time = time.time() - start_time
    
    # Create tile stats
    tile_stats = TileStats(
        total_tiles=total_tiles,
        processed_tiles=processed_tiles,
        skipped_tiles=skipped_tiles,
        batch_count=batch_count,
        memory_cleanups=memory_cleanups
    )
    
    result = LayerResult(
        layer_name=layer_name,
        layer_visible=layer_metadata.get("layer_visible", True),
        resolution=f"{layer_metadata['png_width_px']}x{layer_metadata['png_height_px']}",
        tile_stats=tile_stats,
        total_detections=len(merged_detections),
        processing_time_seconds=round(processing_time, 2),
        detections=merged_detections
    )
    
    logger.info(f"✅ Layer {layer_name} complete!")
    logger.info(f"📊 Stats: {total_tiles} tiles → {len(merged_detections)} detections in {processing_time:.2f}s")
    
    return result

async def process_fallback_image(image: Image.Image, page_metadata: Dict, confidence: float) -> Tuple[TileStats, List[Detection], float]:
    """Process single image without layers (fallback mode)"""
    start_time = time.time()
    
    logger.info(f"🔍 Processing image (fallback mode)")
    
    # Step 1: Create tiles
    tiles = create_tiles_optimized(image, CONFIG["tile_size"], CONFIG["tile_overlap"])
    total_tiles = len(tiles)
    
    # Step 2: Calculate optimal batching
    available_memory = psutil.virtual_memory().available // (1024 * 1024)  # MB
    batch_size = calculate_optimal_batching(total_tiles, available_memory)
    
    # Step 3: Process tiles in batches
    all_detections_with_meta = []
    processed_tiles = 0
    skipped_tiles = 0
    memory_cleanups = 0
    batch_count = 0
    
    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i + batch_size]
        batch_count += 1
        
        logger.info(f"📦 Processing batch {batch_count}: tiles {i+1}-{min(i+batch_size, total_tiles)} of {total_tiles}")
        
        # Memory check
        memory_usage = check_memory_usage()
        if memory_usage > CONFIG["memory_threshold"]:
            logger.warning(f"⚠️ High memory usage ({memory_usage:.1f}%), cleaning up...")
            cleanup_memory()
            memory_cleanups += 1
        
        # Process batch
        try:
            batch_results = await run_yolo_batch(batch, confidence)
            all_detections_with_meta.extend(batch_results)
            processed_tiles += len(batch)
            
        except Exception as e:
            logger.error(f"❌ Batch {batch_count} failed: {e}")
            skipped_tiles += len(batch)
            all_detections_with_meta.extend([([], meta) for _, meta in batch])
        
        # Progress logging
        progress = (processed_tiles + skipped_tiles) / total_tiles * 100
        logger.info(f"⏳ Progress: {progress:.1f}% ({processed_tiles + skipped_tiles}/{total_tiles} tiles)")
    
    # Step 4: Convert coordinates
    logger.info("📐 Converting coordinates...")
    png_pixel_detections = convert_tile_coordinates_to_png_pixels(all_detections_with_meta)
    pdf_detections = convert_png_pixels_to_pdf_coordinates(png_pixel_detections, page_metadata)
    
    # Step 5: Merge overlapping detections
    logger.info("🔗 Merging overlapping detections...")
    merged_detections = merge_overlapping_detections_fast(pdf_detections)
    
    # Final cleanup
    if CONFIG["enable_memory_cleanup"]:
        cleanup_memory()
        memory_cleanups += 1
    
    processing_time = time.time() - start_time
    
    # Create tile stats
    tile_stats = TileStats(
        total_tiles=total_tiles,
        processed_tiles=processed_tiles,
        skipped_tiles=skipped_tiles,
        batch_count=batch_count,
        memory_cleanups=memory_cleanups
    )
    
    logger.info(f"✅ Fallback processing complete!")
    logger.info(f"📊 Stats: {total_tiles} tiles → {len(merged_detections)} detections in {processing_time:.2f}s")
    
    return tile_stats, merged_detections, processing_time

# ==================== API ENDPOINTS ====================

@app.post("/analyze-pdf-with-layers/", response_model=VESResult)
async def analyze_pdf_with_layer_support(
    file: UploadFile = File(...),
    tekeningtype: str = Form(...),  # plattegrond, doorsnede, gevel, bestektekening, detailtekening
    confidence: float = Form(default=0.25),
    dpi: int = Form(default=200)
):
    """
    🏗 VES API v2 with PDF Layer Support
    
    NEW in v2:
    - Detects PDF layers automatically
    - If layers found: processes each layer separately with YOLO detection
    - If no layers: falls back to normal batching (backward compatible)
    - Each detection includes layer information when available
    
    Returns:
    - JSON with detections per layer (if layers) or standard format (if no layers)
    """
    
    start_time = time.time()
    
    try:
        logger.info(f"🚀 VES API v2 request: {file.filename}, type: {tekeningtype}")
        
        # Validate drawing type
        valid_types = ["plattegrond", "doorsnede", "gevel", "bestektekening", "detailtekening"]
        if tekeningtype not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid tekeningtype. Must be one of: {valid_types}"
            )
        
        # Validate other inputs
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        if not 0.1 <= confidence <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence must be between 0.1 and 1.0")
        
        if YOLO_MODEL is None:
            raise HTTPException(status_code=503, detail="YOLO model not loaded - Please provide trained model at models/ves_model.pt")
        
        logger.info(f"⚙️ Settings: type={tekeningtype}, confidence={confidence}, dpi={dpi}")
        
        # Step 1: Read PDF
        pdf_bytes = await file.read()
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty PDF file")
        
        logger.info(f"📄 PDF size: {len(pdf_bytes)} bytes")
        
        # Step 2: Open PDF and detect layers
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        logger.info("=== Checking for PDF layers ===")
        layer_info = extract_comprehensive_layers(pdf_document)
        has_layers = layer_info["has_layers"]
        
        if has_layers:
            logger.info(f"✅ Found {layer_info['layer_count']} layers in PDF")
            for layer in layer_info["layers"]:
                logger.info(f"  - Layer: {layer['name']} (visible: {layer['visible']})")
        else:
            logger.info("📄 No layers detected - using fallback processing")
        
        # Step 3: Process each page
        page_results = []
        
        for page_num in range(len(pdf_document)):
            page_start_time = time.time()
            logger.info(f"📐 Processing page {page_num + 1} of {len(pdf_document)}")
            
            if has_layers:
                # NEW: Process per layer
                logger.info(f"🔍 Processing page {page_num + 1} with layer support")
                
                # Convert PDF page to PNG per layer
                layer_images = convert_pdf_page_to_png_per_layer(
                    pdf_document, page_num, layer_info["layers"], dpi
                )
                
                if not layer_images:
                    logger.warning(f"⚠️ No layer images generated for page {page_num + 1}, falling back")
                    # Fallback to single image
                    image, page_metadata = convert_pdf_page_to_png_fallback(pdf_document, page_num, dpi)
                    tile_stats, detections, processing_time = await process_fallback_image(image, page_metadata, confidence)
                    
                    page_result = PageResult(
                        page=page_num + 1,
                        page_width_mm=page_metadata["pdf_width_mm"],
                        page_height_mm=page_metadata["pdf_height_mm"],
                        has_layers=False,
                        total_layers=0,
                        total_processing_time=processing_time,
                        original_resolution=f"{page_metadata['png_width_px']}x{page_metadata['png_height_px']}",
                        tile_stats=tile_stats,
                        total_detections=len(detections),
                        processing_time_seconds=processing_time,
                        detections=detections
                    )
                else:
                    # Process each layer
                    layer_results = []
                    total_layer_processing_time = 0
                    
                    for image, layer_metadata in layer_images:
                        layer_result = await process_layer_image(image, layer_metadata, confidence)
                        layer_results.append(layer_result)
                        total_layer_processing_time += layer_result.processing_time_seconds
                    
                    # Create page result with layers
                    page_result = PageResult(
                        page=page_num + 1,
                        page_width_mm=layer_images[0][1]["pdf_width_mm"],
                        page_height_mm=layer_images[0][1]["pdf_height_mm"],
                        has_layers=True,
                        total_layers=len(layer_results),
                        total_processing_time=total_layer_processing_time,
                        layers=layer_results
                    )
            else:
                # Fallback: Process as single image (backward compatible)
                logger.info(f"📄 Processing page {page_num + 1} in fallback mode (no layers)")
                
                image, page_metadata = convert_pdf_page_to_png_fallback(pdf_document, page_num, dpi)
                tile_stats, detections, processing_time = await process_fallback_image(image, page_metadata, confidence)
                
                page_result = PageResult(
                    page=page_num + 1,
                    page_width_mm=page_metadata["pdf_width_mm"],
                    page_height_mm=page_metadata["pdf_height_mm"],
                    has_layers=False,
                    total_layers=0,
                    total_processing_time=processing_time,
                    original_resolution=f"{page_metadata['png_width_px']}x{page_metadata['png_height_px']}",
                    tile_stats=tile_stats,
                    total_detections=len(detections),
                    processing_time_seconds=processing_time,
                    detections=detections
                )
            
            page_results.append(page_result)
            page_processing_time = time.time() - page_start_time
            logger.info(f"✅ Page {page_num + 1} completed in {page_processing_time:.2f}s")
        
        pdf_document.close()
        
        # Step 4: Create final result
        total_processing_time = time.time() - start_time
        
        result = VESResult(
            status="success",
            filename=file.filename,
            total_pages=len(page_results),
            model_used=CONFIG["model_path"],
            has_layers=has_layers,
            total_processing_time=round(total_processing_time, 2),
            pages=page_results
        )
        
        # Log final summary
        if has_layers:
            total_layer_detections = sum(
                sum(layer.total_detections for layer in page.layers) 
                for page in page_results if page.layers
            )
            total_layers = sum(page.total_layers for page in page_results)
            logger.info(f"🎉 Layer-based analysis complete!")
            logger.info(f"📊 Summary: {len(page_results)} pages, {total_layers} total layers, {total_layer_detections} detections")
        else:
            total_detections = sum(page.total_detections or 0 for page in page_results)
            logger.info(f"🎉 Fallback analysis complete!")
            logger.info(f"📊 Summary: {len(page_results)} pages, {total_detections} detections")
        
        logger.info(f"⏱ Total time: {total_processing_time:.2f}s")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health/")
async def health_check():
    """Enhanced health check with layer support info"""
    memory = psutil.virtual_memory()
    
    return {
        "status": "healthy",
        "model_loaded": YOLO_MODEL is not None,
        "model_classes": MODEL_CLASSES,
        "config": CONFIG,
        "memory_usage": {
            "percent": memory.percent,
            "available_mb": memory.available // (1024 * 1024),
            "total_mb": memory.total // (1024 * 1024)
        },
        "version": "2.0.0 - Layer Support",
        "new_features_v2": [
            "🔍 Automatic PDF layer detection",
            "📋 YOLO detection per layer",
            "🔄 Fallback to normal batching if no layers",
            "🏷️ Layer-tagged detection results",
            "📊 Per-layer processing statistics"
        ],
        "model_requirements": {
            "path": CONFIG["model_path"],
            "exists": os.path.exists(CONFIG["model_path"]),
            "note": "⚠️ YOU NEED TO PROVIDE A TRAINED YOLO MODEL" if not os.path.exists(CONFIG["model_path"]) else "✅ Model loaded"
        }
    }

@app.get("/")
async def root():
    """API information with layer support details"""
    return {
        "title": "VES Detection API v2 - Layer Support",
        "description": "High-Performance PDF → YOLO → JSON Pipeline with PDF Layer Support",
        "version": "2.0.0",
        "new_features_v2": [
            "🔍 Automatic PDF layer detection using OCG analysis",
            "📋 PDF → PNG conversion per layer",
            "🎯 YOLO object detection per layer separately",
            "🔄 Intelligent fallback to normal batching if no layers",
            "🏷️ Detection results tagged with layer information",
            "📊 Per-layer processing statistics and timing"
        ],
        "workflow": {
            "with_layers": [
                "1. Upload PDF + drawing type",
                "2. Detect PDF layers using OCG analysis",
                "3. Convert PDF → PNG per layer",
                "4. Run YOLO detection on each layer separately",
                "5. Return detections grouped by layer"
            ],
            "without_layers": [
                "1. Upload PDF + drawing type",
                "2. No layers detected",
                "3. Fallback: Convert PDF → PNG (single image)",
                "4. Run YOLO detection with normal batching",
                "5. Return standard detection format"
            ]
        },
        "output_formats": {
            "with_layers": {
                "pages": [{
                    "has_layers": True,
                    "total_layers": 3,
                    "layers": [{
                        "layer_name": "Walls",
                        "layer_visible": True,
                        "total_detections": 25,
                        "detections": ["wall detection objects in PDF coordinates (mm)"]
                    }]
                }]
            },
            "without_layers": {
                "pages": [{
                    "has_layers": False,
                    "total_detections": 50,
                    "detections": ["all detection objects in PDF coordinates (mm)"]
                }]
            }
        },
        "optimizations": [
            "Intelligent batch processing (8 tiles per batch)",
            "Memory management with automatic cleanup",
            "Resolution scaling for oversized images",
            "Fast overlap merging with label grouping",
            "Progress tracking and error recovery",
            "Per-layer processing isolation"
        ],
        "performance": {
            "max_resolution": "8000x8000px per layer (configurable)",
            "typical_processing_time": "3-8 minutes for 7k x 9k drawings with layers",
            "memory_usage": "Optimized for 512MB-1GB RAM",
            "tile_processing": "200+ tiles per layer handled efficiently"
        },
        "endpoints": {
            "/analyze-pdf-with-layers/": "Main analysis endpoint with layer support",
            "/health/": "Health check with layer support info",
            "/": "This information"
        },
        "parameters": {
            "file": "PDF file to analyze",
            "tekeningtype": "plattegrond|doorsnede|gevel|bestektekening|detailtekening",
            "confidence": "0.1-1.0 (YOLO detection threshold)",
            "dpi": "150-600 (PDF to PNG conversion quality)"
        },
        "compatibility": {
            "master_api": "✅ Compatible with Master API v4.1.1+",
            "n8n": "✅ Direct HTTP request support",
            "backward_compatible": "✅ Same output format when no layers detected"
        },
        "model_setup": {
            "required_file": "models/ves_model.pt",
            "instructions": [
                "1. Train your YOLO model on Roboflow with building drawing images",
                "2. Annotate objects (walls, doors, windows, etc.)",
                "3. Export as YOLOv8 PyTorch (.pt) format",
                "4. Place the .pt file at models/ves_model.pt",
                "5. Restart the API"
            ],
            "note": "⚠️ The API will not work without a trained YOLO model"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8002))
    logger.info(f"🚀 Starting VES Detection API v2 - Layer Support on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
