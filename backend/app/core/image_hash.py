import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

def solide_hash_robust(image_path, size=48):
    """
    Generates a solid hash invariant to ARBITRARY ROTATION, SCALING, and SATURATION.
    
    Logic:
    1. 'Quantize' Blur to remove noise.
    2. Normalize Rotation (De-rotate using Principal Axis).
    3. Normalize Flip (Canonical 180-degree orientation using 3rd-order moments).
    4. Compute D-Hash (Structural Gradient).
    """
    try:
        # 1. READ & GRAYSCALE (Using imdecode for better Windows path support)
        # We use fromfile to handle non-ASCII characters in paths
        file_bytes = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Image could not be decoded: {image_path}")
            
        # 2. QUANTIZE (Pre-processing)
        # Smooth out noise so it doesn't affect the angle calculation
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # 3. ROTATION NORMALIZATION (The Article's Method)
        # We rotate the image so its "mass" aligns with the X-axis.
        img = _normalize_rotation_moments(img)
        
        # 4. AUTO-CROP FLAT BACKGROUNDS
        # Strip away empty/black studio backgrounds so the structure fills the hash grid completely.
        img = _crop_to_content(img)
        
        # 5. RESIZE (Downsampling)
        # For D-Hash, we need width = size + 1 to compute adjacent differences
        img = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)

        # 5. FLIP NORMALIZATION (Canonical View)
        # Uses 3rd-order moments to resolve the 180-degree ambiguity perfectly.
        img = _get_canonical_view(img)

        # 6. HASH GENERATION (D-Hash)
        # Difference-Hash: Compares adjacent pixels horizontally
        diff = img[:, 1:] > img[:, :-1]
        
        # Pack bits
        bits = diff.flatten().astype(np.uint8)
        packed_bytes = np.packbits(bits)
        return packed_bytes.tobytes().hex()

    except Exception:
        # Re-raise to let caller handle it properly
        raise

def _normalize_rotation_moments(img):
    """
    Detects the principal axis of the image content and rotates it 
    to be horizontal. This handles arbitrary rotations (e.g., 45 degrees).
    """
    # Threshold to isolate the "object" or "structure" from background
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate Moments
    moments = cv2.moments(thresh)
    
    # Calculate orientation angle (theta)
    # Formula: 0.5 * atan2(2 * mu11, mu20 - mu02)
    mu11 = moments['mu11']
    mu20 = moments['mu20']
    mu02 = moments['mu02']
    
    # Avoid division by zero
    if (mu20 - mu02) == 0:
        return img 
        
    angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle_degrees = np.degrees(angle)
    
    # Rotate the image to align the axis
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    # Apply rotation (with padding to avoid cutting off corners)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def _crop_to_content(img):
    """
    Finds the bounding box of the non-empty content and crops the image.
    Uses Gaussian Blur + Otsu's binarization to cleanly separate the foreground 
    object from non-uniform studio backgrounds (like soft gradients).
    This prevents massive flat/gradient backgrounds from dominating the 
    hashing grid with identical 0 bits.
    """
    # Smooth out noise that might throw off the thresholding
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Use Otsu's method to automatically find the optimal threshold to separate
    # the background light/gradient from the physical object structure.
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find all non-zero pixels
    coords = cv2.findNonZero(thresh)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Ensure we don't crop to a 0x0 or 1x1 size, keep at least a small valid region
        if w > 4 and h > 4:
            return img[y:y+h, x:x+w]
            
    return img

def _get_canonical_view(img):
    """
    Fixes the 180-degree ambiguity from PCA rotation using 3rd-order spatial central moments (skewness).
    Ensures that the image content is 'skewed' towards the top-left (negative x and y skew).
    """
    # Threshold to isolate structure
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate Moments
    moments = cv2.moments(thresh)
    
    # Use 3rd order central moments to determine skewness.
    # mu30: Horizontal Skewness
    # mu03: Vertical Skewness
    mu30 = moments['mu30']
    mu03 = moments['mu03']
    
    canonical_img = img.copy()
    
    # Enforce a canonical orientation: we want negative skew (mass shifted towards top-left)
    if mu30 > 0:
        canonical_img = cv2.flip(canonical_img, 1) # Flip horizontally
        
    if mu03 > 0:
        canonical_img = cv2.flip(canonical_img, 0) # Flip vertically
        
    return canonical_img

def _hamming_distance(hash1, hash2):
    """
    Computes the bitwise difference between two hex strings.
    """
    # If either is not a string or empty, it's invalid
    if not isinstance(hash1, str) or not isinstance(hash2, str):
        raise ValueError("Hashes must be hex strings")
    if hash1.startswith("Error") or hash2.startswith("Error"):
        raise ValueError("Invalid hash (contains error string)")
        
    # Convert hex strings to integers
    int(hash1, 16)
    int(hash2, 16)
    
def _hex_to_bool_grid(hex_str):
    """
    Converts a hex string hash back into a 2D boolean numpy array.
    assumes hash was flattened row-major.
    """
    # 1. Hex to Bytes
    byte_data = bytes.fromhex(hex_str)
    
    # 2. Bytes to Bits (Numpy uint8 array of 0s and 1s)
    # unpackbits works on uint8 array
    # We used np.packbits(bits) previously.
    # np.unpackbits gives strictly 8 bits per byte
    bits = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
    
    # 3. Reshape to square
    total_bits = bits.size
    size = int(np.sqrt(total_bits))
    
    if size * size != total_bits:
        raise ValueError(f"Hash length ({total_bits} bits) is not a perfect square, cannot infer grid size.")
        
    return bits.reshape((size, size)).astype(bool)

def measure_similarity(hash1, hash2, hash_size=None):
    """
    Compares two hashes and returns a similarity score between 0.0 and 1.0.
    1.0 means identical, 0.0 means completely opposite.
    
    This implementation uses Spatial Difference Weighting.
    It penalizes heavily for grouped/clustered block differences (structural changes),
    while being lenient on sparse, separated bit flips (noise/compression).
    """
    if not hash1 or not hash2 or hash1.startswith("Error") or hash2.startswith("Error"):
        return 0.0
        
    try:
        # Convert both to boolean grids
        grid1 = _hex_to_bool_grid(hash1)
        grid2 = _hex_to_bool_grid(hash2)
    except Exception as e:
        logger.error("hash_parsing_failed", hash1=hash1, hash2=hash2, error=str(e))
        return 0.0

    if grid1.shape != grid2.shape:
        return 0.0
        
    # 1. Base Difference
    # diff is True where the hashes differ
    diff = (grid1 != grid2).astype(np.float32)
    total_bits = grid1.size
    
    # 2. Adaptive Max Cost Scaling
    # Instead of comparing against a totally scrambled imaginary board (which is impossible 
    # since flat surfaces like car doors don't produce bits in D-Hash), we measure the 
    # MAX potential feature changes based on the UNION of active bits in both hashes.
    # An active bit (True) represents a structural edge.
    active_union = (grid1 | grid2).astype(np.float32)
    
    if np.sum(active_union) == 0:
        return 1.0 # Both images are completely empty/flat
        
    # 3. Spatial Weighting via Convolution
    # Apply a 3x3 uniform convolution to count neighboring differences.
    # This weights grouped structural changes extremely heavily compared to isolated bit flips.
    kernel = np.ones((3, 3), dtype=np.float32)
    
    # Border isolation ensures edge bits don't wrap in the filter.
    density = cv2.filter2D(diff, -1, kernel, borderType=cv2.BORDER_ISOLATED)
    
    # We multiply the original difference by its local density.
    # Isolated difference: cost 1
    # 3x3 clustered difference: cost 9
    weighted_cost = diff * density
    
    # 4. Normalize Score Adaptively
    # The max cost for the visible features is if EVERY active edge in the union 
    # was completely flipped and perfectly clustered. 
    density_union = cv2.filter2D(active_union, -1, kernel, borderType=cv2.BORDER_ISOLATED)
    max_cost = np.sum(active_union * density_union)
    
    # Fallback to prevent division by zero or negative similarity
    if max_cost <= 0:
        return 0.0
        
    actual_cost = np.sum(weighted_cost)
    
    # Similarity is 1.0 minus the normalized cost, bounded
    sim = 1.0 - (actual_cost / max_cost)
    
    return float(max(0.0, sim))
