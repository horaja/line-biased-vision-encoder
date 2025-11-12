"""
Preprocessing pipeline for generating line drawings and magno images.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def create_magno_image(image_path, output_size=64):
    """
    Create a magno-channel image emphasizing low spatial frequencies.
    Simulates magnocellular pathway with low-pass filtering and non-linear contrast response.

    Args:
        image_path: Path to input image
        output_size: Output size for the magno image

    Returns:
        Magno image as numpy array (RGB)
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to grayscale (magno cells are achromatic)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Strong low-pass filtering (simulate large receptive fields)
    magno = cv2.GaussianBlur(gray, (21, 21), 7.0)

    # Downsample to output size (further removes high frequencies)
    magno = cv2.resize(magno, (output_size, output_size),
                       interpolation=cv2.INTER_AREA)

    # Non-linear contrast response (enhance low contrast, saturate high)
    magno = magno.astype(np.float32) / 255.0
    magno = np.power(magno, 0.7)
    magno = np.uint8(magno * 255)

    # Convert to RGB for consistency
    magno_rgb = cv2.cvtColor(magno, cv2.COLOR_GRAY2RGB)

    return magno_rgb


class InformativeDrawingsPreprocessor:
    """
    Wrapper for the informative-drawings preprocessing pipeline.
    """
    
    def __init__(
        self,
        informative_drawings_path: str = "third_party/informative-drawings",
        model_name: str = "contour_style"
    ):
        """
        Args:
            informative_drawings_path: Path to the informative-drawings repository
            model_name: Name of the pretrained model to use
        """
        self.repo_path = Path(informative_drawings_path)
        self.model_name = model_name
        
        # Verify the repo exists
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"Informative drawings repo not found at {self.repo_path}. "
                "Make sure you've initialized the git submodule."
            )
        
        # Verify the model checkpoint exists
        checkpoint_path = self.repo_path / "checkpoints" / model_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint_path}. "
                f"Please download the pretrained model and place it in checkpoints/"
            )
    
    def process_dataset(
        self,
        input_dir: str,
        output_magno_dir: str,
        output_lines_dir: str,
        output_color_dir: str,
        magno_size: int = 64,
        color_size: int = 256,
        use_gpu: bool = True
    ):
        """
        Process a dataset to generate line drawings and magno images.

        Args:
            input_dir: Directory containing input images
            output_magno_dir: Directory to save magno images
            output_lines_dir: Directory to save line drawings
            output_color_dir: Directory to save color images
            magno_size: Size for magno images
            color_size: Size for color images
            use_gpu: Whether to use GPU for processing
        """
        import torch
        import torchvision.transforms as transforms

        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Create output directories
        for out_dir in [output_magno_dir, output_lines_dir, output_color_dir]:
            Path(out_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing images from {input_dir}")
        logger.info(f"Outputs: magno={output_magno_dir}, lines={output_lines_dir}, color={output_color_dir}")
        logger.info(f"Configuration: magno_size={magno_size}x{magno_size}, color_size={color_size}x{color_size}")

        # Add informative-drawings to path and import modules
        sys.path.insert(0, str(self.repo_path))
        try:
            from model import Generator
        except ImportError as e:
            logger.error(f"Failed to import informative-drawings modules: {e}")
            logger.error("Make sure the informative-drawings submodule is initialized")
            raise

        # Load the line drawing model
        logger.info(f"Loading model: {self.model_name}")
        device = torch.device('cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu')

        # Model configuration (from informative-drawings test.py defaults)
        input_nc = 3
        output_nc = 1
        n_blocks = 3  # Default from test.py - matches pretrained checkpoints

        # Load generator
        netG = Generator(input_nc, output_nc, n_blocks)
        checkpoint_path = self.repo_path / "checkpoints" / self.model_name / "netG_A_latest.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
        netG.to(device)
        netG.eval()
        logger.info("Model loaded successfully")

        # Define image transform
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.rglob(f'*{ext}'))

        logger.info(f"Found {len(image_files)} images to process")

        # Process each image
        for i, img_path in enumerate(image_files, 1):
            # Get relative path to maintain directory structure
            rel_path = img_path.relative_to(input_path)
            img_name = img_path.stem
            class_dir = rel_path.parent

            if i % 100 == 0 or i == 1:
                logger.info(f"Processing image {i}/{len(image_files)}: {rel_path}")

            try:
                # Load and transform image
                img_pil = Image.open(img_path).convert('RGB')
                img_tensor = transform(img_pil).unsqueeze(0).to(device)

                # Generate line drawing
                with torch.no_grad():
                    line_tensor = netG(img_tensor)

                # Save line drawing
                line_output_dir = Path(output_lines_dir) / class_dir
                line_output_dir.mkdir(parents=True, exist_ok=True)
                line_image = transforms.ToPILImage()(line_tensor.squeeze(0).cpu())
                line_path = line_output_dir / f"{img_name}_line.png"
                line_image.save(line_path)

                # Generate and save magno image
                magno_output_dir = Path(output_magno_dir) / class_dir
                magno_output_dir.mkdir(parents=True, exist_ok=True)
                magno_array = create_magno_image(img_path, output_size=magno_size)
                magno_image = Image.fromarray(magno_array)
                magno_path = magno_output_dir / f"{img_name}_magno.png"
                magno_image.save(magno_path)

                # Resize and save color image
                color_output_dir = Path(output_color_dir) / class_dir
                color_output_dir.mkdir(parents=True, exist_ok=True)
                color_image = img_pil.resize((color_size, color_size), Image.LANCZOS)
                color_path = color_output_dir / f"{img_name}_color.png"
                color_image.save(color_path)

            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                continue

        logger.info("Preprocessing completed successfully")


def preprocess_imagenette(
    raw_data_root: str,
    preprocessed_root: str,
    magno_size: int = 64,
    color_size: int = 256,
    splits: Optional[list] = None
):
    """
    Convenience function to preprocess the full ImageNette dataset.
    
    Args:
        raw_data_root: Path to raw ImageNette data
        preprocessed_root: Path to save preprocessed data
        magno_size: Size for magno images
        color_size: Size for color images
        splits: List of splits to process (default: ['train', 'val'])
    """
    if splits is None:
        splits = ['train', 'val']
    
    preprocessor = InformativeDrawingsPreprocessor()
    
    for split in splits:
        logger.info(f"Processing {split} split...")
        
        input_dir = os.path.join(raw_data_root, split)
        magno_dir = os.path.join(preprocessed_root, 'magno_images', split)
        lines_dir = os.path.join(preprocessed_root, 'line_drawings', split)
        color_dir = os.path.join(preprocessed_root, 'color_images', split)
        
        preprocessor.process_dataset(
            input_dir=input_dir,
            output_magno_dir=magno_dir,
            output_lines_dir=lines_dir,
            output_color_dir=color_dir,
            magno_size=magno_size,
            color_size=color_size
        )
        
        logger.info(f"Completed {split} split")