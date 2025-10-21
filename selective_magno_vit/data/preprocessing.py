"""
Preprocessing pipeline for generating line drawings and magno images.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Optional
import shutil

logger = logging.getLogger(__name__)


class InformativeDrawingsPreprocessor:
    """
    Wrapper for the informative-drawings preprocessing pipeline.
    """
    
    def __init__(
        self,
        informative_drawings_path: str = "third_party/informative-drawings",
        model_name: str = "opensketch_style"
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
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Create output directories
        for out_dir in [output_magno_dir, output_lines_dir, output_color_dir]:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing images from {input_dir}")
        logger.info(f"Outputs: magno={output_magno_dir}, lines={output_lines_dir}, color={output_color_dir}")
        logger.info(f"Configuration: magno_size={magno_size}x{magno_size}, color_size={color_size}x{color_size}")
        
        # Build the command
        test_script = self.repo_path / "test_magno.py"
        
        # Check if test_magno.py exists, if not we need to create it
        if not test_script.exists():
            logger.warning(f"test_magno.py not found in {self.repo_path}")
            logger.info("Creating test_magno.py wrapper...")
            self._create_test_magno_script()
        
        cmd = [
            sys.executable,
            str(test_script),
            "--name", self.model_name,
            "--dataroot", str(input_path),
            "--magno_output_dir", output_magno_dir,
            "--line_drawing_output_dir", output_lines_dir,
            "--color_output_dir", output_color_dir,
            "--magno_size", str(magno_size),
            "--color_size", str(color_size),
        ]
        
        if use_gpu:
            cmd.append("--gpu_ids")
            cmd.append("0")
        else:
            cmd.append("--gpu_ids")
            cmd.append("-1")
        
        # Run the preprocessing
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Preprocessing completed successfully")
            logger.debug(f"Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Preprocessing failed with error code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            raise
    
    def _create_test_magno_script(self):
        """
        Create a custom test_magno.py script if it doesn't exist.
        This is a wrapper around the informative-drawings test.py script
        that adds support for magno image output.
        """
        test_magno_content = '''"""
Test script for generating line drawings and magno images.
Extended from the original test.py to support magno processing.
"""

import os
import sys
from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html
from PIL import Image
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np

def create_magno_image(image_path, output_size=64):
    """
    Create a magno-channel-like image using simple preprocessing.
    This is a placeholder - you should replace with your actual magno processing.
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to simulate magno channel
    magno = cv2.GaussianBlur(gray, (5, 5), 2.0)
    
    # Apply edge detection
    edges = cv2.Canny(magno, 50, 150)
    
    # Combine
    magno = cv2.addWeighted(magno, 0.7, edges, 0.3, 0)
    
    # Resize
    magno = cv2.resize(magno, (output_size, output_size))
    
    # Convert to RGB for consistency
    magno_rgb = cv2.cvtColor(magno, cv2.COLOR_GRAY2RGB)
    
    return magno_rgb

if __name__ == '__main__':
    opt = TestOptions().parse()
    
    # Add custom options for our use case
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    
    # Get custom output directories from command line
    magno_output_dir = None
    line_drawing_output_dir = None
    color_output_dir = None
    magno_size = 64
    color_size = 256
    
    for i, arg in enumerate(sys.argv):
        if arg == '--magno_output_dir' and i + 1 < len(sys.argv):
            magno_output_dir = sys.argv[i + 1]
        elif arg == '--line_drawing_output_dir' and i + 1 < len(sys.argv):
            line_drawing_output_dir = sys.argv[i + 1]
        elif arg == '--color_output_dir' and i + 1 < len(sys.argv):
            color_output_dir = sys.argv[i + 1]
        elif arg == '--magno_size' and i + 1 < len(sys.argv):
            magno_size = int(sys.argv[i + 1])
        elif arg == '--color_size' and i + 1 < len(sys.argv):
            color_size = int(sys.argv[i + 1])
    
    # Create output directories
    for out_dir in [magno_output_dir, line_drawing_output_dir, color_output_dir]:
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    
    if opt.eval:
        model.eval()
    
    print(f'Processing {len(dataset)} images...')
    
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        
        # Get the image name
        img_name = Path(img_path[0]).stem
        
        # Get class directory structure
        relative_path = Path(img_path[0]).relative_to(opt.dataroot)
        class_dir = relative_path.parent
        
        print(f'Processing image {i+1}/{len(dataset)}: {img_name}')
        
        # Save line drawing
        if line_drawing_output_dir and 'fake_B' in visuals:
            line_output_class_dir = Path(line_drawing_output_dir) / class_dir
            line_output_class_dir.mkdir(parents=True, exist_ok=True)

            line_tensor = visuals['fake_B']
            line_image = transforms.ToPILImage()(line_tensor.squeeze(0).cpu())
            line_path = line_output_class_dir / f"{img_name}_line.png"
            line_image.save(line_path)
            print(f'  Saved line drawing: {line_path.name} (size: {line_image.size})')
        
        # Save color image
        if color_output_dir:
            color_output_class_dir = Path(color_output_dir) / class_dir
            color_output_class_dir.mkdir(parents=True, exist_ok=True)

            color_image = Image.open(img_path[0]).convert('RGB')
            color_image = color_image.resize((color_size, color_size), Image.LANCZOS)
            color_path = color_output_class_dir / f"{img_name}_color.png"
            color_image.save(color_path)
            print(f'  Saved color image: {color_path.name} (size: {color_image.size})')
        
        # Create and save magno image
        if magno_output_dir:
            magno_output_class_dir = Path(magno_output_dir) / class_dir
            magno_output_class_dir.mkdir(parents=True, exist_ok=True)

            magno_array = create_magno_image(img_path[0], output_size=magno_size)
            magno_image = Image.fromarray(magno_array)
            magno_path = magno_output_class_dir / f"{img_name}_magno.png"
            magno_image.save(magno_path)
            print(f'  Saved magno image: {magno_path.name} (size: {magno_image.size})')
    
    print('Processing complete!')
'''
        
        output_path = self.repo_path / "test_magno.py"
        with open(output_path, 'w') as f:
            f.write(test_magno_content)
        
        logger.info(f"Created test_magno.py at {output_path}")


def preprocess_imagenette(
    raw_data_root: str,
    preprocessed_root: str,
    magno_size: int = 64,
    color_size: int = 256,
    splits: list = None
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