#!/usr/bin/env python3
"""
LSTV Detector - Castellvi Classification

Combines SPINEPS and TotalSpineSeg outputs to detect and classify
lumbosacral transitional vertebrae (LSTV).

Castellvi Classification:
- Type I:  Enlarged transverse process (TP height > 19mm)
- Type II: Incomplete lumbarization/sacralization (pseudarthrosis)
- Type III: Complete lumbarization/sacralization (fusion)
- Type IV: Type II on one side + Type III on other side

Usage:
    python 04_detect_lstv.py \
        --spineps_dir results/spineps \
        --totalspine_dir results/totalspineseg \
        --output_dir results/lstv_detection
"""

import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from typing import Dict, List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LSTVDetector:
    """Detect and classify LSTV cases."""
    
    # Costal process labels in SPINEPS
    COSTAL_PROCESS_LEFT = 43
    COSTAL_PROCESS_RIGHT = 44
    
    # Transverse process height threshold (mm)
    TP_HEIGHT_THRESHOLD = 19.0
    
    def __init__(self, spineps_dir: Path, totalspine_dir: Path):
        self.spineps_dir = spineps_dir
        self.totalspine_dir = totalspine_dir
    
    def measure_tp_height(self, semantic_mask_path: Path) -> Dict[str, float]:
        """
        Measure transverse process height from SPINEPS costal process mask.
        
        Returns dict with 'left' and 'right' TP heights in mm.
        """
        try:
            nii = nib.load(semantic_mask_path)
            data = nii.get_fdata().astype(int)
            
            # Get voxel dimensions
            pixdim = nii.header['pixdim'][1:4]  # mm per voxel
            
            heights = {}
            
            # Measure left TP
            left_mask = (data == self.COSTAL_PROCESS_LEFT)
            if left_mask.sum() > 0:
                # Find superior-inferior extent (assume axis 1 is S-I)
                si_coords = np.where(left_mask)[1]
                height_voxels = si_coords.max() - si_coords.min()
                heights['left'] = height_voxels * pixdim[1]
            else:
                heights['left'] = 0.0
            
            # Measure right TP
            right_mask = (data == self.COSTAL_PROCESS_RIGHT)
            if right_mask.sum() > 0:
                si_coords = np.where(right_mask)[1]
                height_voxels = si_coords.max() - si_coords.min()
                heights['right'] = height_voxels * pixdim[1]
            else:
                heights['right'] = 0.0
            
            return heights
        
        except Exception as e:
            logger.warning(f"Could not measure TP height: {e}")
            return {'left': 0.0, 'right': 0.0}
    
    def count_lumbar_vertebrae(self, totalspine_mask_path: Path) -> int:
        """
        Count lumbar vertebrae from TotalSpineSeg output.
        
        Returns number of lumbar vertebrae (typically 5 or 6).
        """
        try:
            nii = nib.load(totalspine_mask_path)
            data = nii.get_fdata().astype(int)
            
            # TotalSpineSeg labels lumbar vertebrae
            # Need to check their label scheme - this is a placeholder
            # Typically L1-L5 are consecutive labels
            
            unique_labels = np.unique(data)
            unique_labels = unique_labels[unique_labels > 0]
            
            # Count labels that correspond to lumbar vertebrae
            # This is simplified - actual implementation needs TotalSpineSeg's label mapping
            lumbar_count = len([l for l in unique_labels if 20 <= l <= 30])
            
            return lumbar_count
        
        except Exception as e:
            logger.warning(f"Could not count vertebrae: {e}")
            return 5  # Default to normal
    
    def detect_fusion(self, totalspine_axial_path: Path) -> bool:
        """
        Detect L5-S1 fusion from axial T2 TotalSpineSeg output.
        
        Returns True if fusion detected (Type III).
        """
        try:
            nii = nib.load(totalspine_axial_path)
            data = nii.get_fdata().astype(int)
            
            # Check if L5 and S1 labels are merged or have no disc space
            # This is simplified - actual implementation needs:
            # 1. Identify L5 and S1 labels
            # 2. Check if they're touching/merged
            # 3. Verify no disc space between them
            
            # Placeholder logic
            unique_labels = np.unique(data)
            
            # If we see separate L5 and S1 labels, no fusion
            # If we only see merged label, fusion present
            # This needs proper implementation with TotalSpineSeg label scheme
            
            return False  # Placeholder
        
        except Exception as e:
            logger.warning(f"Could not detect fusion: {e}")
            return False
    
    def classify_castellvi(self, study_id: str) -> Dict:
        """
        Classify LSTV according to Castellvi criteria.
        
        Returns dict with classification results.
        """
        result = {
            'study_id': study_id,
            'lstv_detected': False,
            'castellvi_type': None,
            'details': {}
        }
        
        # Get file paths
        spineps_semantic = self.spineps_dir / 'segmentations' / f"{study_id}_seg-spine_msk.nii.gz"
        totalspine_sag = self.totalspine_dir / f"{study_id}_sagittal_vertebrae.nii.gz"
        totalspine_axial = self.totalspine_dir / f"{study_id}_axial_vertebrae.nii.gz"
        
        # Check Type I: Enlarged transverse process
        type_i_left = False
        type_i_right = False
        
        if spineps_semantic.exists():
            tp_heights = self.measure_tp_height(spineps_semantic)
            result['details']['tp_height_left_mm'] = tp_heights['left']
            result['details']['tp_height_right_mm'] = tp_heights['right']
            
            if tp_heights['left'] > self.TP_HEIGHT_THRESHOLD:
                type_i_left = True
            if tp_heights['right'] > self.TP_HEIGHT_THRESHOLD:
                type_i_right = True
        
        # Check Type II: Extra lumbar vertebra (L6)
        type_ii = False
        
        if totalspine_sag.exists():
            lumbar_count = self.count_lumbar_vertebrae(totalspine_sag)
            result['details']['lumbar_count'] = lumbar_count
            
            if lumbar_count == 6:
                type_ii = True
        
        # Check Type III: Fusion
        type_iii = False
        
        if totalspine_axial.exists():
            fusion_detected = self.detect_fusion(totalspine_axial)
            result['details']['fusion_detected'] = fusion_detected
            
            if fusion_detected:
                type_iii = True
        
        # Classify according to Castellvi
        if type_i_left or type_i_right:
            if not (type_ii or type_iii):
                result['lstv_detected'] = True
                result['castellvi_type'] = 'Type I'
                result['details']['subtype'] = 'bilateral' if (type_i_left and type_i_right) else 'unilateral'
        
        if type_ii:
            result['lstv_detected'] = True
            if type_iii:
                result['castellvi_type'] = 'Type IV'
            else:
                result['castellvi_type'] = 'Type II'
        
        if type_iii and not type_ii:
            result['lstv_detected'] = True
            result['castellvi_type'] = 'Type III'
        
        return result
    
    def detect_all_studies(self, output_dir: Path) -> List[Dict]:
        """Run LSTV detection on all available studies."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all studies with SPINEPS output
        spineps_seg_dir = self.spineps_dir / 'segmentations'
        study_files = sorted(spineps_seg_dir.glob("*_seg-spine_msk.nii.gz"))
        
        results = []
        lstv_count = 0
        
        logger.info(f"Processing {len(study_files)} studies...")
        
        for seg_file in study_files:
            study_id = seg_file.name.replace('_seg-spine_msk.nii.gz', '')
            
            logger.info(f"[{study_id}]")
            
            result = self.classify_castellvi(study_id)
            results.append(result)
            
            if result['lstv_detected']:
                lstv_count += 1
                logger.info(f"  ✓ LSTV: {result['castellvi_type']}")
                logger.info(f"    Details: {result['details']}")
            else:
                logger.info(f"  ✗ No LSTV detected")
        
        # Save results
        results_file = output_dir / 'lstv_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary
        summary = {
            'total_studies': len(results),
            'lstv_detected': lstv_count,
            'lstv_rate': lstv_count / len(results) if results else 0,
            'castellvi_breakdown': {
                'Type I': len([r for r in results if r['castellvi_type'] == 'Type I']),
                'Type II': len([r for r in results if r['castellvi_type'] == 'Type II']),
                'Type III': len([r for r in results if r['castellvi_type'] == 'Type III']),
                'Type IV': len([r for r in results if r['castellvi_type'] == 'Type IV']),
            }
        }
        
        summary_file = output_dir / 'lstv_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("\n" + "=" * 70)
        logger.info("LSTV DETECTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total studies:    {summary['total_studies']}")
        logger.info(f"LSTV detected:    {summary['lstv_detected']} ({summary['lstv_rate']:.1%})")
        logger.info("")
        logger.info("Castellvi Breakdown:")
        for type_name, count in summary['castellvi_breakdown'].items():
            if count > 0:
                logger.info(f"  {type_name}: {count}")
        logger.info("")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Summary saved to: {summary_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='LSTV Detection and Castellvi Classification'
    )
    parser.add_argument('--spineps_dir', required=True,
                       help='SPINEPS output directory')
    parser.add_argument('--totalspine_dir', required=True,
                       help='TotalSpineSeg output directory')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for detection results')
    
    args = parser.parse_args()
    
    detector = LSTVDetector(
        Path(args.spineps_dir),
        Path(args.totalspine_dir)
    )
    
    results = detector.detect_all_studies(Path(args.output_dir))
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
