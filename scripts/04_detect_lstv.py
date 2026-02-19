#!/usr/bin/env python3
"""
LSTV Detector - Castellvi Classification (Updated for TotalSpineSeg)

Combines SPINEPS and TotalSpineSeg outputs to detect and classify
lumbosacral transitional vertebrae (LSTV).

TotalSpineSeg Label Map (from their documentation):
- 41-45: L1-L5 vertebrae
- 50: Sacrum
- 91: disc_T12_L1
- 92: disc_L1_L2
- 93: disc_L2_L3
- 94: disc_L3_L4
- 95: disc_L4_L5
- 100: disc_L5_S

Castellvi Classification:
- Type I:  Enlarged transverse process (TP height > 19mm)
- Type II: Incomplete lumbarization/sacralization (extra vertebra L6)
- Type III: Complete lumbarization/sacralization (L5-S fusion - missing disc_L5_S)
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
    """Detect and classify LSTV cases using TotalSpineSeg labels."""

    # SPINEPS costal process labels
    COSTAL_PROCESS_LEFT = 43
    COSTAL_PROCESS_RIGHT = 44

    # TotalSpineSeg lumbar vertebrae labels (from their documentation)
    LUMBAR_LABELS = {
        41: 'L1',
        42: 'L2',
        43: 'L3',
        44: 'L4',
        45: 'L5',
        46: 'L6',  # Only present in lumbarization cases
    }

    # TotalSpineSeg disc labels
    DISC_LABELS = {
        91: 'T12-L1',
        92: 'L1-L2',
        93: 'L2-L3',
        94: 'L3-L4',
        95: 'L4-L5',
        100: 'L5-S',  # Critical for Type III detection
    }

    SACRUM_LABEL = 50

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
                # Find superior-inferior extent (assume axis 2 is S-I in RAS)
                si_coords = np.where(left_mask)[2]
                height_voxels = si_coords.max() - si_coords.min()
                heights['left'] = height_voxels * pixdim[2]
            else:
                heights['left'] = 0.0

            # Measure right TP
            right_mask = (data == self.COSTAL_PROCESS_RIGHT)
            if right_mask.sum() > 0:
                si_coords = np.where(right_mask)[2]
                height_voxels = si_coords.max() - si_coords.min()
                heights['right'] = height_voxels * pixdim[2]
            else:
                heights['right'] = 0.0

            return heights

        except Exception as e:
            logger.warning(f"Could not measure TP height: {e}")
            return {'left': 0.0, 'right': 0.0}

    def count_lumbar_vertebrae(self, labeled_mask_path: Path) -> Dict:
        """
        Count lumbar vertebrae from TotalSpineSeg labeled output.

        Returns dict with:
        - lumbar_count: number of lumbar vertebrae
        - has_l6: boolean indicating L6 presence
        - present_labels: list of present lumbar labels
        """
        try:
            nii = nib.load(labeled_mask_path)
            data = nii.get_fdata().astype(int)

            unique_labels = np.unique(data)
            unique_labels = unique_labels[unique_labels > 0]

            present_lumbar = []
            for label, name in self.LUMBAR_LABELS.items():
                if label in unique_labels:
                    present_lumbar.append((label, name))

            has_l6 = 46 in unique_labels

            return {
                'lumbar_count': len(present_lumbar),
                'has_l6': has_l6,
                'present_labels': present_lumbar,
            }

        except Exception as e:
            logger.warning(f"Could not count vertebrae: {e}")
            return {'lumbar_count': 5, 'has_l6': False, 'present_labels': []}

    def detect_l5s_disc(self, labeled_mask_path: Path) -> Dict:
        """
        Check for L5-S disc from TotalSpineSeg output.

        Returns dict with:
        - has_l5s_disc: boolean indicating if disc_L5_S (label 100) is present
        - present_discs: list of present disc labels
        - missing_l5s: boolean indicating if L5-S disc is specifically missing
        """
        try:
            nii = nib.load(labeled_mask_path)
            data = nii.get_fdata().astype(int)

            unique_labels = np.unique(data)
            unique_labels = unique_labels[unique_labels > 0]

            present_discs = []
            for label, name in self.DISC_LABELS.items():
                if label in unique_labels:
                    present_discs.append((label, name))

            has_l5s_disc = 100 in unique_labels
            has_l4l5_disc = 95 in unique_labels
            has_sacrum = self.SACRUM_LABEL in unique_labels

            # Missing L5-S disc is suspicious if we have L4-L5 disc and sacrum
            missing_l5s = (not has_l5s_disc) and has_l4l5_disc and has_sacrum

            return {
                'has_l5s_disc': has_l5s_disc,
                'present_discs': present_discs,
                'missing_l5s': missing_l5s,
            }

        except Exception as e:
            logger.warning(f"Could not check L5-S disc: {e}")
            return {'has_l5s_disc': True, 'present_discs': [], 'missing_l5s': False}

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
        spineps_semantic = self.spineps_dir / 'segmentations' / study_id / f"{study_id}_seg-spine_msk.nii.gz"
        totalspine_sag = self.totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_labeled.nii.gz"
        totalspine_axial = self.totalspine_dir / study_id / 'axial' / f"{study_id}_axial_labeled.nii.gz"

        # Check Type I: Enlarged transverse process
        type_i_left = False
        type_i_right = False

        if spineps_semantic.exists():
            tp_heights = self.measure_tp_height(spineps_semantic)
            result['details']['tp_height_left_mm'] = round(tp_heights['left'], 2)
            result['details']['tp_height_right_mm'] = round(tp_heights['right'], 2)

            if tp_heights['left'] > self.TP_HEIGHT_THRESHOLD:
                type_i_left = True
            if tp_heights['right'] > self.TP_HEIGHT_THRESHOLD:
                type_i_right = True

        # Check Type II: Extra lumbar vertebra (L6)
        type_ii = False

        if totalspine_sag.exists():
            vertebra_info = self.count_lumbar_vertebrae(totalspine_sag)
            result['details']['lumbar_count'] = vertebra_info['lumbar_count']
            result['details']['has_l6'] = vertebra_info['has_l6']
            result['details']['lumbar_labels'] = [name for _, name in vertebra_info['present_labels']]

            if vertebra_info['has_l6']:
                type_ii = True

        # Check Type III: L5-S fusion (missing disc_L5_S)
        type_iii = False

        # Check sagittal first (primary)
        if totalspine_sag.exists():
            disc_info = self.detect_l5s_disc(totalspine_sag)
            result['details']['has_l5s_disc_sag'] = disc_info['has_l5s_disc']
            result['details']['missing_l5s_sag'] = disc_info['missing_l5s']

            if disc_info['missing_l5s']:
                type_iii = True

        # Verify with axial if available
        if totalspine_axial.exists():
            disc_info_axial = self.detect_l5s_disc(totalspine_axial)
            result['details']['has_l5s_disc_axial'] = disc_info_axial['has_l5s_disc']
            result['details']['missing_l5s_axial'] = disc_info_axial['missing_l5s']

            # Confirm Type III if both sagittal and axial show fusion
            if disc_info_axial['missing_l5s']:
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

        # Find all studies with both SPINEPS and TotalSpineSeg output
        spineps_seg_dir = self.spineps_dir / 'segmentations'
        study_dirs = sorted([d for d in spineps_seg_dir.iterdir() if d.is_dir()])

        results = []
        lstv_count = 0

        logger.info(f"Processing {len(study_dirs)} studies...")

        for study_dir in study_dirs:
            study_id = study_dir.name

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
