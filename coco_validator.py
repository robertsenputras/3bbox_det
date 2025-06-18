import json
import os
from pycocotools import mask as maskUtils
import numpy as np

def validate_coco_dataset(coco_file):
    print(f"Validating COCO dataset: {coco_file}")
    
    # Load COCO dataset
    with open(coco_file) as json_file:
        coco_data = json.load(json_file)
    
    def validate_basic_structure(data):
        """Validate the basic structure of COCO dataset"""
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            assert key in data, f"Required key '{key}' not found in the COCO dataset"
            assert len(data[key]) > 0, f"Required key '{key}' is empty"
        
        print("✓ Basic structure validation passed")
        return True

    def validate_categories(categories):
        """Validate category entries"""
        required_keys = ["id", "name", "supercategory"]
        category_ids = set()
        
        for cat in categories:
            # Check required keys
            for key in required_keys:
                assert key in cat, f"Category missing required key: {key}"
            
            # Check ID uniqueness
            assert cat['id'] not in category_ids, f"Duplicate category ID found: {cat['id']}"
            category_ids.add(cat['id'])
        
        print(f"✓ Categories validation passed ({len(categories)} categories)")
        return category_ids

    def validate_images(images):
        """Validate image entries"""
        required_keys = ["id", "file_name", "height", "width"]
        image_ids = set()
        
        for img in images:
            # Check required keys
            for key in required_keys:
                assert key in img, f"Image missing required key: {key}"
            
            # Check ID uniqueness
            assert img['id'] not in image_ids, f"Duplicate image ID found: {img['id']}"
            image_ids.add(img['id'])
            
            # Check if image file exists
            img_path = os.path.join(os.path.dirname(coco_file), '..', 'images', 
                                  'train' if 'train' in coco_file else 'val', 
                                  img['file_name'])
            assert os.path.exists(img_path), f"Image file not found: {img_path}"
            
            # Validate dimensions
            assert img['height'] > 0 and img['width'] > 0, f"Invalid dimensions for image {img['id']}"
        
        print(f"✓ Images validation passed ({len(images)} images)")
        return image_ids

    def validate_annotations(annotations, valid_image_ids, valid_category_ids):
        """Validate annotation entries"""
        required_keys = ['id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox', 'iscrowd']
        ann_ids = set()
        
        for ann in annotations:
            # Check required keys
            for key in required_keys:
                assert key in ann, f"Annotation missing required key: {key}"
            
            # Check ID uniqueness
            assert ann['id'] not in ann_ids, f"Duplicate annotation ID found: {ann['id']}"
            ann_ids.add(ann['id'])
            
            # Validate references
            assert ann['image_id'] in valid_image_ids, f"Invalid image_id in annotation: {ann['image_id']}"
            assert ann['category_id'] in valid_category_ids, f"Invalid category_id in annotation: {ann['category_id']}"
            
            # Validate bbox format
            assert len(ann['bbox']) == 4, f"Invalid bbox format in annotation {ann['id']}"
            assert all(isinstance(x, (int, float)) for x in ann['bbox']), "Invalid bbox values"
            
            # Validate segmentation
            if ann['iscrowd']:
                assert isinstance(ann['segmentation'], dict), "RLE segmentation should be a dictionary for iscrowd=1"
                assert 'counts' in ann['segmentation'], "RLE segmentation missing 'counts'"
                assert 'size' in ann['segmentation'], "RLE segmentation missing 'size'"
            else:
                if isinstance(ann['segmentation'], dict):
                    assert 'counts' in ann['segmentation'], "RLE segmentation missing 'counts'"
                    
            # Validate iscrowd
            assert ann['iscrowd'] in [0, 1], f"Invalid iscrowd value in annotation {ann['id']}"
            
            # Validate area
            assert ann['area'] > 0, f"Invalid area in annotation {ann['id']}"
        
        print(f"✓ Annotations validation passed ({len(annotations)} annotations)")
        return True

    try:
        # Run all validations
        validate_basic_structure(coco_data)
        category_ids = validate_categories(coco_data['categories'])
        image_ids = validate_images(coco_data['images'])
        validate_annotations(coco_data['annotations'], image_ids, category_ids)
        
        print("\n✅ All validations passed! The dataset follows COCO format.")
        print(f"Summary:")
        print(f"- Categories: {len(coco_data['categories'])}")
        print(f"- Images: {len(coco_data['images'])}")
        print(f"- Annotations: {len(coco_data['annotations'])}")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Validation failed: {str(e)}")
        return False

if __name__ == '__main__':
    # Validate both train and val sets
    coco_root = "coco_dataset/annotations"
    for split in ['train', 'val']:
        print(f"\nValidating {split} split:")
        coco_file = os.path.join(coco_root, f"instances_{split}.json")
        validate_coco_dataset(coco_file)