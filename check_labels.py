import os
import json
from pycocotools.coco import COCO
from tqdm import tqdm

def check_coco_dataset(coco_root):
    for split in ['train', 'val']:
        print(f"\nChecking {split} set:")
        
        # Load annotation file
        ann_file = os.path.join(coco_root, 'annotations', f'instances_{split}.json')
        if not os.path.exists(ann_file):
            print(f"❌ Error: Annotation file not found: {ann_file}")
            continue
            
        # Load COCO API
        coco = COCO(ann_file)
        
        # Check categories
        cats = coco.loadCats(coco.getCatIds())
        print(f"\nCategories ({len(cats)}):")
        for cat in cats:
            print(f"- ID: {cat['id']}, Name: {cat['name']}")
        
        # Check images
        img_ids = coco.getImgIds()
        print(f"\nImages: {len(img_ids)}")
        
        # Check annotations
        ann_ids = coco.getAnnIds()
        anns = coco.loadAnns(ann_ids)
        print(f"Annotations: {len(anns)}")
        
        # Check distribution of annotations per image
        img_to_anns = {}
        for ann in anns:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        images_without_anns = set(img_ids) - set(img_to_anns.keys())
        if images_without_anns:
            print(f"\n⚠️ Warning: {len(images_without_anns)} images have no annotations!")
            
        print("\nAnnotations per image distribution:")
        print(f"- Min: {min([len(anns) for anns in img_to_anns.values()])}")
        print(f"- Max: {max([len(anns) for anns in img_to_anns.values()])}")
        print(f"- Avg: {sum([len(anns) for anns in img_to_anns.values()]) / len(img_to_anns):.1f}")
        
        # Verify image files exist
        print("\nChecking image files...")
        missing_images = []
        for img_id in tqdm(img_ids):
            img_info = coco.loadImgs([img_id])[0]
            img_path = os.path.join(coco_root, 'images', split, img_info['file_name'])
            if not os.path.exists(img_path):
                missing_images.append(img_info['file_name'])
        
        if missing_images:
            print(f"\n❌ Error: {len(missing_images)} image files are missing!")
            print("First few missing images:")
            for img in missing_images[:5]:
                print(f"- {img}")
        else:
            print("✅ All image files exist")
            
        # Check annotation format
        print("\nChecking annotation format...")
        invalid_anns = []
        for ann in tqdm(anns):
            # Check required fields
            required_fields = ['segmentation', 'area', 'iscrowd', 'bbox', 'category_id', 'id']
            missing_fields = [field for field in required_fields if field not in ann]
            if missing_fields:
                invalid_anns.append(f"Annotation {ann['id']} missing fields: {missing_fields}")
                continue
                
            # Check bbox format
            if len(ann['bbox']) != 4:
                invalid_anns.append(f"Annotation {ann['id']} has invalid bbox format")
                continue
                
            # Check segmentation format
            if isinstance(ann['segmentation'], dict):
                if 'counts' not in ann['segmentation'] or 'size' not in ann['segmentation']:
                    invalid_anns.append(f"Annotation {ann['id']} has invalid RLE segmentation format")
        
        if invalid_anns:
            print(f"\n❌ Error: Found {len(invalid_anns)} invalid annotations!")
            print("First few issues:")
            for issue in invalid_anns[:5]:
                print(f"- {issue}")
        else:
            print("✅ All annotations have valid format")

if __name__ == '__main__':
    coco_root = 'coco_dataset'
    check_coco_dataset(coco_root) 