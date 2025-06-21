import os

train_dir = "data/train"
counts = {
    cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in os.listdir(train_dir)
}

for k in sorted(counts.keys(), key=lambda x: int(x)):
    class_id = int(k) + 1
    total_images = sum(counts.values())
    print(f"Clase {class_id}: {counts[k]} imágenes")
    
print(f"Total de imágenes: {total_images}")
