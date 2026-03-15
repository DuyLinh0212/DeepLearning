import torch
from dataset import MRDataset, load_data

def test_pipeline():
    print("--- KIỂM TRA MRDATASET (TẬP VALID) ---")
    
    # Khởi tạo dataset cho tập valid
    valid_dataset = MRDataset(
        task='abnormal',           # Có thể đổi thành 'meniscus' hoặc 'abnormal' tùy bạn
        split='valid',
        data_root='./data',   # Trỏ đúng vào thư mục data của bạn
        label_root='./labels', # Thư mục chứa file label csv
        target_slices=32,
        image_size=224,
        augment=False
    )
    
    print(f"\nTổng số mẫu tìm thấy: {len(valid_dataset)}")
    
    if len(valid_dataset) > 0:
        # Lấy thử sample đầu tiên ra để xem
        planes_data, label = valid_dataset[0]
        
        print("\n--- THÔNG TIN TENSOR MẪU ĐẦU TIÊN ---")
        print(f"Label: {label}")
        print(f"Số lượng mặt phẳng (planes): {len(planes_data)} (Axial, Coronal, Sagittal)")
        
        planes_name = ['Axial', 'Coronal', 'Sagittal']
        for i, plane_tensor in enumerate(planes_data):
            print(f"Kích thước tensor {planes_name[i]}: {plane_tensor.shape}")
            print(f"Giá trị Max: {plane_tensor.max():.4f}, Min: {plane_tensor.min():.4f}")
            
        # Kì vọng shape in ra sẽ là: torch.Size([3, 32, 224, 224])
    else:
        print("Không có dữ liệu, hãy kiểm tra lại đường dẫn tới thư mục /data/valid!")

    print("\n--- KIỂM TRA DATALOADER ---")
    try:
        _, valid_loader, _ = load_data(task='abnormal', batch_size=2, target_slices=32, image_size=224)
        batch_data, batch_labels = next(iter(valid_loader))
        print(f"DataLoader hoạt động tốt!")
        print(f"Kích thước Batch Axial: {batch_data[0].shape}") 
        # Kì vọng: torch.Size([2, 3, 32, 224, 224])
    except Exception as e:
        print(f"Lỗi khi chạy DataLoader: {e}")

if __name__ == '__main__':
    test_pipeline()