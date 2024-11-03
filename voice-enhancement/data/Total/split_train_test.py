import os
import shutil
import random


def split_flac_files(source_dir, train_dir, test_dir, train_ratio=0.8):
    """
    Chia các tệp .flac từ source_dir thành hai thư mục train_dir và test_dir với tỷ lệ train_ratio.

    Args:
        source_dir (str): Đường dẫn tới thư mục nguồn chứa các tệp .flac.
        train_dir (str): Đường dẫn tới thư mục đích để lưu các tệp train.
        test_dir (str): Đường dẫn tới thư mục đích để lưu các tệp test.
        train_ratio (float): Tỷ lệ phần trăm tệp sẽ được chia vào thư mục train (mặc định là 0.8).
    """

    # Kiểm tra xem thư mục nguồn có tồn tại không
    if not os.path.exists(source_dir):
        print(f"Thư mục nguồn không tồn tại: {source_dir}")
        return

    # Tạo thư mục train và test nếu chúng chưa tồn tại
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Liệt kê tất cả các tệp .flac trong thư mục nguồn
    all_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.flac')]
    total_files = len(all_files)

    if total_files == 0:
        print("Không tìm thấy tệp .flac nào trong thư mục nguồn.")
        return

    print(f"Tổng số tệp .flac tìm thấy: {total_files}")

    # Trộn ngẫu nhiên danh sách các tệp
    random.shuffle(all_files)

    # Tính toán số lượng tệp cho train và test
    train_count = int(total_files * train_ratio)
    test_count = total_files - train_count

    print(f"Số tệp sẽ được chia vào thư mục train: {train_count}")
    print(f"Số tệp sẽ được chia vào thư mục test: {test_count}")

    # Chia danh sách tệp
    train_files = all_files[:train_count]
    test_files = all_files[train_count:]

    # Hàm sao chép tệp
    def copy_files(file_list, destination_dir):
        for file_name in file_list:
            src_path = os.path.join(source_dir, file_name)
            dest_path = os.path.join(destination_dir, file_name)
            shutil.copy2(src_path, dest_path)
            print(f"Sao chép: {src_path} -> {dest_path}")

    # Sao chép các tệp vào thư mục train
    print("\nBắt đầu sao chép các tệp vào thư mục train...")
    copy_files(train_files, train_dir)

    # Sao chép các tệp vào thư mục test
    print("\nBắt đầu sao chép các tệp vào thư mục test...")
    copy_files(test_files, test_dir)

    print("\nHoàn thành việc chia tệp thành thư mục train và test.")


if __name__ == "__main__":
    source_directory = "/Users/triductran/SpartanDev/my-work/Speech-enhancement/data/Total/clean_voice_audio"
    train_directory = "/Users/triductran/SpartanDev/my-work/Speech-enhancement/data/Train/clean_voice"
    test_directory = "/Users/triductran/SpartanDev/my-work/Speech-enhancement/data/Test/clean_voice"

    split_flac_files(source_directory, train_directory, test_directory, train_ratio=0.8)
