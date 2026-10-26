# Traffic Violation MVP

Phát hiện vi phạm giao thông từ video camera: YOLO nhận diện xe → ByteTrack theo dõi từng xe → luật theo vạch/vùng → lưu ảnh bằng chứng vào SQLite → dashboard web để xem trực tiếp và duyệt lỗi.

## Tính năng

| Lỗi | Cách phát hiện | Cấu hình |
|---|---|---|
| Vượt đèn đỏ | Điểm tiếp đất của xe (giữa cạnh dưới khung) cắt vạch dừng ảo, đi từ phía tới, khi đèn đang đỏ | `rules.red_light` |
| Quá tốc độ | Thời gian đi qua giữa 2 vạch có khoảng cách thực đã biết (nội suy thời điểm cắt vạch dưới mức khung hình) | `rules.speeding` |
| Sai làn | Xe nằm trong đa giác làn không cho phép loại xe đó, liên tục N khung hình | `rules.wrong_lane` |

- **Trạng thái đèn**: `roi` đọc màu đèn trong một vùng ảnh (HSV, làm mượt qua nhiều khung); `timer` dùng chu kỳ cố định; `manual` chỉnh tay. Mọi chế độ đều chỉnh tay đè lên được từ dashboard.
- **Bù ngược sáng** (`backlight`): đo mức ngược sáng của mỗi khung (tỉ lệ vùng cháy sáng và vùng tối), nâng vùng tối bằng đường cong tone, nén vùng sáng, rồi tăng tương phản cục bộ (CLAHE) trên kênh độ sáng. Có 3 chế độ `off` / `on` / `auto`; `auto` có ngưỡng trễ nên không bật tắt liên tục. Màu đèn tín hiệu vẫn đọc từ ảnh gốc.
- **Đọc biển số**: khi có vi phạm, tìm biển số trong ảnh xe (YOLO11 tinh chỉnh cho biển số) rồi OCR (fast-plate-ocr). Nếu chưa đọc được thì thử tiếp trên các khung hình sau trong 2 giây và giữ kết quả tin cậy nhất. Kết quả được chuẩn hoá theo định dạng biển Việt Nam (`51A-123.45`, `29-B1 256.62`), sửa ký tự dễ nhầm theo vị trí (O↔0, B↔8…), chọn kiểu ô tô/xe máy theo loại xe đã nhận diện. Độ tin cậy dưới `min_conf` thì bỏ trống để cán bộ nhập tay; biển do cán bộ nhập không bị ghi đè.
- **Bằng chứng**: ảnh toàn cảnh có chú thích và ảnh cắt sát xe, lưu ở `data/evidence/YYYYMMDD/`, thông tin lưu ở `data/violations.db`.
- **Dashboard**: hình trực tiếp (MJPEG), số liệu, nhật ký vi phạm, xác nhận/loại lỗi, nhập biển số, xuất CSV, bấm lên hình để lấy toạ độ cấu hình.

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # chọn đúng CUDA của máy
pip install -r requirements.txt
./models/download.sh     # yolo11n.pt + plate-v1s.pt
./samples/download.sh    # video mẫu cars.mp4 (camera đường đô thị, 640x360)
```

## Chạy

```bash
# Dashboard: http://127.0.0.1:8000
python run.py --config configs/sample.yaml --serve

# Không giao diện: xử lý hết video, ghi video đã chú thích
python run.py --config configs/sample.yaml --out out.mp4

# Camera thật / RTSP
python run.py --config configs/sample.yaml --serve --source rtsp://user:pass@ip:554/stream

# Kiểm tra vạch/vùng cấu hình trên một khung hình (có lưới toạ độ)
python run.py --config configs/sample.yaml --preview preview.jpg

pytest -q

# Đo độ chính xác đọc biển số trên ảnh biển Việt Nam
./eval/vn/download.sh
python eval/run_plates.py eval/vn --labels eval/vn/labels.json
```

## Độ chính xác đọc biển số

Trên 8 ảnh chụp gần có 24 biển Việt Nam (biển dài 1 dòng, biển vuông 2 dòng, biển vàng):

| | Kết quả |
|---|---|
| Biển đọc được bằng mắt | 21/21 đúng hoàn toàn (178/178 ký tự) |
| Biển bị khung cắt mất 1 ký tự | 1, đọc thiếu ký tự |
| Biển mờ không đọc được | 2, cả hai bị loại (không đoán bừa) |

Mẫu còn nhỏ và là ảnh chụp gần. Trên video mẫu 640x360, biển chỉ rộng khoảng 15 px nên **không đọc được biển nào**: cần camera Full HD trở lên hoặc camera chụp cận biển số đặt riêng.

## API

| Method | Đường dẫn | Mô tả |
|---|---|---|
| GET | `/stream.mjpg`, `/snapshot.jpg` | Hình đã chú thích |
| GET | `/api/stats` | FPS, số xe, trạng thái đèn, bù ngược sáng, số lỗi theo loại |
| GET | `/api/violations?type=&status=&limit=` | Danh sách vi phạm |
| PATCH | `/api/violations/{id}` | `{"status": "confirmed"|"rejected"|"pending", "plate": "51A-123.45"}` |
| GET | `/api/violations.csv` | Xuất CSV |
| POST | `/api/light` | `{"state": "red"|"yellow"|"green"|"auto"}` |
| POST | `/api/backlight` | `{"mode": "off"|"on"|"auto"}` |
| GET | `/evidence/...` | Ảnh bằng chứng |

## Cấu hình cho camera mới

1. Chạy `--preview` hoặc mở dashboard, bấm lên hình để lấy toạ độ.
2. Đặt `stop_line` và một `approach_point` ở phía xe đi tới.
3. Đo khoảng cách thực giữa `line_a` và `line_b` ngoài hiện trường. Tốc độ chỉ đúng khi số đo này đúng.
4. Vẽ đa giác cho từng làn và khai báo `allowed`.
5. Nếu thấy được cột đèn, dùng `light.mode: roi` với khung bao cụm đèn.

## Giới hạn hiện tại

- **Biển số** cần camera đủ phân giải. Model phát hiện biển số dùng giấy phép **AGPL-3.0**; dùng thương mại thì phải công khai mã nguồn hoặc tự huấn luyện model thay thế.
- **Mũ bảo hiểm** chưa có, vì model COCO không có lớp này. Cần model riêng được huấn luyện trên dữ liệu xe máy Việt Nam.
- **Đo tốc độ** dùng 2 vạch, chưa hiệu chỉnh phối cảnh (homography). `distance_m` trong cấu hình mẫu là ước lượng từ vạch kẻ đường, chưa đo thực tế.
- **Đèn** trong video mẫu chạy theo `timer` vì video không quay thấy cột đèn, nên lỗi vượt đèn đỏ ở video mẫu chỉ để minh hoạ.
- `yolo11n` là model nhỏ nhất; nhận diện nhầm SUV thành `truck`. Dùng `yolo11s/m` hoặc tinh chỉnh trên dữ liệu Việt Nam để tốt hơn.
- Dashboard chưa có đăng nhập; chỉ nên chạy ở `127.0.0.1` hoặc sau reverse proxy có xác thực.
