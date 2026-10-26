# Traffic Violation MVP

Phát hiện vi phạm giao thông từ video camera: YOLO nhận diện xe → ByteTrack theo dõi từng xe → luật theo vạch/vùng → lưu ảnh bằng chứng vào SQLite → dashboard web để xem trực tiếp và duyệt lỗi.

## Tính năng

| Lỗi | Cách phát hiện | Cấu hình |
|---|---|---|
| Vượt đèn đỏ | Điểm tiếp đất của xe (giữa cạnh dưới khung) cắt vạch dừng ảo, đi từ phía tới, khi đèn đang đỏ | `rules.red_light` |
| Quá tốc độ | Thời gian đi qua giữa 2 vạch có khoảng cách thực đã biết (nội suy thời điểm cắt vạch dưới mức khung hình) | `rules.speeding` |
| Sai làn | Xe nằm trong đa giác làn không cho phép loại xe đó, liên tục N khung hình | `rules.wrong_lane` |

| Rẽ sai quy định (cấm rẽ trái/phải) | Xe đi qua vạch vào giao lộ (từ phía tới) rồi đi qua vạch của hướng bị cấm | `rules.no_turn` |
| Không đội mũ bảo hiểm | Model riêng (helmet / no_helmet) quét xe máy mỗi vài khung hình; ghi lỗi khi đủ số khung hình và tỉ lệ phiếu "không đội mũ" | `helmet`, `rules.no_helmet` |

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

## Bản Java

Thư mục `java/` là bản viết lại toàn bộ bằng Java 21: nhận diện YOLO qua ONNX Runtime (GPU CUDA nếu có, không thì CPU), ByteTrack, đủ 5 luật, bù ngược sáng, đọc biển số, SQLite, dashboard (Javalin, cùng giao diện và API). Đọc/ghi video bằng `ffmpeg` (file, RTSP, webcam; xuất MP4 H.264). Dùng chung file cấu hình YAML với bản Python.

```bash
# cần JDK 21, Maven, ffmpeg; model ONNX xuất từ .pt (kích thước đầu vào động)
python tools/export_onnx.py models/yolo11n.pt models/yolo11s.pt models/plate-v1s.pt models/helmet.pt
cd java && mvn package && cd ..

java/tvd.sh --config configs/sample.yaml --serve               # dashboard http://127.0.0.1:8000
java/tvd.sh --config configs/sample.yaml --out out.mp4         # không giao diện, ghi video
java/tvd.sh --config configs/plate_demo.yaml --out plates.mp4  # hiện biển số mọi xe (plate.read_all)
java -cp "java/target/tvd.jar:java/target/lib/*" vn.tvd.tools.EvalPlates eval/vn eval/vn/labels.json
```

`java/tvd.sh` tự thêm thư viện CUDA 13 / cuDNN 9 (gói pip `nvidia-*` trong venv) vào `LD_LIBRARY_PATH`; `TVD_DEVICE=cpu` để ép chạy CPU.

Đối chiếu với bản Python trên `samples/cars.mp4` (1190 khung hình): cùng 167 xe, 10 lỗi vượt đèn đỏ, 12 lỗi quá tốc độ; hộp nhận diện trùng tới 0,1 px. Đọc biển số 21/21 biển (178/178 ký tự) như bản Python. Tốc độ trên RTX 5070 Laptop: khoảng 50 khung hình/giây khi tắt đọc biển số.

Chỉ bản Java có `plate.read_all` (đọc biển số mọi xe trên hình, không chỉ xe vi phạm, để hiện trực tiếp).

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

## Model mũ bảo hiểm

Huấn luyện từ **HELMET dataset** (Lin, Deng, Albers, Siebert 2020, CC BY 4.0, https://osf.io/4pwj8/): 910 đoạn video 1920x1080 từ camera giao thông Myanmar, mỗi xe máy có nhãn người lái và từng người ngồi sau có đội mũ hay không. Nhãn được gộp thành 2 lớp: `helmet` (mọi người đều đội) và `no_helmet` (ít nhất một người không đội). Tập train/val/test giữ nguyên cách chia theo đoạn video của tác giả.

```bash
# tải annotation.zip, data_split.csv và một hoặc nhiều part_N.zip từ OSF vào datasets/helmet/, giải nén ảnh vào datasets/helmet/raw/
python tools/prepare_helmet.py --raw datasets/helmet/raw --ann datasets/helmet/ann/annotation \
    --split datasets/helmet/data_split.csv --out datasets/helmet/yolo --every 4
curl -L -o models/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
python tools/train_helmet.py --data datasets/helmet/yolo/data.yaml --name helmet_v1
cp runs/helmet/helmet_v1/weights/best.pt models/helmet.pt
python tools/eval_helmet.py --model models/helmet.pt --data datasets/helmet/yolo --split test

# chạy thử trên video ghép từ các đoạn test
python run.py --config configs/helmet_demo.yaml --serve
```

## Cấu hình cho camera mới

1. Chạy `--preview` hoặc mở dashboard, bấm lên hình để lấy toạ độ.
2. Đặt `stop_line` và một `approach_point` ở phía xe đi tới.
3. Đo khoảng cách thực giữa `line_a` và `line_b` ngoài hiện trường. Tốc độ chỉ đúng khi số đo này đúng.
4. Vẽ đa giác cho từng làn và khai báo `allowed`.
5. Nếu thấy được cột đèn, dùng `light.mode: roi` với khung bao cụm đèn.

## Giới hạn hiện tại

- **Biển số** cần camera đủ phân giải. Model phát hiện biển số dùng giấy phép **AGPL-3.0**; dùng thương mại thì phải công khai mã nguồn hoặc tự huấn luyện model thay thế.
- **Mũ bảo hiểm chưa dùng được** (tắt trong `configs/sample.yaml`). Model v1 huấn luyện trên 28 đoạn từ một điểm camera ở Yangon, trong đó người đội mũ rất ít. Trên tập test (630 lượt xe): bắt được 62–68% xe có người không đội mũ, nhưng **56–63% xe mà mọi người đều đội mũ bị báo nhầm**, kể cả mũ thấy rõ. Cần huấn luyện lại trên nhiều điểm camera hơn và dữ liệu Việt Nam.
- **Đo tốc độ** dùng 2 vạch, chưa hiệu chỉnh phối cảnh (homography). `distance_m` trong cấu hình mẫu là ước lượng từ vạch kẻ đường, chưa đo thực tế.
- **Đèn** trong video mẫu chạy theo `timer` vì video không quay thấy cột đèn, nên lỗi vượt đèn đỏ ở video mẫu chỉ để minh hoạ.
- `yolo11n` là model nhỏ nhất; nhận diện nhầm SUV thành `truck`. Dùng `yolo11s/m` hoặc tinh chỉnh trên dữ liệu Việt Nam để tốt hơn.
- Dashboard chưa có đăng nhập; chỉ nên chạy ở `127.0.0.1` hoặc sau reverse proxy có xác thực.
