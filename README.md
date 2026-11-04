# Traffic Violation MVP

Phát hiện vi phạm giao thông từ video camera: YOLO nhận diện xe → ByteTrack theo dõi từng xe → luật theo vạch/vùng/biển báo → lưu ảnh bằng chứng vào SQLite → dashboard web để xem trực tiếp và duyệt lỗi.

Mục lục: [Tính năng](#tính-năng) · [Cài đặt](#cài-đặt) · [Chạy demo](#chạy-demo) · [Tự làm video demo](#tự-làm-video-demo) · [Cấu hình camera mới](#cấu-hình-cho-camera-mới) · [Huấn luyện model](#huấn-luyện-model) · [Bản Java](#bản-java) · [API](#api) · [Giới hạn](#giới-hạn-hiện-tại)

## Tính năng

| Lỗi | Cách phát hiện | Khóa cấu hình |
|---|---|---|
| Vượt đèn đỏ | Điểm tiếp đất của xe (giữa cạnh dưới khung) cắt vạch dừng ảo, đi từ phía tới, khi đèn đang đỏ | `rules.red_light` |
| Quá tốc độ | Thời gian đi qua giữa 2 vạch có khoảng cách thực đã biết. Giới hạn lấy từ cấu hình hoặc từ số trên biển P.127 đọc được | `rules.speeding` |
| Sai làn | Xe nằm trong đa giác làn không cho phép loại xe đó, liên tục N khung hình | `rules.wrong_lane` |
| Rẽ sai (cấm rẽ trái/phải) | Xe qua vạch vào giao lộ (từ phía tới) rồi qua vạch của hướng bị cấm | `rules.no_turn` |
| Không đội mũ bảo hiểm | Model riêng (helmet / no_helmet) quét xe máy mỗi vài khung hình; ghi lỗi khi đủ số lần quét và tỉ lệ phiếu "không đội mũ" | `helmet`, `rules.no_helmet` |
| Chở quá số người | ≥ 3 người cỡ người lớn trên một xe máy (trẻ em không tính) | `rules.overload` |
| Đè / lấn vạch liền | Điểm tiếp đất cắt đường vạch liền và ở hẳn bên kia vài khung hình | `rules.solid_line` |
| Không nhường người đi bộ | Xe di chuyển trong vạch đi bộ khi có người đi bộ ở gần (cần camera nhìn từ trên xuống) | `rules.no_yield` |
| Đi ngược chiều (P.102) | Xe đi ngược `direction` của vùng một khoảng ≥ `min_dist` px | `rules.wrong_way` |
| Vào đường cấm (P.103…) | Loại xe bị cấm ở trong vùng N khung hình | `rules.no_entry` |
| Dừng/đỗ nơi cấm (P.131) | Đứng yên trong vùng quá `max_stop_seconds` | `rules.no_parking` |
| Đi trên vỉa hè | Chạy dọc vỉa hè ≥ `min_dist` px (băng ngang để vào nhà thì không tính) | `rules.sidewalk` |
| Dừng chắn giao lộ | Đứng yên trong ô giao lộ ≥ 5 s khi đèn đỏ | `rules.block_junction` |

- **Luật theo biển báo** (`require_sign`): model biển báo Việt Nam (29 nhóm biển) chạy vài khung hình một lần; vùng/luật có `require_sign` chỉ được xét khi đang thấy biển tương ứng (giữ `latch_seconds` sau khi biển khuất).
- **Trạng thái đèn**: `roi` đọc màu đèn trong một vùng ảnh (HSV, làm mượt qua nhiều khung); `timer` dùng chu kỳ cố định; `manual` chỉnh tay. Mọi chế độ đều chỉnh tay đè lên được từ dashboard.
- **Bù ngược sáng** (`backlight`): nâng vùng tối, nén vùng sáng, CLAHE trên kênh độ sáng. Chế độ `off` / `on` / `auto` (có ngưỡng trễ). Màu đèn vẫn đọc từ ảnh gốc.
- **Đọc biển số**: YOLO11 tinh chỉnh cho biển số + OCR (fast-plate-ocr), chuẩn hoá theo định dạng biển Việt Nam (`51A-123.45`, `29-B1 256.62`), sửa ký tự dễ nhầm theo vị trí. Độ tin cậy dưới `min_conf` thì để trống cho cán bộ nhập tay.
- **Bằng chứng**: ảnh toàn cảnh có chú thích và ảnh cắt sát xe ở `<data>/evidence/YYYYMMDD/`, thông tin ở `<data>/violations.db`.
- **Dashboard**: hình trực tiếp (MJPEG), số liệu, nhật ký vi phạm, xác nhận/loại lỗi, nhập biển số, xuất CSV, bấm lên hình để lấy toạ độ, chuyển giữa nhiều demo, tải video lên để chạy thử.

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # chọn đúng CUDA của máy
pip install -r requirements.txt
./models/download.sh     # yolo11n.pt + plate-v1s.pt
./samples/download.sh    # video mẫu cars.mp4 (camera đường đô thị, 640x360)
curl -L -o models/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
```

Các model tự huấn luyện (`helmet*.pt`, `vn_signs.pt`, `speed_value.pt`, `lane_icons.pt`) không có link tải: chép từ máy đã train vào `models/`, hoặc train lại theo phần [Huấn luyện model](#huấn-luyện-model).

## Chạy demo

### 1. Demo nhanh với video mẫu

```bash
# Dashboard: http://127.0.0.1:8000
python run.py --config configs/sample.yaml --serve

# Không giao diện: xử lý hết video, ghi video đã chú thích
python run.py --config configs/sample.yaml --out out.mp4 --fast

# Camera thật / RTSP / webcam
python run.py --config configs/sample.yaml --serve --source rtsp://user:pass@ip:554/stream
python run.py --config configs/sample.yaml --serve --source 0
```

Tham số hay dùng của `run.py`:

| Tham số | Ý nghĩa |
|---|---|
| `--config` | file cấu hình YAML của một camera |
| `--demos` | danh sách demo (mỗi demo một nút trên dashboard), thay cho `--config` |
| `--source` | ghi đè nguồn video (file, `rtsp://...`, số webcam) |
| `--data` | thư mục lưu CSDL và ảnh bằng chứng (mặc định `data`) — nên tách riêng mỗi demo |
| `--serve` / `--host` / `--port` | chạy dashboard (mặc định `127.0.0.1:8000`) |
| `--out` | ghi video đã chú thích ra `.mp4` |
| `--fast` | không chờ theo FPS gốc khi đọc file |
| `--max-frames` | dừng sau N khung hình |
| `--preview p.jpg` / `--preview-frame` | vẽ vạch/vùng cấu hình lên một khung hình (có lưới toạ độ) |

### 2. Dashboard nhiều demo (dùng khi trình bày)

`configs/demos.yaml` liệt kê các demo; mỗi demo là một nút trên dashboard, demo đầu tiên chạy khi mở. Chỉ một demo chạy tại một thời điểm. Mục `upload_templates` là bộ luật áp dụng cho video người xem tải lên.

```bash
python run.py --demos configs/demos.yaml --serve --data data_demos_py
# hoặc bản Java (nhanh hơn, có đọc biển số mọi xe)
java/tvd.sh --demos configs/demos.yaml --serve --data data_demos
```

Mở cho máy khác trong mạng LAN xem: thêm `--host 0.0.0.0` (dashboard chưa có đăng nhập, chỉ dùng trong mạng nội bộ).

### 3. Danh sách cấu hình demo

| Cấu hình | Video | Lỗi minh hoạ | Ghi chú |
|---|---|---|---|
| `sample.yaml` | `cars.mp4` (640x360) | đèn đỏ (timer), tốc độ, sai làn | đèn chạy theo chu kỳ, chỉ minh hoạ |
| `lane_demo.yaml` | `cars.mp4` | sai làn | giả định làn phải chỉ dành cho xe lớn |
| `hcm_redlight_real.yaml` | `hcm_redlight_real.mp4` | vượt đèn đỏ **thật** | đèn thật, xe vượt thật |
| `hanoi_redlight_demo.yaml` | `hanoi_redlight.mp4` | đèn đỏ, tốc độ, sai làn xe máy | đèn + vạch vẽ thêm |
| `hanoi_signs_demo.yaml` | `hanoi_signs.mp4` | ngược chiều, vào đường cấm, đỗ sai | biển dán thêm |
| `demo_nguoc_chieu.yaml` | `demo_nguoc_chieu.mp4` | đi ngược chiều (P.102) | biển dán thêm |
| `demo_vao_duong_cam.yaml` | `demo_vao_duong_cam.mp4` | cấm ô tô (P.103a) | biển dán thêm |
| `hanoi_all_demo.yaml` | `hanoi_all.mp4` | gần như tất cả các lỗi | đèn vẽ + 4 biển dán |
| `hanoi_backlit_demo.yaml` | `hanoi_backlit.mp4` | bù ngược sáng tự bật | ngược sáng giả lập |
| `helmet_demo.yaml` | `helmet_test.mp4` | không đội mũ | camera Myanmar |
| `nhatrang_demo.yaml` | `vn_drive_nhatrang.mp4` | chỉ nhận diện biển báo | dashcam |
| `plate_demo.yaml` | `vn_plates_stills.mp4` | chỉ đọc biển số (bản Java) | ảnh tĩnh ghép |
| `upload_generic.yaml` | video tải lên | mũ bảo hiểm, biển báo, biển số | không cần vẽ vùng |

Video gốc Hà Nội / TP.HCM / Nha Trang là video YouTube giấy phép **CC BY**: ghi nguồn khi chiếu (tên tác giả ở đầu mỗi file cấu hình). Các demo có chữ "dán thêm" / "vẽ thêm" là **dàn dựng**: xe thật nhưng đèn hoặc biển được ghép vào, nên lỗi chỉ để minh hoạ.

### 4. Tải video lên từ dashboard

Khi chạy với `--demos`, dashboard có nút tải video. Video được chạy với bộ luật trong `upload_templates` (mặc định `upload_generic.yaml`: mũ bảo hiểm, biển báo, biển số — các luật không cần vẽ vùng). Nếu video quay cùng góc với một demo có sẵn thì chọn luôn bộ luật của demo đó.

### 5. Kiểm thử

```bash
pytest -q
./eval/vn/download.sh && python eval/run_plates.py eval/vn --labels eval/vn/labels.json   # đọc biển số
python eval/eval_backlight.py samples/raw_hanoi_slowtv.mp4 samples/hanoi_backlit.mp4      # bù ngược sáng
python eval/eval_helmet.py models/helmet_vn_v3.pt                                          # mũ bảo hiểm
```

## Tự làm video demo

Khi chỉ có video giao thông thật mà không có đèn/biển cần cho lỗi muốn demo, có thể ghép thêm. Xe vẫn thật, hệ thống vẫn đọc đèn/biển từ hình như với camera thật.

```bash
# Vẽ vạch dừng + cột đèn đổi màu (xanh 12 s, vàng 3 s, đỏ 10 s); --lane-y vẽ thêm làn "XE MAY"
python tools/make_violation_demo.py samples/raw_hanoi_slowtv.mp4 samples/hanoi_redlight.mp4 --seconds 300 \
    --stop-line 1010 860 990 1080 --light-box 1700 120 1790 380 --lane-y 962

# Dán ảnh biển thật lên cột (biển tròn: PNG X Y CỠ; biển chữ nhật: --rect-sign PNG X Y W H)
python tools/make_sign_demo.py samples/raw_hanoi_slowtv.mp4 samples/hanoi_signs.mp4 --seconds 300 \
    --sign samples/signs/p102_cam_nguoc_chieu.png 575 600 90 \
    --sign samples/signs/p103a_cam_oto.png 575 700 90

# Ngược sáng giả lập: 45 s đầu giữ nguyên, sau đó nắng chói tăng dần trong 15 s
python tools/make_backlit.py samples/raw_hanoi_slowtv.mp4 samples/hanoi_backlit.mp4 --seconds 300
```

Ảnh biển có sẵn trong `samples/signs/`. Muốn thêm biển khác: cắt từ bộ dữ liệu biển báo (`datasets/vn_signs`), lưu PNG vuông. Sau khi tạo video: copy một cấu hình gần giống, đổi `source`, `camera_id`, toạ độ (`--preview`), rồi thêm một mục vào `configs/demos.yaml`:

```yaml
  - key: ten_demo            # không dấu, không trùng
    title: Tên hiện trên nút
    config: configs/ten_demo.yaml
    note: Nguồn video, giấy phép, phần nào dàn dựng
```

## Cấu hình cho camera mới

1. Chạy `python run.py --config cfg.yaml --preview p.jpg` hoặc mở dashboard, bấm lên hình để lấy toạ độ (toạ độ theo điểm ảnh của khung hình gốc).
2. **Đèn đỏ**: đặt `stop_line` và một `approach_point` ở phía xe đi tới. Nếu thấy cột đèn: `light.mode: roi` với khung bao cụm đèn; không thấy thì `timer` hoặc `manual`.
3. **Tốc độ**: đo khoảng cách thực giữa `line_a` và `line_b` ngoài hiện trường (`distance_m`). Tốc độ chỉ đúng khi số đo này đúng. Thêm `require_sign: [speed limit]` để chỉ xét khi thấy biển.
4. **Sai làn**: vẽ đa giác từng làn và khai báo `allowed`.
5. **Luật theo vùng** (`wrong_way`, `no_entry`, `no_parking`, `sidewalk`, `block_junction`): vẽ `polygon`, thêm `require_sign` bằng tên lớp biển báo (xem `LABELS_VI` trong `tvd/signs.py`), ví dụ đầy đủ ở `configs/hanoi_all_demo.yaml`.
6. **Vạch liền** (`solid_line.lines[].points`) và **vạch đi bộ** (`no_yield.crosswalks[].polygon`): vẽ polyline/đa giác trên vạch thật.
7. Chạy `--preview` lại để kiểm tra, rồi chạy `--out` trên vài phút video và xem ảnh bằng chứng trước khi dùng.

## Huấn luyện model

Mọi script train đều tự **chờ GPU rảnh** (`tools/gpu_wait.py`: chờ tới khi không còn tiến trình nào chiếm > 1,5 GB VRAM), nên có thể xếp nhiều lần train nối nhau. Ghi log ra file để theo dõi:

```bash
nohup python tools/train_helmet.py ... > runs_helmet_vX.log 2>&1 &
tail -f runs_helmet_vX.log
```

Ultralytics lưu kết quả ở `runs/detect/runs/<project>/<name>/weights/best.pt` (detect) hoặc `runs/classify/runs/<project>/<name>/...` (classify). Sau khi train xong: kiểm tra số liệu, copy `best.pt` vào `models/` với tên mới (đừng ghi đè model đang dùng), sửa đường dẫn model trong cấu hình, và **xuất ONNX nếu dùng bản Java**:

```bash
python tools/export_onnx.py models/<model_moi>.pt    # -> models/onnx/<model_moi>.onnx
```

Tổng quan các model:

| Model | Loại | Dữ liệu | Dùng ở |
|---|---|---|---|
| `yolo11n/s.pt` | detect (COCO) | có sẵn | nhận diện xe, người |
| `plate-v1s.pt` | detect | có sẵn (AGPL-3.0) | biển số |
| `helmet.pt` | detect 2 lớp | HELMET (Myanmar) | mũ bảo hiểm v1 |
| `helmet_vn_v3.pt` | detect 2 lớp | HELMET + ảnh VN tự gán nhãn | mũ bảo hiểm (đang dùng) |
| `vn_signs.pt` | detect 29 lớp | biển báo VN (Kaggle) | biển báo, `require_sign` |
| `speed_value.pt` | classify | ảnh cắt biển P.127 | đọc số trên biển tốc độ |
| `lane_icons.pt` | classify | hình vẽ trên biển R.412 | đọc biển làn theo loại xe (thử nghiệm, chưa nối vào luật) |

### A. Mũ bảo hiểm — bước 1: dữ liệu HELMET (Myanmar)

**HELMET dataset** (Lin, Deng, Albers, Siebert 2020, CC BY 4.0, https://osf.io/4pwj8/): 910 đoạn video 1920x1080 từ camera giao thông Myanmar. Nhãn gộp thành 2 lớp: `0 helmet` (mọi người đều đội) và `1 no_helmet` (ít nhất một người không đội); mỗi hộp bao cả xe máy lẫn người ngồi trên.

```bash
# tải annotation.zip, data_split.csv và một hoặc nhiều part_N.zip từ OSF vào datasets/helmet/, giải nén ảnh vào datasets/helmet/raw/
python tools/prepare_helmet.py --raw datasets/helmet/raw --ann datasets/helmet/ann/annotation \
    --split datasets/helmet/data_split.csv --out datasets/helmet/yolo --every 4
python tools/train_helmet.py --data datasets/helmet/yolo/data.yaml --name helmet_v1
cp runs/detect/runs/helmet/helmet_v1/weights/best.pt models/helmet.pt
python tools/eval_helmet.py --model models/helmet.pt --data datasets/helmet/yolo --split test
```

Model này báo nhầm nhiều ở góc quay Việt Nam (xem [Giới hạn](#giới-hạn-hiện-tại)), nên cần bước 2.

### B. Mũ bảo hiểm — bước 2: thêm dữ liệu Việt Nam (làm cho mỗi góc camera mới)

Đây là quy trình để **train thêm cho trường hợp mới** (góc camera, thành phố, ban đêm, mưa…): gán nhãn bán tự động từ video, trộn với dữ liệu cũ, rồi tinh chỉnh tiếp từ model đang có.

**Bước 1 — đề xuất hộp.** YOLO COCO tìm xe máy + người, gộp mỗi xe với người ngồi trên thành một hộp, cắt ảnh và tạo lưới ảnh đánh số để duyệt:

```bash
python tools/prepare_vn_helmet.py propose samples/<video>.mp4 --out datasets/vn_helmet_<ten> \
    --split train --start 0 --end 240 --every 2
python tools/prepare_vn_helmet.py propose samples/<video>.mp4 --out datasets/vn_helmet_<ten> \
    --split val --start 240 --end 300 --every 2      # val lấy đoạn thời gian KHÁC train
```

`--every` là số giây giữa 2 khung lấy mẫu; `--min-width` bỏ xe quá nhỏ (mặc định 60 px).

**Bước 2 — gán nhãn.** Mở các ảnh `datasets/vn_helmet_<ten>/review/grid_*.jpg`, viết file nhãn, mỗi dòng một số hộp:

```text
12 n      # không đội mũ (ít nhất một người)
13 h      # đội mũ (hộp không ghi mặc định là h)
27 x      # không rõ: hộp bị tô xám, không dùng làm mẫu dương hay nền
```

```bash
python tools/prepare_vn_helmet.py write --out datasets/vn_helmet_<ten> --split train --labels review_train.txt
python tools/prepare_vn_helmet.py write --out datasets/vn_helmet_<ten> --split val --labels review_val.txt
```

Nên gán ít nhất vài chục hộp `n`; người không đội mũ thường hiếm, cần chọn đoạn video có nhiều.

**Bước 3 — trộn dữ liệu.** Tạo file danh sách ảnh trộn dữ liệu cũ + mới (để model không quên góc cũ) và một file yaml, ví dụ `datasets/vn_helmet/v2_mix.yaml`:

```yaml
train: /đường/dẫn/tuyệt/đối/datasets/vn_helmet/v2_train.txt   # mỗi dòng một ảnh; có thể lặp ảnh VN 3 lần để tăng trọng số
val:   /đường/dẫn/tuyệt/đối/datasets/vn_helmet/v2_val.txt
test:  /đường/dẫn/tuyệt/đối/datasets/vn_helmet_rear/images/test   # một bộ KHÔNG dùng để train
names: {0: helmet, 1: no_helmet}
```

```bash
ls $PWD/datasets/helmet/yolo/images/train/*.jpg > datasets/vn_helmet/v2_train.txt
for i in 1 2 3; do ls $PWD/datasets/vn_helmet_<ten>/images/train/*.jpg >> datasets/vn_helmet/v2_train.txt; done
ls $PWD/datasets/helmet/yolo/images/val/*.jpg $PWD/datasets/vn_helmet_<ten>/images/val/*.jpg > datasets/vn_helmet/v2_val.txt
```

**Bước 4 — tinh chỉnh từ model đang có** (learning rate nhỏ, không warmup — đây là cách đã train `helmet_vn_v3`):

```bash
python tools/train_helmet.py --data datasets/vn_helmet/v2_mix.yaml --base models/helmet_vn_v3.pt \
    --name helmet_vn_v4 --epochs 30 --batch 4 --optimizer AdamW --lr0 0.0003 --warmup 0
cp runs/detect/runs/helmet/helmet_vn_v4/weights/best.pt models/helmet_vn_v4.pt
```

Train từ đầu (khi dữ liệu đã đủ nhiều): bỏ `--base` (mặc định `yolo11s.pt`), giữ `--optimizer auto`, `--epochs 60`. GPU 8 GB: `--batch 4` ở `imgsz 1280`.

**Bước 5 — đánh giá và so sánh.** Thêm bộ mới vào `SETS` trong `eval/eval_helmet.py`, rồi:

```bash
python eval/eval_helmet.py models/helmet_vn_v3.pt models/helmet_vn_v4.pt
```

Chỉ số quan trọng nhất là **tỉ lệ báo nhầm "không đội mũ" trên người có đội** (mỗi báo nhầm là một lỗi sai gửi cho dân), không phải mAP. Chỉ thay model trong cấu hình khi model mới không tăng báo nhầm ở bộ cũ. Sau đó chỉnh `rules.no_helmet.min_frames` / `min_share` trên video thật.

### C. Biển báo Việt Nam

Dữ liệu: "Vietnamese Traffic Signs Detection and Recognition" (Kaggle `jaydenguyenx/vietnamese-traffic-signs-detection-and-recognition`; chưa rõ giấy phép, chỉ dùng cho demo): 1170 khung dashcam, 29 nhóm biển, đã ở định dạng YOLO.

```bash
mkdir -p datasets/vn_signs
curl -L -o datasets/vn_signs/archive.zip \
    https://www.kaggle.com/api/v1/datasets/download/jaydenguyenx/vietnamese-traffic-signs-detection-and-recognition
unzip datasets/vn_signs/archive.zip -d datasets/vn_signs/
# viết datasets/vn_signs/data.yaml: path tới thư mục dữ liệu + danh sách names theo đúng thứ tự lớp
python tools/train_signs.py --data datasets/vn_signs/data.yaml
cp runs/detect/runs/signs/vn_signs/weights/best.pt models/vn_signs.pt
```

Biển nhỏ (trung vị ~26 px) nên train ở `imgsz 1600`; `fliplr=0` vì lật ảnh làm đổi nghĩa biển (cấm rẽ trái ↔ phải). **Train tiếp với ảnh mới** (biển chưa có, góc camera cố định): gán nhãn ảnh mới bằng công cụ YOLO bất kỳ (CVAT, Label Studio, Roboflow) với **đúng thứ tự lớp cũ** (lớp mới thêm vào cuối), gộp vào `train/`, rồi tinh chỉnh từ model cũ:

```bash
python tools/train_signs.py --data datasets/vn_signs/data.yaml --model models/vn_signs.pt --name vn_signs_v3 --epochs 40
```

Có thể dùng chính model hiện tại để gán nhãn nháp: `yolo predict model=models/vn_signs.pt source=<ảnh> save_txt=True imgsz=1600`, rồi sửa lại bằng tay.

### D. Số trên biển giới hạn tốc độ (classifier)

Model biển báo chỉ có một lớp "speed limit"; classifier nhỏ đọc số 40/50/60…

```bash
python tools/prepare_speed_signs.py crops --out datasets/speed_signs    # cắt biển + lưới duyệt review/grid_*.jpg
# viết review.txt: "<số> <giá trị>", ví dụ "12 50"; "x" = không đọc được
python tools/prepare_speed_signs.py write --out datasets/speed_signs --labels review.txt
yolo classify train model=yolo11n-cls.pt data=datasets/speed_signs/cls epochs=60 imgsz=64 \
    project=runs/speed name=speed_cls fliplr=0
cp runs/classify/runs/speed/speed_cls/weights/best.pt models/speed_value.pt
```

Cấu hình dùng: `signs.speed_model` (mặc định `models/speed_value.pt`).

### E. Biển làn theo loại xe R.412 (thử nghiệm)

Tách các hình vẽ trắng trên biển xanh rồi phân loại (car, moto, truck, bus, arrow, other):

```bash
python tools/prepare_lane_signs.py icons --out datasets/lane_panels --video samples/vo_van_kiet.mp4
# xem datasets/lane_panels/review_icons/grid_*.jpg, viết icons.txt "<số> <lớp>"
python tools/prepare_lane_signs.py write-icons --out datasets/lane_panels --labels icons.txt
yolo classify train model=yolo11n-cls.pt data=datasets/lane_panels/icons_cls epochs=80 imgsz=64 \
    project=runs/lane_icons name=icons_cls fliplr=0
cp runs/classify/runs/lane_icons/icons_cls/weights/best.pt models/lane_icons.pt
```

`tvd/lanesign.py` (`LaneSignReader`) đọc biển bằng model này nhưng **chưa được nối vào luật sai làn**; hiện `wrong_lane` vẫn dùng `allowed` khai báo tay.

### F. Train cho trường hợp khác — gợi ý

| Muốn cải thiện | Cách làm |
|---|---|
| Nhận diện xe Việt Nam (xe máy dày đặc, xe ba gác, SUV bị gọi `truck`) | Gán nhãn khung hình từ camera thật theo các lớp COCO đang dùng (`car`, `motorcycle`, `bus`, `truck`), tinh chỉnh `yolo11s.pt` với `yolo detect train`; giữ nguyên tên lớp để luật không phải sửa |
| Ban đêm, mưa, ngược sáng | Lấy mẫu khung hình đúng điều kiện đó bằng `prepare_vn_helmet.py propose` (cho mũ) hoặc gán nhãn thêm, trộn vào bộ cũ, tinh chỉnh như bước B4. Đo trước/sau bằng `eval/` |
| Biển số (thay model AGPL) | Gán nhãn hộp biển số trên ảnh camera VN, train `yolo11s.pt` 1 lớp `plate`, đặt vào `plate.detector`; đo bằng `eval/run_plates.py` |
| Chở quá số người | Không cần model riêng; chỉnh `rules.overload` (`adult_ratio`, `min_box_height`, `min_share`) trên video thật |
| Lớp biển báo mới | Thêm vào cuối `names` của `vn_signs`, train tiếp (mục C), thêm tên tiếng Việt vào `LABELS_VI` trong `tvd/signs.py`, dùng trong `require_sign` |

Nguyên tắc chung: val/test phải lấy từ **đoạn video hoặc camera khác** với train (khung hình liền nhau gần như giống hệt nhau, chia ngẫu nhiên sẽ cho số liệu đẹp giả); luôn giữ một bộ test không bao giờ train; so sánh model mới với model cũ trên **mọi** bộ test trước khi thay.

## Bản Java

Thư mục `java/` là bản viết lại bằng Java 21: nhận diện YOLO qua ONNX Runtime (GPU CUDA nếu có, không thì CPU), ByteTrack, đủ các luật, bù ngược sáng, đọc biển số, SQLite, dashboard (Javalin, cùng giao diện và API). Đọc/ghi video bằng `ffmpeg`. Dùng chung file cấu hình YAML với bản Python.

```bash
# cần JDK 21, Maven, ffmpeg; model ONNX xuất từ .pt (kích thước đầu vào động)
python tools/export_onnx.py models/yolo11n.pt models/yolo11s.pt models/plate-v1s.pt \
    models/helmet_vn_v3.pt models/vn_signs.pt models/speed_value.pt
cd java && mvn package && cd ..

java/tvd.sh --config configs/sample.yaml --serve               # dashboard http://127.0.0.1:8000
java/tvd.sh --demos configs/demos.yaml --serve --data data_demos
java/tvd.sh --config configs/sample.yaml --out out.mp4         # không giao diện, ghi video
java/tvd.sh --config configs/plate_demo.yaml --out plates.mp4  # hiện biển số mọi xe (plate.read_all)
java -cp "java/target/tvd.jar:java/target/lib/*" vn.tvd.tools.EvalPlates eval/vn eval/vn/labels.json
```

`java/tvd.sh` tự thêm thư viện CUDA 13 / cuDNN 9 (gói pip `nvidia-*` trong venv) vào `LD_LIBRARY_PATH`; `TVD_DEVICE=cpu` để ép chạy CPU. Sau khi train model mới, **luôn chạy lại `export_onnx.py`** — bản Java đọc `models/onnx/<tên>.onnx` theo tên file `.pt` trong cấu hình.

Đối chiếu với bản Python trên `samples/cars.mp4` (1190 khung hình): cùng 167 xe, 10 lỗi vượt đèn đỏ, 12 lỗi quá tốc độ; hộp nhận diện trùng tới 0,1 px. Đọc biển số 21/21 biển như bản Python. Tốc độ trên RTX 5070 Laptop: khoảng 50 khung hình/giây khi tắt đọc biển số. Chỉ bản Java có `plate.read_all` (đọc biển số mọi xe trên hình).

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

## Giới hạn hiện tại

- **Các demo dàn dựng** (đèn vẽ, biển dán) chỉ để minh hoạ; chỉ `hcm_redlight_real` là lỗi thật.
- **Biển số** cần camera đủ phân giải. Model phát hiện biển số dùng giấy phép **AGPL-3.0**; dùng thương mại thì phải công khai mã nguồn hoặc tự huấn luyện model thay thế.
- **Mũ bảo hiểm**: model v1 (`helmet.pt`, chỉ dữ liệu Myanmar) báo nhầm 56–63% xe mà mọi người đều đội mũ. `helmet_vn_v3.pt` đã thêm ảnh Việt Nam (Hà Nội nhìn ngang, dashcam nhìn từ sau, Ngã 4 Sở, Nha Trang) nhưng số ảnh VN còn ít; cần đo lại bằng `eval/eval_helmet.py` cho từng góc camera trước khi dùng thật, và train thêm theo mục B.
- **Biển báo**: dữ liệu train chưa rõ giấy phép; biển nhỏ/xa dễ bị bỏ sót.
- **Đo tốc độ** dùng 2 vạch, chưa hiệu chỉnh phối cảnh (homography). `distance_m` trong các cấu hình mẫu là ước lượng, chưa đo thực tế.
- **Không nhường người đi bộ** cần camera nhìn từ trên xuống vạch đi bộ; camera nhìn ngang sẽ chồng người và xe.
- `yolo11n` nhận diện nhầm SUV thành `truck`; dùng `yolo11s/m` hoặc tinh chỉnh trên dữ liệu Việt Nam (mục F).
- Dashboard chưa có đăng nhập; chỉ nên chạy ở `127.0.0.1` hoặc sau reverse proxy có xác thực.
