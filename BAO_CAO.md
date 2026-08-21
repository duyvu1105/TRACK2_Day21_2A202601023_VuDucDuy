# Báo cáo Lab MLOps - Day 21 (CI/CD cho AI Systems)

## Bước 1 - Thực nghiệm cục bộ với MLflow

- Đã chạy 24 thí nghiệm RandomForest với các bộ siêu tham số khác nhau
  (`n_estimators`: 50-1000, `max_depth`: 3-None, `min_samples_split`: 2/5/10,
  `bootstrap`, `criterion`, `max_features`, `class_weight`), ghi nhận đầy đủ
  `accuracy` và `f1_score` trên MLflow (SQLite).
- Bộ siêu tham số tốt nhất: **n_estimators=300, max_depth=None,
  min_samples_split=2, bootstrap=False** -> accuracy **0.686**, f1 **0.6843**
  trên tập eval (2998 mẫu huấn luyện). Lý do: `max_depth=None` cho phép cây
  khai thác hết cấu trúc dữ liệu; `bootstrap=False` dùng toàn bộ dữ liệu,
  kết hợp với 300 cây đạt độ chính xác cao nhất trong 24 thí nghiệm.

## Bước 2 - CI/CD với GitHub Actions và DVC

- DVC remote: `gs://mlops-lab-duyvu-2026/dvc`; dữ liệu đã push lên GCS.
- Pipeline `mlops.yml` gồm 4 jobs: **Unit Test -> Train -> Eval -> Deploy**,
  xác thực GCP bằng Workload Identity Federation (không cần key file).
- Eval gate: accuracy >= **0.68**.
- Deploy qua OS Login; VM `35.253.215.226:8000` trả `{"status":"ok"}` ở
  `/health` và kết quả dự đoán hợp lệ ở `/predict`.

## Bước 3 - Huấn luyện liên tục

- Chạy `add_new_data.py`: 2998 -> **5996** mẫu, `dvc add` + `dvc push`,
  commit file `.dvc` -> `git push` tự động kích hoạt pipeline.
- Accuracy tăng **0.686 -> 0.760** (f1 0.6843 -> 0.7588): thêm dữ liệu cải
  thiện hiệu quả mô hình; model mới được triển khai tự động lên VM.

## Khó khăn và cách giải quyết

1. **Deploy SSH lỗi xác thực**: chuyển từ SSH key thường sang OS Login kết
   hợp WIF, đăng ký key tạm của CI qua `gcloud compute os-login ssh-keys add`.

## Bonus đã triển khai

- **Bonus 2**: `src/train.py` hỗ trợ `random_forest`, `gradient_boosting` và
  `logistic_regression` qua `model_type` trong `params.yaml`.
- **Bonus 3**: mỗi lần train tạo `outputs/report.txt` với confusion matrix,
  precision và recall theo lớp; workflow upload report cùng `metrics.json`.
- **Bonus 4**: workflow so sánh accuracy với metrics của model đang deploy và
  hủy cập nhật nếu model mới kém hơn.
- **Bonus 5**: ghi phân phối nhãn vào `metrics.json`, đồng thời cảnh báo nếu
  một lớp chiếm dưới 10% dữ liệu train.
- **Bonus 1**: workflow đã hỗ trợ DagsHub qua ba GitHub Secrets
  (`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`,
  `MLFLOW_TRACKING_PASSWORD`); run `a054c35` đã ghi nhận thành công trên
  DagsHub với `accuracy=0.76` và `f1_score=0.7588`.